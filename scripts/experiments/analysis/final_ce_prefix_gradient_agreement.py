#!/usr/bin/env python3
"""Measure Final-CE gradient coherence on shared network prefixes.

The local-data-size condition is applied only while adapting a model fork.
Every gradient diagnostic is then evaluated, without an optimizer update, on
the same deterministic official-test batches.  For prefix i and test batch b,

    rho_i = ||sum_b w_b g_{i,b}|| / sum_b w_b ||g_{i,b}|| = U_i / A_i.

The prefixes contain only the shared teacher trunk.  Private BYOT branches and
the final classifier head are excluded:

    B1    = stem + layer1
    B2    = B1 + layer2
    B3    = B2 + layer3
    Final = B3 + layer4
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.resnet_byot import multi_resnet18_kd
from scripts.experiments.analysis.local_data_size_internal_probe import (
    DATASET_NUM_CLASSES,
    atomic_json_dump,
    build_heads,
    class_balanced_subset,
    dataset_targets,
    evaluate_internal_metrics,
    load_datasets,
    load_global_model,
    local_subset_indices,
    seed_everything,
    selected_deltas,
    train_teacher_only,
)


PREFIXES = ("b1", "b2", "b3", "final")
GROUP_ORDER = ("b1", "b2", "b3", "final")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subparser):
        subparser.add_argument("--global_checkpoint", required=True)
        subparser.add_argument(
            "--dataset", choices=tuple(DATASET_NUM_CLASSES), required=True
        )
        subparser.add_argument("--datadir", default="./data")
        subparser.add_argument("--device", default="cuda:0")
        subparser.add_argument("--reference_batch_size", type=int, default=512)
        subparser.add_argument("--num_workers", type=int, default=0)
        subparser.add_argument("--seed", type=int, default=0)
        subparser.add_argument(
            "--test_samples_per_class", type=int, default=0,
            help="0 uses the complete official test set.",
        )

    prepare = subparsers.add_parser(
        "prepare", help="Measure the centralized checkpoint on the common test reference."
    )
    add_common(prepare)
    prepare.add_argument("--output", required=True)

    run = subparsers.add_parser(
        "run", help="Adapt local forks and measure their test-reference gradients."
    )
    add_common(run)
    run.add_argument("--global_reference", required=True)
    run.add_argument("--probe_checkpoint", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--sample_size", type=int, required=True)
    run.add_argument("--clients", type=int, default=10)
    run.add_argument("--train_budget", choices=("steps", "epochs"), default="steps")
    run.add_argument("--local_steps", type=int, default=100)
    run.add_argument("--local_epochs", type=int, default=5)
    run.add_argument("--local_batch_size", type=int, default=50)
    run.add_argument("--lr", type=float, default=0.01)
    run.add_argument("--momentum", type=float, default=0.9)
    run.add_argument("--weight_decay", type=float, default=1e-5)
    run.add_argument(
        "--local_objective",
        choices=("teacher_ce", "kd_only", "blend", "adaptive_kd"),
        default="teacher_ce",
        help=(
            "Local adaptation loss. teacher_ce reproduces the original diagnostic; "
            "kd_only adds detached-final logit KD and feature imitation; blend also "
            "adds branch-label CE using the original BYOT convex mixture; "
            "adaptive_kd uses the selected client-adaptive KD-only rule."
        ),
    )
    run.add_argument(
        "--sd_alpha", type=float, default=1.0,
        help="KD coefficient lambda for kd_only, or KD mixture alpha for blend.",
    )
    run.add_argument(
        "--sd_beta", type=float, default=0.01,
        help="Coefficient on the sum of B1/B2/B3-to-final feature MSE losses.",
    )
    run.add_argument(
        "--sd_temperature", type=float, default=0.5,
        help="Common teacher/student KD temperature; KL is scaled by T^2.",
    )
    run.add_argument(
        "--sd_branch_reduction", choices=("sum", "mean"), default="sum",
        help="Reduction across the three active branch losses.",
    )
    run.add_argument("--adaptive_lambda_max", type=float, default=1.0)
    run.add_argument(
        "--adaptive_round_scale", type=float, default=1.0,
        help=(
            "Resolved round-schedule multiplier in [0,1]. A centralized "
            "checkpoint has no FL round, so 1.0 represents post-warm-up use."
        ),
    )
    run.add_argument("--adaptive_proxy_temperature", type=float, default=1.0)
    run.add_argument("--adaptive_reliability_power", type=float, default=1.0)
    run.add_argument("--adaptive_skew_power", type=float, default=2.0)
    run.add_argument("--adaptive_soft_tau", type=float, default=0.85)
    run.add_argument("--adaptive_soft_temperature", type=float, default=0.05)
    return parser.parse_args()


def shared_trunk_group(parameter_name):
    """Map a parameter to the incremental shared-trunk group it belongs to."""
    if parameter_name.startswith(("conv1.", "bn1.", "layer1.")):
        return "b1"
    if parameter_name.startswith("layer2."):
        return "b2"
    if parameter_name.startswith("layer3."):
        return "b3"
    if parameter_name.startswith("layer4."):
        return "final"
    return None


def shared_trunk_parameters(model):
    selected = []
    group_counts = {group: 0 for group in GROUP_ORDER}
    group_numel = {group: 0 for group in GROUP_ORDER}
    for name, parameter in model.named_parameters():
        group = shared_trunk_group(name)
        if group is None:
            continue
        if not parameter.requires_grad:
            raise ValueError(f"Shared parameter unexpectedly has gradients disabled: {name}")
        selected.append((name, parameter, group))
        group_counts[group] += 1
        group_numel[group] += int(parameter.numel())
    if not selected or any(group_counts[group] == 0 for group in GROUP_ORDER):
        raise RuntimeError(
            "Failed to identify every ResNet shared-trunk group: "
            f"counts={group_counts}"
        )
    return selected, group_counts, group_numel


def prefix_gradient_agreement(model, dataloader, reference_samples, device):
    """Return sample-weighted U, A, and rho for every cumulative prefix."""
    if reference_samples <= 0:
        raise ValueError("reference_samples must be positive.")
    model.eval()
    selected, group_counts, group_numel = shared_trunk_parameters(model)
    parameters = [parameter for _, parameter, _ in selected]

    # One accumulated mean-gradient tensor per shared parameter is enough to
    # recover U for every cumulative prefix.  This avoids storing one full
    # gradient vector per test batch.
    mean_gradients = [torch.zeros_like(parameter) for parameter in parameters]
    activity = {prefix: 0.0 for prefix in PREFIXES}
    loss_sum = 0.0
    correct = 0
    seen = 0
    batches = 0

    for x, target in dataloader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        batch_samples = int(target.numel())
        if batch_samples == 0:
            continue
        model.zero_grad(set_to_none=True)
        _, logits = model.forward_teacher(x)
        loss = F.cross_entropy(logits, target, reduction="mean")
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=False, create_graph=False,
            allow_unused=True,
        )
        weight = float(batch_samples) / float(reference_samples)

        group_norm_sq = {group: 0.0 for group in GROUP_ORDER}
        with torch.no_grad():
            for index, ((_, parameter, group), gradient) in enumerate(
                zip(selected, gradients)
            ):
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                detached = gradient.detach()
                mean_gradients[index].add_(detached, alpha=weight)
                group_norm_sq[group] += float(torch.sum(detached * detached).item())

        cumulative_norm_sq = 0.0
        for prefix in PREFIXES:
            cumulative_norm_sq += group_norm_sq[prefix]
            activity[prefix] += weight * max(cumulative_norm_sq, 0.0) ** 0.5

        loss_sum += float(loss.item()) * batch_samples
        correct += int(logits.argmax(dim=1).eq(target).sum().item())
        seen += batch_samples
        batches += 1

    model.zero_grad(set_to_none=True)
    if seen != reference_samples:
        raise RuntimeError(
            f"Reference loader produced {seen} samples, expected {reference_samples}."
        )

    mean_group_norm_sq = {group: 0.0 for group in GROUP_ORDER}
    for (_, _, group), mean_gradient in zip(selected, mean_gradients):
        mean_group_norm_sq[group] += float(
            torch.sum(mean_gradient * mean_gradient).item()
        )

    prefixes = {}
    cumulative_mean_norm_sq = 0.0
    cumulative_parameters = 0
    for prefix in PREFIXES:
        cumulative_mean_norm_sq += mean_group_norm_sq[prefix]
        cumulative_parameters += group_numel[prefix]
        net_u = max(cumulative_mean_norm_sq, 0.0) ** 0.5
        raw_a = float(activity[prefix])
        rho = net_u / raw_a if raw_a > 0.0 else 0.0
        # Numerical accumulation can exceed the triangle-inequality bound by a
        # few ulps. Keep the public ratio in its mathematical [0, 1] range.
        rho = min(1.0, max(0.0, rho))
        prefixes[prefix] = {
            "net_gradient_norm_U": float(net_u),
            "mean_batch_gradient_norm_A": raw_a,
            "agreement_rho": float(rho),
            "cancellation_fraction_1_minus_rho": float(1.0 - rho),
            "parameter_count": int(cumulative_parameters),
        }

    del mean_gradients
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "definition": (
            "Final-CE sample-weighted gradient resultant on cumulative shared-trunk "
            "prefixes; private branch parameters and final classifier fc are excluded"
        ),
        "reference_samples": int(seen),
        "reference_batches": int(batches),
        "test_ce": float(loss_sum / max(seen, 1)),
        "test_accuracy_pct": float(100.0 * correct / max(seen, 1)),
        "prefixes": prefixes,
        "incremental_group_parameter_tensors": group_counts,
        "incremental_group_parameter_count": group_numel,
    }


def metric_deltas(postlocal, global_reference):
    deltas = {"prefixes": {}}
    for prefix in PREFIXES:
        deltas["prefixes"][prefix] = {}
        for key in (
            "net_gradient_norm_U",
            "mean_batch_gradient_norm_A",
            "agreement_rho",
            "cancellation_fraction_1_minus_rho",
        ):
            deltas["prefixes"][prefix][key] = float(
                postlocal["prefixes"][prefix][key]
                - global_reference["prefixes"][prefix][key]
            )
    for key in ("test_ce", "test_accuracy_pct"):
        deltas[key] = float(postlocal[key] - global_reference[key])
    return deltas


def build_reference_loader(args, test_dataset, selection_seed, device):
    reference, count = class_balanced_subset(
        test_dataset,
        args.test_samples_per_class,
        selection_seed,
        DATASET_NUM_CLASSES[args.dataset],
    )
    loader = DataLoader(
        reference,
        batch_size=args.reference_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return loader, count


@torch.no_grad()
def estimate_adaptive_kd_lambda(model, dataloader, args, device):
    """Resolve the selected soft-b client lambda before local optimization."""
    if args.adaptive_lambda_max < 0.0:
        raise ValueError("--adaptive_lambda_max must be non-negative.")
    if not 0.0 <= args.adaptive_round_scale <= 1.0:
        raise ValueError("--adaptive_round_scale must be in [0, 1].")
    if args.adaptive_proxy_temperature <= 0.0:
        raise ValueError("--adaptive_proxy_temperature must be positive.")
    if args.adaptive_reliability_power <= 0.0:
        raise ValueError("--adaptive_reliability_power must be positive.")
    if args.adaptive_skew_power <= 0.0:
        raise ValueError("--adaptive_skew_power must be positive.")
    if args.adaptive_soft_temperature <= 0.0:
        raise ValueError("--adaptive_soft_temperature must be positive.")

    was_training = model.training
    model.eval()
    temperature = float(args.adaptive_proxy_temperature)

    # This deliberately mirrors two separate full-loader proxy passes in
    # ``fedbyot``. The train transform is therefore sampled independently for
    # teacher-label reliability and prediction-entropy reliability.
    label_probability_sum = 0.0
    label_probability_count = 0
    for x, target in dataloader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        _, logits = model.forward_teacher(x)
        probability = F.softmax(logits / temperature, dim=1)
        label_probability_sum += float(
            probability.gather(1, target.view(-1, 1)).sum().item()
        )
        label_probability_count += int(target.numel())

    probability_sum = None
    prediction_count = 0
    for x, _ in dataloader:
        x = x.to(device, non_blocking=True)
        _, logits = model.forward_teacher(x)
        probability = F.softmax(logits / temperature, dim=1)
        batch_sum = probability.sum(dim=0)
        probability_sum = (
            batch_sum if probability_sum is None else probability_sum + batch_sum
        )
        prediction_count += int(probability.size(0))

    if was_training:
        model.train()
    if label_probability_count <= 0 or prediction_count <= 0:
        raise RuntimeError("Adaptive lambda proxy loader produced no samples.")

    teacher_label_probability = min(
        1.0,
        max(0.0, label_probability_sum / float(label_probability_count)),
    )
    teacher_reliability = teacher_label_probability ** float(
        args.adaptive_reliability_power
    )
    mean_probability = probability_sum / float(prediction_count)
    class_count = max(int(mean_probability.numel()), 2)
    prediction_entropy = float(
        (-(mean_probability * mean_probability.clamp_min(1e-12).log()).sum()
         / math.log(class_count)).item()
    )
    prediction_entropy = min(1.0, max(0.0, prediction_entropy))
    powered_skew_reliability = prediction_entropy ** float(
        args.adaptive_skew_power
    )
    logit = max(
        -60.0,
        min(
            60.0,
            (prediction_entropy - float(args.adaptive_soft_tau))
            / float(args.adaptive_soft_temperature),
        ),
    )
    soft_gate = 1.0 / (1.0 + math.exp(-logit))
    skew_scale = (
        (1.0 - soft_gate) * powered_skew_reliability + soft_gate
    )
    round_lambda = float(args.adaptive_lambda_max) * float(
        args.adaptive_round_scale
    )
    effective_lambda = round_lambda * teacher_reliability * skew_scale
    return {
        "lambda_max": float(args.adaptive_lambda_max),
        "round_scale": float(args.adaptive_round_scale),
        "round_lambda": float(round_lambda),
        "teacher_label_probability": float(teacher_label_probability),
        "teacher_reliability": float(teacher_reliability),
        "prediction_entropy_reliability": float(prediction_entropy),
        "powered_skew_reliability": float(powered_skew_reliability),
        "soft_gate": float(soft_gate),
        "skew_scale": float(skew_scale),
        "effective_lambda": float(effective_lambda),
        "proxy_samples_per_pass": int(label_probability_count),
    }


def train_self_distillation(model, dataloader, args, device):
    """Adapt one fork with the repository's fixed-lambda BYOT loss.

    The final logits are detached only when they form the branch KD target;
    Final CE still updates the complete teacher path. All three private
    branches are active and their losses remain attached to the corresponding
    shared prefix, matching ``fedbyot`` with no adaptive filters or proxies.
    """
    if args.local_objective not in ("kd_only", "blend", "adaptive_kd"):
        raise ValueError(
            "train_self_distillation requires local_objective=kd_only, blend, "
            "or adaptive_kd."
        )
    if args.sd_alpha < 0.0:
        raise ValueError("--sd_alpha must be non-negative.")
    if args.local_objective == "blend" and args.sd_alpha > 1.0:
        raise ValueError("--sd_alpha must be in [0, 1] for blend.")
    if args.sd_beta < 0.0:
        raise ValueError("--sd_beta must be non-negative.")
    if args.sd_temperature <= 0.0:
        raise ValueError("--sd_temperature must be positive.")

    adaptive = None
    if args.local_objective == "adaptive_kd":
        adaptive = estimate_adaptive_kd_lambda(model, dataloader, args, device)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    totals = {
        "total": 0.0,
        "final_ce": 0.0,
        "branch_ce": 0.0,
        "branch_kd": 0.0,
        "feature_mse": 0.0,
    }
    total_examples = 0
    completed_steps = 0
    start_time = time.time()
    temperature = float(args.sd_temperature)
    branch_divisor = 3.0 if args.sd_branch_reduction == "mean" else 1.0

    def update(batch):
        nonlocal total_examples, completed_steps
        x, target = batch
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        output, m1, m2, m3, final_fea, f1, f2, f3 = model(x)

        final_ce = F.cross_entropy(output, target)
        branch_ce = sum(
            F.cross_entropy(branch_logits, target)
            for branch_logits in (m1, m2, m3)
        ) / branch_divisor
        with torch.no_grad():
            teacher_prob = F.softmax(output / temperature, dim=1)
        branch_kd = sum(
            F.kl_div(
                F.log_softmax(branch_logits / temperature, dim=1),
                teacher_prob,
                reduction="batchmean",
            ) * (temperature ** 2)
            for branch_logits in (m1, m2, m3)
        ) / branch_divisor
        feature_mse = sum(
            F.mse_loss(branch_feature, final_fea.detach())
            for branch_feature in (f1, f2, f3)
        ) / branch_divisor

        if args.local_objective in ("kd_only", "adaptive_kd"):
            kd_coefficient = (
                adaptive["effective_lambda"]
                if adaptive is not None else float(args.sd_alpha)
            )
            loss = (
                final_ce
                + kd_coefficient * branch_kd
                + float(args.sd_beta) * feature_mse
            )
        else:
            loss = (
                final_ce
                + (1.0 - float(args.sd_alpha)) * branch_ce
                + float(args.sd_alpha) * branch_kd
                + float(args.sd_beta) * feature_mse
            )
        loss.backward()
        optimizer.step()

        batch_examples = int(target.numel())
        for name, value in (
            ("total", loss),
            ("final_ce", final_ce),
            ("branch_ce", branch_ce),
            ("branch_kd", branch_kd),
            ("feature_mse", feature_mse),
        ):
            totals[name] += float(value.detach().item()) * batch_examples
        total_examples += batch_examples
        completed_steps += 1

    if args.train_budget == "epochs":
        for _ in range(args.local_epochs):
            for batch in dataloader:
                update(batch)
    else:
        iterator = iter(dataloader)
        while completed_steps < args.local_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            update(batch)

    model.zero_grad(set_to_none=True)
    result = {
        "loss_per_example": totals["total"] / max(total_examples, 1),
        "final_ce_per_example": totals["final_ce"] / max(total_examples, 1),
        "branch_ce_per_example": totals["branch_ce"] / max(total_examples, 1),
        "branch_kd_per_example": totals["branch_kd"] / max(total_examples, 1),
        "feature_mse_per_example": totals["feature_mse"] / max(total_examples, 1),
        "optimizer_steps": int(completed_steps),
        "examples_processed": int(total_examples),
        "wall_time_seconds": float(time.time() - start_time),
    }
    if adaptive is not None:
        result["adaptive_lambda"] = adaptive
    return result


def prepare_mode(args, device):
    _, test_dataset = load_datasets(args)
    selection_seed = int(args.seed) + 1
    loader, reference_count = build_reference_loader(
        args, test_dataset, selection_seed, device
    )
    model, checkpoint = load_global_model(
        args.global_checkpoint, args.dataset, device
    )
    metrics = prefix_gradient_agreement(
        model, loader, reference_count, device
    )
    payload = {
        "format_version": 1,
        "experiment": "final_ce_prefix_gradient_agreement_reference",
        "dataset": args.dataset,
        "global_checkpoint": os.path.abspath(args.global_checkpoint),
        "global_checkpoint_args": checkpoint.get("args", {}),
        "global_completed_epochs": int(checkpoint.get("completed_epochs", -1)),
        "reference_split": f"official_{args.dataset}_test",
        "reference_test_samples": int(reference_count),
        "reference_batch_size": int(args.reference_batch_size),
        "reference_samples_per_class": int(args.test_samples_per_class),
        "reference_selection_seed": int(selection_seed),
        "model_mode": "eval",
        "optimizer_updates_during_measurement": 0,
        "global_reference_metrics": {
            "prefix_gradient_agreement": metrics,
        },
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def run_mode(args, device):
    train_dataset, test_dataset = load_datasets(args)
    with open(args.global_reference, "r", encoding="utf-8") as handle:
        reference_payload = json.load(handle)
    if reference_payload.get("dataset") != args.dataset:
        raise ValueError(
            f"Reference dataset={reference_payload.get('dataset')} but run dataset={args.dataset}."
        )
    expected_checkpoint = os.path.abspath(args.global_checkpoint)
    recorded_checkpoint = os.path.abspath(reference_payload["global_checkpoint"])
    if expected_checkpoint != recorded_checkpoint:
        raise ValueError(
            f"Reference checkpoint={recorded_checkpoint}, run checkpoint={expected_checkpoint}."
        )
    if int(reference_payload["reference_batch_size"]) != int(args.reference_batch_size):
        raise ValueError("Reference batch-size mismatch.")
    if int(reference_payload["reference_samples_per_class"]) != int(
        args.test_samples_per_class
    ):
        raise ValueError("Reference sample-selection mismatch.")

    loader, reference_count = build_reference_loader(
        args,
        test_dataset,
        int(reference_payload["reference_selection_seed"]),
        device,
    )
    if reference_count != int(reference_payload["reference_test_samples"]):
        raise ValueError("Reference sample-count mismatch.")
    global_metrics = reference_payload["global_reference_metrics"][
        "prefix_gradient_agreement"
    ]

    probe = torch.load(args.probe_checkpoint, map_location="cpu")
    if probe.get("dataset") != args.dataset:
        raise ValueError(
            f"Probe dataset={probe.get('dataset')} but run dataset={args.dataset}."
        )
    if probe.get("metrics") != "logits_cka":
        raise ValueError(
            f"Expected a logits_cka probe, got metrics={probe.get('metrics')}."
        )
    probe_checkpoint = os.path.abspath(probe["global_checkpoint"])
    if probe_checkpoint != expected_checkpoint:
        raise ValueError(
            f"Probe checkpoint={probe_checkpoint}, run checkpoint={expected_checkpoint}."
        )
    if int(probe.get("reference_test_samples", -1)) != reference_count:
        raise ValueError("Probe/reference sample-count mismatch.")
    if int(probe.get("reference_selection_seed", -1)) != int(
        reference_payload["reference_selection_seed"]
    ):
        raise ValueError("Probe/reference sample-selection mismatch.")
    heads = build_heads(
        probe["head_dimensions"], probe["heads_state_dict"],
        DATASET_NUM_CLASSES[args.dataset], device,
    )
    global_internal_metrics = probe["global_reference_metrics"]

    base_model, checkpoint = load_global_model(
        args.global_checkpoint, args.dataset, device
    )
    base_state = {
        key: value.detach().cpu().clone()
        for key, value in base_model.state_dict().items()
    }
    base_model.to("cpu")
    del base_model

    clients = []
    train_targets = dataset_targets(train_dataset)
    for client_id in range(args.clients):
        client_seed = int(args.seed) * 2_000_003 + client_id * 193_939 + 29
        seed_everything(client_seed)
        indices, subset_seed = local_subset_indices(
            len(train_dataset), args.sample_size, args.seed, client_id
        )
        subset = Subset(train_dataset, indices.tolist())
        generator = torch.Generator()
        generator.manual_seed(client_seed)
        local_loader = DataLoader(
            subset,
            batch_size=args.local_batch_size,
            shuffle=True,
            drop_last=(
                args.train_budget == "steps"
                and len(subset) >= args.local_batch_size
            ),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=generator,
        )

        model = multi_resnet18_kd(
            num_classes=DATASET_NUM_CLASSES[args.dataset], in_channels=3
        ).to(device)
        model.load_state_dict(base_state, strict=True)
        if args.local_objective == "teacher_ce":
            training = train_teacher_only(model, local_loader, args, device)
        else:
            training = train_self_distillation(model, local_loader, args, device)
        postlocal_internal = evaluate_internal_metrics(
            model, heads, loader, DATASET_NUM_CLASSES[args.dataset],
            "logits_cka", device,
        )
        postlocal_gradient = prefix_gradient_agreement(
            model, loader, reference_count, device
        )
        internal_deltas = selected_deltas(
            postlocal_internal, global_internal_metrics
        )
        gradient_deltas = metric_deltas(postlocal_gradient, global_metrics)
        postlocal_metrics = dict(postlocal_internal)
        postlocal_metrics["prefix_gradient_agreement"] = postlocal_gradient
        deltas = dict(internal_deltas)
        deltas["prefix_gradient_agreement"] = gradient_deltas

        labels = train_targets[indices]
        counts = np.bincount(
            labels, minlength=DATASET_NUM_CLASSES[args.dataset]
        )
        clients.append({
            "client_id": int(client_id),
            "client_seed": int(client_seed),
            "subset_seed": int(subset_seed),
            "subset_indices_sha256": hashlib.sha256(
                np.ascontiguousarray(indices, dtype=np.int64).tobytes()
            ).hexdigest(),
            "sampling_scheme": "iid_nested_prefix_without_replacement_within_client",
            "local_samples": int(args.sample_size),
            "observed_classes": int(np.count_nonzero(counts)),
            "local_class_count_min": int(counts.min()),
            "local_class_count_max": int(counts.max()),
            "local_class_count_std": float(counts.std(ddof=0)),
            "training": training,
            "postlocal_metrics": postlocal_metrics,
            "delta_from_round_start_global": deltas,
        })
        rhos = postlocal_gradient["prefixes"]
        cosine = postlocal_internal["logits"]["depth_summary"][
            "off_diagonal_centered_cosine_mean"
        ]
        cka = postlocal_internal["cka"]["depth_summary"][
            "off_diagonal_linear_cka_mean"
        ]
        print(
            f"client={client_id:02d} n={args.sample_size} "
            f"steps={training['optimizer_steps']} "
            f"cos={cosine:.4f} cka={cka:.4f} "
            f"rho(b1/b2/b3/final)="
            f"{rhos['b1']['agreement_rho']:.4f}/"
            f"{rhos['b2']['agreement_rho']:.4f}/"
            f"{rhos['b3']['agreement_rho']:.4f}/"
            f"{rhos['final']['agreement_rho']:.4f}",
            flush=True,
        )
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "format_version": 1,
        "experiment": (
            "centralized_checkpoint_local_n_prefix_gradient_agreement"
            if args.local_objective == "teacher_ce" else
            "centralized_checkpoint_local_n_self_distillation"
        ),
        "research_axis": "within_client_across_shared_trunk_prefixes",
        "dataset": args.dataset,
        "num_classes": int(DATASET_NUM_CLASSES[args.dataset]),
        "metrics": "logits_cka_final_ce_prefix_gradient_U_A_rho",
        "global_checkpoint": expected_checkpoint,
        "global_completed_epochs": int(checkpoint.get("completed_epochs", -1)),
        "global_reference": os.path.abspath(args.global_reference),
        "probe_checkpoint": os.path.abspath(args.probe_checkpoint),
        "reference_split": f"official_{args.dataset}_test",
        "reference_test_samples": int(reference_count),
        "reference_batch_size": int(args.reference_batch_size),
        "sample_size": int(args.sample_size),
        "sampling_seed": int(args.seed),
        "clients": int(args.clients),
        "local_training": {
            "objective": args.local_objective,
            "private_branch_forward_or_update": args.local_objective != "teacher_ce",
            "self_distillation": {
                "active_branches": ["b1", "b2", "b3"],
                "private_branch_initialization": (
                    "state stored in the common teacher-only centralized checkpoint; "
                    "these private parameters were not optimized during checkpoint training"
                ),
                "branch_gradient_mode": "attached_to_shared_prefix",
                "final_teacher_kd_target_detached": True,
                "alpha_or_lambda": (
                    None if args.local_objective == "adaptive_kd"
                    else float(args.sd_alpha)
                ),
                "feature_beta": float(args.sd_beta),
                "teacher_temperature": float(args.sd_temperature),
                "student_temperature": float(args.sd_temperature),
                "kd_loss_scale": "temperature_squared",
                "branch_loss_reduction": args.sd_branch_reduction,
                "adaptive_weighting_or_filtering": (
                    args.local_objective == "adaptive_kd"
                ),
                "adaptive_rule": {
                    "name": "selected_soft_b_client_lambda",
                    "objective": "final_ce_plus_adaptive_branch_kd_plus_feature_mse",
                    "branch_ce_in_objective": False,
                    "lambda_max": float(args.adaptive_lambda_max),
                    "resolved_round_scale": float(args.adaptive_round_scale),
                    "round_context": (
                        "centralized checkpoint has no communication round; "
                        "round scale is supplied explicitly"
                    ),
                    "proxy_temperature": float(args.adaptive_proxy_temperature),
                    "teacher_reliability_proxy": "mean_teacher_true_label_probability",
                    "teacher_reliability_power": float(
                        args.adaptive_reliability_power
                    ),
                    "skew_proxy": "normalized_entropy_of_mean_teacher_prediction",
                    "skew_power": float(args.adaptive_skew_power),
                    "skew_correction": "soft_relax",
                    "soft_tau": float(args.adaptive_soft_tau),
                    "soft_temperature": float(args.adaptive_soft_temperature),
                } if args.local_objective == "adaptive_kd" else None,
            } if args.local_objective != "teacher_ce" else None,
            "budget_type": args.train_budget,
            "local_steps": int(args.local_steps),
            "local_epochs": int(args.local_epochs),
            "batch_size": int(args.local_batch_size),
            "lr": float(args.lr),
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
        },
        "measurement": {
            "model_mode": "eval",
            "optimizer_updates": 0,
            "loss": "native final-head cross entropy",
            "internal_metrics": (
                "frozen-probe centered logits, softmax JSD, directional variance, "
                "probe accuracy, and cross-depth linear CKA"
            ),
            "common_reference_batches_across_all_conditions": True,
            "prefixes_exclude_private_branches_and_final_fc": True,
        },
        "global_reference_metrics": {
            "prefix_gradient_agreement": global_metrics,
            "logits": global_internal_metrics["logits"],
            "cka": global_internal_metrics["cka"],
        },
        "client_results": clients,
    }
    atomic_json_dump(result, args.output)
    print(f"saved metrics: {args.output}", flush=True)


def main():
    args = parse_args()
    seed_everything(args.seed)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable.")
    if args.reference_batch_size <= 0:
        raise ValueError("--reference_batch_size must be positive.")
    device = torch.device(args.device)
    if args.mode == "prepare":
        prepare_mode(args, device)
    else:
        run_mode(args, device)


if __name__ == "__main__":
    main()
