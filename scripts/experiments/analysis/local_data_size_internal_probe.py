#!/usr/bin/env python3
"""Controlled within-client representation diagnostic for local data size.

The experiment deliberately excludes every BYOT/private branch objective.  A
post-local model is trained only through the final teacher CE path.  Raw trunk
features at B1/B2/B3/final are then decoded by one fixed set of linear probes
trained on the same round-start global model.

Two modes are provided:

``prepare``
    Fit the shared probes once and save round-start reference metrics.

``run``
    Fork the same global checkpoint into several local models, train each fork
    on a deterministic IID subset of the requested size, and evaluate internal
    feature geometry and depth-to-depth semantic consistency on the full
    official CIFAR-100 test set.
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import get_global_dataset
from models.resnet_byot import multi_resnet18_kd


DEPTHS = ("b1", "b2", "b3", "final")
PAIRS = tuple(
    (DEPTHS[left], DEPTHS[right])
    for left in range(len(DEPTHS))
    for right in range(left + 1, len(DEPTHS))
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subparser):
        subparser.add_argument("--global_checkpoint", required=True)
        subparser.add_argument("--datadir", default="./data")
        subparser.add_argument("--device", default="cuda:0")
        subparser.add_argument("--batch_size", type=int, default=256)
        subparser.add_argument("--num_workers", type=int, default=0)
        subparser.add_argument("--seed", type=int, default=0)

    prepare = subparsers.add_parser("prepare", help="Fit and save the frozen shared probes.")
    add_common(prepare)
    prepare.add_argument("--probe_output", required=True)
    prepare.add_argument("--probe_epochs", type=int, default=30)
    prepare.add_argument("--probe_lr", type=float, default=0.1)
    prepare.add_argument("--probe_weight_decay", type=float, default=5e-4)
    prepare.add_argument(
        "--probe_samples_per_class", type=int, default=500,
        help="Deterministic train samples per class for fitting probes; 0 uses all.",
    )
    prepare.add_argument(
        "--test_samples_per_class", type=int, default=0,
        help="Reference test samples per class; 0 uses the full official test set.",
    )

    run = subparsers.add_parser("run", help="Train and evaluate local-model forks.")
    add_common(run)
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
        "--test_samples_per_class", type=int, default=0,
        help="Reference test samples per class; 0 uses the full official test set.",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json_dump(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, delete=False,
        prefix=f".{output_path.name}.", suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        temporary_path = handle.name
    os.replace(temporary_path, output_path)


def atomic_torch_save(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, delete=False,
        prefix=f".{output_path.name}.", suffix=".tmp",
    ) as handle:
        temporary_path = handle.name
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def dataset_targets(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "target"):
        return np.asarray(dataset.target, dtype=np.int64)
    raise ValueError("Dataset does not expose targets.")


def class_balanced_subset(dataset, samples_per_class, seed):
    if samples_per_class <= 0:
        return dataset, len(dataset)
    labels = dataset_targets(dataset)
    selected = []
    rng = np.random.default_rng(seed)
    for class_id in range(100):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < samples_per_class:
            raise ValueError(
                f"Class {class_id} has {len(indices)} samples, fewer than "
                f"requested {samples_per_class}."
            )
        indices = indices.copy()
        rng.shuffle(indices)
        selected.extend(indices[:samples_per_class].tolist())
    rng.shuffle(selected)
    return Subset(dataset, selected), len(selected)


def load_datasets(args):
    args.dataset = "cifar100"
    args.in_channels = 3
    args.num_classes = 100
    args.cifar100_class_count = 0
    args.cifar100_subset_seed = 0
    train_dataset, _, test_dataset = get_global_dataset(args)
    return train_dataset, test_dataset


def load_global_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("global_model", checkpoint)
    model = multi_resnet18_kd(num_classes=100, in_channels=3)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model, checkpoint


def raw_features(model, x):
    """Return GAP trunk features before private BYOT bottlenecks/heads."""
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.layer1(x)
    b1 = F.adaptive_avg_pool2d(x, 1).flatten(1)
    x = model.layer2(x)
    b2 = F.adaptive_avg_pool2d(x, 1).flatten(1)
    x = model.layer3(x)
    b3 = F.adaptive_avg_pool2d(x, 1).flatten(1)
    x = model.layer4(x)
    final = F.adaptive_avg_pool2d(x, 1).flatten(1)
    return {"b1": b1, "b2": b2, "b3": b3, "final": final}


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    collected = {depth: [] for depth in DEPTHS}
    targets = []
    for x, target in dataloader:
        x = x.to(device, non_blocking=True)
        features = raw_features(model, x)
        for depth in DEPTHS:
            collected[depth].append(features[depth].cpu())
        targets.append(target.long().cpu())
    return {
        depth: torch.cat(chunks, dim=0) for depth, chunks in collected.items()
    }, torch.cat(targets, dim=0)


def fit_linear_probes(features, targets, args, device):
    dimensions = {depth: int(features[depth].size(1)) for depth in DEPTHS}
    heads = nn.ModuleDict({
        depth: nn.Linear(dimensions[depth], 100) for depth in DEPTHS
    }).to(device)
    dataset = TensorDataset(*(features[depth] for depth in DEPTHS), targets)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.SGD(
        heads.parameters(), lr=args.probe_lr, momentum=0.9,
        weight_decay=args.probe_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.probe_epochs)
    )
    for epoch in range(args.probe_epochs):
        heads.train()
        epoch_loss = 0.0
        samples = 0
        for batch in loader:
            feature_batch = {
                depth: batch[index].to(device, non_blocking=True)
                for index, depth in enumerate(DEPTHS)
            }
            target = batch[-1].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = sum(
                F.cross_entropy(heads[depth](feature_batch[depth]), target)
                for depth in DEPTHS
            ) / float(len(DEPTHS))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * int(target.numel())
            samples += int(target.numel())
        scheduler.step()
        print(
            f"probe epoch {epoch + 1:03d}/{args.probe_epochs:03d} "
            f"loss={epoch_loss / max(samples, 1):.6f}",
            flush=True,
        )
    heads.eval()
    for parameter in heads.parameters():
        parameter.requires_grad = False
    return heads, dimensions


def build_heads(dimensions, state, device):
    heads = nn.ModuleDict({
        depth: nn.Linear(int(dimensions[depth]), 100) for depth in DEPTHS
    }).to(device)
    heads.load_state_dict(state, strict=True)
    heads.eval()
    for parameter in heads.parameters():
        parameter.requires_grad = False
    return heads


def js_divergence(left, right, eps=1e-12):
    left = left.clamp_min(eps)
    right = right.clamp_min(eps)
    midpoint = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - midpoint.log())).sum(dim=-1)
        + (right * (right.log() - midpoint.log())).sum(dim=-1)
    )


def feature_geometry(features, targets):
    result = {}
    for depth in DEPTHS:
        raw = features[depth].float()
        raw_norm = torch.linalg.vector_norm(raw, dim=1)
        normalized = F.normalize(raw, dim=1, eps=1e-12)
        class_means = []
        within_per_class = []
        class_counts = []
        for class_id in range(100):
            class_features = normalized[targets == class_id]
            if class_features.numel() == 0:
                raise ValueError(f"Reference set has no samples for class {class_id}.")
            class_mean = class_features.mean(dim=0)
            class_means.append(class_mean)
            within_per_class.append(
                ((class_features - class_mean) ** 2).sum(dim=1).mean()
            )
            class_counts.append(int(class_features.size(0)))
        class_means = torch.stack(class_means, dim=0)
        global_mean = class_means.mean(dim=0)
        within = torch.stack(within_per_class).mean()
        between = ((class_means - global_mean) ** 2).sum(dim=1).mean()
        total = ((normalized - global_mean) ** 2).sum(dim=1).mean()
        result[depth] = {
            "feature_dim": int(raw.size(1)),
            "samples": int(raw.size(0)),
            "samples_per_class_min": int(min(class_counts)),
            "samples_per_class_max": int(max(class_counts)),
            "raw_feature_norm_mean": float(raw_norm.mean().item()),
            "raw_feature_norm_std": float(raw_norm.std(unbiased=False).item()),
            "within_class_variance": float(within.item()),
            "between_class_variance": float(between.item()),
            "total_variance": float(total.item()),
            "between_within_ratio": float((between / within.clamp_min(1e-12)).item()),
        }
    return result


def logit_metrics(features, targets, heads, device):
    logits = {}
    with torch.no_grad():
        for depth in DEPTHS:
            chunks = []
            for feature_chunk in features[depth].split(2048):
                chunks.append(heads[depth](feature_chunk.to(device)).cpu())
            logits[depth] = torch.cat(chunks, dim=0).float()

    probabilities = {
        depth: F.softmax(logits[depth], dim=1) for depth in DEPTHS
    }
    centered = {
        depth: logits[depth] - logits[depth].mean(dim=1, keepdim=True)
        for depth in DEPTHS
    }
    directions = {
        depth: F.normalize(centered[depth], dim=1, eps=1e-12)
        for depth in DEPTHS
    }

    sanity = {}
    for depth in DEPTHS:
        probs = probabilities[depth]
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1) / math.log(100)
        centered_norm = torch.linalg.vector_norm(centered[depth], dim=1)
        sanity[depth] = {
            "accuracy_pct": float(
                100.0 * logits[depth].argmax(dim=1).eq(targets).float().mean().item()
            ),
            "nll": float(F.cross_entropy(logits[depth], targets).item()),
            "entropy_normalized_mean": float(entropy.mean().item()),
            "max_probability_mean": float(probs.max(dim=1).values.mean().item()),
            "centered_logit_norm_mean": float(centered_norm.mean().item()),
            "centered_logit_norm_std": float(centered_norm.std(unbiased=False).item()),
            "near_zero_centered_logit_fraction": float(
                centered_norm.lt(1e-8).float().mean().item()
            ),
        }

    pairwise = {}
    pair_cosines = []
    pair_js = []
    for left, right in PAIRS:
        cosine = (directions[left] * directions[right]).sum(dim=1).clamp(-1.0, 1.0)
        divergence = js_divergence(probabilities[left], probabilities[right])
        left_py = probabilities[left].gather(1, targets[:, None]).squeeze(1)
        right_py = probabilities[right].gather(1, targets[:, None]).squeeze(1)
        pair_name = f"{left}-{right}"
        pairwise[pair_name] = {
            "centered_logit_cosine_mean": float(cosine.mean().item()),
            "centered_logit_cosine_std": float(cosine.std(unbiased=False).item()),
            "softmax_js_divergence_mean": float(divergence.mean().item()),
            "softmax_js_divergence_std": float(divergence.std(unbiased=False).item()),
            "absolute_true_label_probability_gap_mean": float(
                (left_py - right_py).abs().mean().item()
            ),
        }
        pair_cosines.append(cosine)
        pair_js.append(divergence)

    stacked_directions = torch.stack(
        [directions[depth] for depth in DEPTHS], dim=1
    )
    mean_direction = stacked_directions.mean(dim=1, keepdim=True)
    directional_variance = (
        (stacked_directions - mean_direction) ** 2
    ).sum(dim=2).mean(dim=1)
    per_sample_pair_cosine = torch.stack(pair_cosines, dim=1).mean(dim=1)
    per_sample_pair_js = torch.stack(pair_js, dim=1).mean(dim=1)
    depth_summary = {
        "directional_variance_mean": float(directional_variance.mean().item()),
        "directional_variance_std": float(directional_variance.std(unbiased=False).item()),
        "off_diagonal_centered_cosine_mean": float(per_sample_pair_cosine.mean().item()),
        "depth_semantic_inconsistency_mean": float((1.0 - per_sample_pair_cosine).mean().item()),
        "off_diagonal_softmax_js_mean": float(per_sample_pair_js.mean().item()),
    }
    return {
        "sanity": sanity,
        "pairwise": pairwise,
        "depth_summary": depth_summary,
    }


def evaluate_internal_metrics(model, heads, dataloader, device):
    features, targets = extract_features(model, dataloader, device)
    result = {
        "feature_geometry": feature_geometry(features, targets),
        "logits": logit_metrics(features, targets, heads, device),
    }
    del features, targets
    return result


def selected_deltas(post, pre):
    feature_delta = {}
    for depth in DEPTHS:
        feature_delta[depth] = {}
        for key in (
            "within_class_variance", "between_class_variance",
            "between_within_ratio", "raw_feature_norm_mean",
        ):
            feature_delta[depth][key] = float(
                post["feature_geometry"][depth][key]
                - pre["feature_geometry"][depth][key]
            )
    pair_delta = {}
    for pair_name in post["logits"]["pairwise"]:
        pair_delta[pair_name] = {}
        for key in (
            "centered_logit_cosine_mean", "softmax_js_divergence_mean",
            "absolute_true_label_probability_gap_mean",
        ):
            pair_delta[pair_name][key] = float(
                post["logits"]["pairwise"][pair_name][key]
                - pre["logits"]["pairwise"][pair_name][key]
            )
    summary_delta = {}
    for key in (
        "directional_variance_mean", "off_diagonal_centered_cosine_mean",
        "depth_semantic_inconsistency_mean", "off_diagonal_softmax_js_mean",
    ):
        summary_delta[key] = float(
            post["logits"]["depth_summary"][key]
            - pre["logits"]["depth_summary"][key]
        )
    return {
        "feature_geometry": feature_delta,
        "pairwise_logits": pair_delta,
        "depth_summary": summary_delta,
    }


def local_subset_indices(dataset_size, sample_size, seed, client_id):
    if sample_size <= 0 or sample_size > dataset_size:
        raise ValueError(
            f"sample_size must be in [1, {dataset_size}], got {sample_size}."
        )
    # The seed does not depend on sample_size, so conditions use nested prefixes
    # for the same (sampling seed, client id) pair.
    subset_seed = int(seed) * 1_000_003 + int(client_id) * 97_409 + 17
    rng = np.random.default_rng(subset_seed)
    return rng.permutation(dataset_size)[:sample_size].astype(np.int64), subset_seed


def train_teacher_only(model, dataloader, args, device):
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    total_loss = 0.0
    total_examples = 0
    completed_steps = 0
    start_time = time.time()

    def update(batch):
        nonlocal total_loss, total_examples, completed_steps
        x, target = batch
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        _, logits = model.forward_teacher(x)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(target.numel())
        total_examples += int(target.numel())
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
    return {
        "loss_per_example": total_loss / max(total_examples, 1),
        "optimizer_steps": int(completed_steps),
        "examples_processed": int(total_examples),
        "wall_time_seconds": float(time.time() - start_time),
    }


def prepare_mode(args, device):
    train_dataset, test_dataset = load_datasets(args)
    probe_train_base = copy.copy(train_dataset)
    probe_train_base.transform = test_dataset.transform
    probe_train, probe_count = class_balanced_subset(
        probe_train_base, args.probe_samples_per_class, args.seed
    )
    reference_test, reference_count = class_balanced_subset(
        test_dataset, args.test_samples_per_class, args.seed + 1
    )
    probe_loader = DataLoader(
        probe_train, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    reference_loader = DataLoader(
        reference_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    model, global_checkpoint = load_global_model(args.global_checkpoint, device)
    features, targets = extract_features(model, probe_loader, device)
    heads, dimensions = fit_linear_probes(features, targets, args, device)
    del features, targets
    baseline_metrics = evaluate_internal_metrics(model, heads, reference_loader, device)
    payload = {
        "format_version": 1,
        "experiment": "within_client_local_data_size_probe",
        "global_checkpoint": os.path.abspath(args.global_checkpoint),
        "global_round": int(global_checkpoint.get("round", -1)),
        "global_checkpoint_args": global_checkpoint.get("args", {}),
        "probe_definition": (
            "depth-specific GAP linear heads fitted once on frozen round-start "
            "global raw trunk features and frozen for every local fork"
        ),
        "head_dimensions": dimensions,
        "heads_state_dict": heads.state_dict(),
        "probe_train_samples": int(probe_count),
        "reference_test_samples": int(reference_count),
        "reference_split": "official_cifar100_test",
        "reference_samples_per_class": int(args.test_samples_per_class),
        "reference_selection_seed": int(args.seed + 1),
        "probe_epochs": int(args.probe_epochs),
        "probe_lr": float(args.probe_lr),
        "probe_weight_decay": float(args.probe_weight_decay),
        "seed": int(args.seed),
        "global_reference_metrics": baseline_metrics,
    }
    atomic_torch_save(payload, args.probe_output)
    print(f"saved shared probe: {args.probe_output}", flush=True)
    print(json.dumps({
        "probe_train_samples": probe_count,
        "reference_test_samples": reference_count,
        "global_probe_accuracy_pct": {
            depth: baseline_metrics["logits"]["sanity"][depth]["accuracy_pct"]
            for depth in DEPTHS
        },
    }, indent=2), flush=True)


def run_mode(args, device):
    train_dataset, test_dataset = load_datasets(args)
    probe = torch.load(args.probe_checkpoint, map_location="cpu")
    expected_reference_samples = int(probe.get("reference_samples_per_class", 0))
    if int(args.test_samples_per_class) != expected_reference_samples:
        raise ValueError(
            "Reference-set mismatch: probe uses test_samples_per_class="
            f"{expected_reference_samples}, but run requested "
            f"{args.test_samples_per_class}."
        )
    reference_test, reference_count = class_balanced_subset(
        test_dataset,
        args.test_samples_per_class,
        int(probe.get("reference_selection_seed", 1)),
    )
    reference_loader = DataLoader(
        reference_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    base_model, global_checkpoint = load_global_model(args.global_checkpoint, device)
    base_state = {
        key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()
    }
    base_model.to("cpu")
    del base_model

    expected_global = os.path.abspath(args.global_checkpoint)
    recorded_global = os.path.abspath(probe["global_checkpoint"])
    if expected_global != recorded_global:
        raise ValueError(
            "Probe/global mismatch: probe was fitted for "
            f"{recorded_global}, but run requested {expected_global}."
        )
    heads = build_heads(
        probe["head_dimensions"], probe["heads_state_dict"], device
    )
    global_reference = probe["global_reference_metrics"]

    clients = []
    for client_id in range(args.clients):
        client_seed = int(args.seed) * 2_000_003 + client_id * 193_939 + 29
        seed_everything(client_seed)
        indices, subset_seed = local_subset_indices(
            len(train_dataset), args.sample_size, args.seed, client_id
        )
        subset = Subset(train_dataset, indices.tolist())
        loader_generator = torch.Generator()
        loader_generator.manual_seed(client_seed)
        local_loader = DataLoader(
            subset, batch_size=args.local_batch_size, shuffle=True,
            drop_last=(
                args.train_budget == "steps"
                and len(subset) >= args.local_batch_size
            ),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda", generator=loader_generator,
        )

        model = multi_resnet18_kd(num_classes=100, in_channels=3).to(device)
        model.load_state_dict(base_state, strict=True)
        training = train_teacher_only(model, local_loader, args, device)
        model.eval()
        post_metrics = evaluate_internal_metrics(model, heads, reference_loader, device)
        deltas = selected_deltas(post_metrics, global_reference)

        local_labels = dataset_targets(train_dataset)[indices]
        counts = np.bincount(local_labels, minlength=100)
        clients.append({
            "client_id": int(client_id),
            "client_seed": int(client_seed),
            "subset_seed": int(subset_seed),
            "sampling_scheme": "iid_nested_prefix_without_replacement_within_client",
            "local_samples": int(args.sample_size),
            "observed_classes": int(np.count_nonzero(counts)),
            "local_class_count_min": int(counts.min()),
            "local_class_count_max": int(counts.max()),
            "local_class_count_std": float(counts.std(ddof=0)),
            "training": training,
            "postlocal_metrics": post_metrics,
            "delta_from_round_start_global": deltas,
        })
        print(
            f"client={client_id:02d} n={args.sample_size} "
            f"steps={training['optimizer_steps']} "
            f"loss={training['loss_per_example']:.4f} "
            f"depth_inconsistency="
            f"{post_metrics['logits']['depth_summary']['depth_semantic_inconsistency_mean']:.4f}",
            flush=True,
        )
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "format_version": 1,
        "experiment": "within_client_local_data_size_probe",
        "research_axis": "within_client_across_depths",
        "global_checkpoint": expected_global,
        "global_round": int(global_checkpoint.get("round", -1)),
        "probe_checkpoint": os.path.abspath(args.probe_checkpoint),
        "reference_split": "official_cifar100_test",
        "reference_test_samples": int(reference_count),
        "sample_size": int(args.sample_size),
        "sampling_seed": int(args.seed),
        "clients": int(args.clients),
        "local_training": {
            "objective": "final_teacher_cross_entropy_only",
            "private_branch_forward_or_update": False,
            "budget_type": args.train_budget,
            "local_steps": int(args.local_steps),
            "local_epochs": int(args.local_epochs),
            "batch_size": int(args.local_batch_size),
            "lr": float(args.lr),
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
        },
        "global_reference_metrics": global_reference,
        "client_results": clients,
    }
    atomic_json_dump(result, args.output)
    print(f"saved metrics: {args.output}", flush=True)


def main():
    args = parse_args()
    seed_everything(args.seed)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable.")
    device = torch.device(args.device)
    if args.mode == "prepare":
        prepare_mode(args, device)
    else:
        run_mode(args, device)


if __name__ == "__main__":
    main()
