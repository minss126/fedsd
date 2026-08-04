#!/usr/bin/env python3
"""Post-hoc representation diagnostic for the IID client-count experiment.

The FL run itself trains only the final classifier.  This script freezes one
aggregated global checkpoint, trains strict linear probes on raw B1/B2/B3/final
features, and then applies those same probes to saved post-local client models.
Consequently, client logit/gradient differences cannot be caused by separately
optimized client probe heads.
"""

import argparse
import json
import math
import os
import random
import sys
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


HEADS = ("b1", "b2", "b3", "final")
PREFIXES = {
    "b1": ("conv1.", "bn1.", "layer1."),
    "b2": ("conv1.", "bn1.", "layer1.", "layer2."),
    "b3": ("conv1.", "bn1.", "layer1.", "layer2.", "layer3."),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global_checkpoint", required=True)
    parser.add_argument("--client_checkpoint_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100"])
    parser.add_argument("--datadir", default="./data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--probe_epochs", type=int, default=30)
    parser.add_argument("--probe_lr", type=float, default=0.1)
    parser.add_argument("--probe_weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--probe_samples_per_class", type=int, default=500,
        help="Class-balanced train samples used for the global linear probes; 0 uses all.",
    )
    parser.add_argument(
        "--common_samples_per_class", type=int, default=20,
        help="Class-balanced test samples used for client logit/gradient comparisons.",
    )
    parser.add_argument(
        "--gradient_batches", type=int, default=2,
        help="Common-reference batches used for post-local gradient diagnostics; 0 uses all.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_targets(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "target"):
        return np.asarray(dataset.target, dtype=np.int64)
    if hasattr(dataset, "samples"):
        return np.asarray([target for _, target in dataset.samples], dtype=np.int64)
    raise ValueError("Cannot read dataset targets.")


def class_balanced_subset(dataset, num_classes, samples_per_class, seed):
    if samples_per_class <= 0:
        return dataset, len(dataset)
    labels = dataset_targets(dataset)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(num_classes):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        selected.extend(indices[:samples_per_class].tolist())
    rng.shuffle(selected)
    return Subset(dataset, selected), len(selected)


def load_model(checkpoint_path, state_key, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get(state_key, checkpoint)
    model = multi_resnet18_kd(num_classes=100, in_channels=3)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def raw_features(model, x):
    """Features before any BYOT-private bottleneck or classifier."""
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
def extract_frozen_features(model, dataloader, device):
    collected = {name: [] for name in HEADS}
    targets = []
    model.eval()
    for x, target in dataloader:
        x = x.to(device, non_blocking=True)
        features = raw_features(model, x)
        for name in HEADS:
            collected[name].append(features[name].detach().cpu())
        targets.append(target.long().cpu())
    return {name: torch.cat(values) for name, values in collected.items()}, torch.cat(targets)


def train_global_linear_probes(train_features, train_targets, args, device):
    dimensions = {name: train_features[name].size(1) for name in HEADS}
    heads = nn.ModuleDict({name: nn.Linear(dimensions[name], 100) for name in HEADS}).to(device)
    dataset = TensorDataset(*(train_features[name] for name in HEADS), train_targets)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.SGD(
        heads.parameters(), lr=args.probe_lr, momentum=0.9,
        weight_decay=args.probe_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.probe_epochs, 1)
    )
    for _ in range(args.probe_epochs):
        heads.train()
        for batch in loader:
            feature_batches = {
                name: batch[index].to(device, non_blocking=True)
                for index, name in enumerate(HEADS)
            }
            target = batch[-1].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = sum(
                F.cross_entropy(heads[name](feature_batches[name]), target)
                for name in HEADS
            ) / float(len(HEADS))
            loss.backward()
            optimizer.step()
        scheduler.step()
    heads.eval()
    for parameter in heads.parameters():
        parameter.requires_grad = False
    return heads


def metric_sums():
    return {"correct": 0, "nll": 0.0, "true_prob": 0.0, "entropy": 0.0, "margin": 0.0, "n": 0}


def update_metrics(sums, logits, target):
    probs = F.softmax(logits, dim=1)
    target_prob = probs.gather(1, target[:, None]).squeeze(1)
    target_logit = logits.gather(1, target[:, None]).squeeze(1)
    competitor = logits.masked_fill(F.one_hot(target, 100).bool(), float("-inf")).max(1).values
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(1) / math.log(100)
    sums["correct"] += int(logits.argmax(1).eq(target).sum().item())
    sums["nll"] += float(F.cross_entropy(logits, target, reduction="sum").item())
    sums["true_prob"] += float(target_prob.sum().item())
    sums["entropy"] += float(entropy.sum().item())
    sums["margin"] += float((target_logit - competitor).sum().item())
    sums["n"] += int(target.numel())


def finish_metrics(sums):
    n = max(int(sums["n"]), 1)
    return {
        "acc": sums["correct"] / n,
        "acc_pct": 100.0 * sums["correct"] / n,
        "nll": sums["nll"] / n,
        "true_label_prob": sums["true_prob"] / n,
        "entropy_norm": sums["entropy"] / n,
        "margin": sums["margin"] / n,
        "samples": sums["n"],
    }


@torch.no_grad()
def evaluate_shared_heads(model, heads, dataloader, device, collect=False):
    sums = {name: metric_sums() for name in HEADS}
    logits_out = {name: [] for name in HEADS}
    probs_out = {name: [] for name in HEADS}
    targets = []
    model.eval()
    heads.eval()
    for x, target in dataloader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        features = raw_features(model, x)
        for name in HEADS:
            logits = heads[name](features[name])
            update_metrics(sums[name], logits, target)
            if collect:
                logits_out[name].append(logits.cpu())
                probs_out[name].append(F.softmax(logits, 1).cpu())
        if collect:
            targets.append(target.cpu())
    result = {name: finish_metrics(sums[name]) for name in HEADS}
    if not collect:
        return result, None
    tensors = {
        "logits": {name: torch.cat(logits_out[name]) for name in HEADS},
        "probs": {name: torch.cat(probs_out[name]) for name in HEADS},
        "targets": torch.cat(targets),
    }
    return result, tensors


def js_divergence(left, right, eps=1e-12):
    left = left.clamp_min(eps)
    right = right.clamp_min(eps)
    midpoint = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - midpoint.log())).sum(-1)
        + (right * (right.log() - midpoint.log())).sum(-1)
    )


def client_prediction_dispersion(client_tensors):
    result = {}
    if not client_tensors:
        return result
    for name in HEADS:
        probs = torch.stack([entry["probs"][name] for entry in client_tensors])
        logits = torch.stack([entry["logits"][name] for entry in client_tensors])
        mean_prob = probs.mean(0)
        mean_logit = logits.mean(0)
        centered = logits - logits.mean(-1, keepdim=True)
        directions = F.normalize(centered, dim=-1, eps=1e-12)
        mean_direction = directions.mean(0)
        result[name] = {
            "client_count": int(probs.size(0)),
            "prob_js_to_mean": float(js_divergence(probs, mean_prob.unsqueeze(0)).mean().item()),
            "prob_l2_to_mean": float(((probs - mean_prob.unsqueeze(0)) ** 2).sum(-1).mean().item()),
            "raw_logit_variance": float(((logits - mean_logit.unsqueeze(0)) ** 2).mean().item()),
            "centered_logit_directional_variance": float(
                ((directions - mean_direction.unsqueeze(0)) ** 2).sum(-1).mean().item()
            ),
        }
    return result


def within_client_depth_metrics(client_tensors):
    result = {name: [] for name in ("b1", "b2", "b3")}
    if not client_tensors:
        return result
    for entry in client_tensors:
        target = entry["targets"]
        final_prob = entry["probs"]["final"]
        final_logit = entry["logits"]["final"]
        final_direction = F.normalize(final_logit - final_logit.mean(-1, keepdim=True), dim=-1)
        for name in result:
            branch_prob = entry["probs"][name]
            branch_logit = entry["logits"][name]
            branch_direction = F.normalize(branch_logit - branch_logit.mean(-1, keepdim=True), dim=-1)
            branch_py = branch_prob.gather(1, target[:, None]).squeeze(1)
            final_py = final_prob.gather(1, target[:, None]).squeeze(1)
            result[name].append({
                "js_to_final": float(js_divergence(branch_prob, final_prob).mean().item()),
                "centered_logit_cosine_to_final": float((branch_direction * final_direction).sum(-1).mean().item()),
                "abs_true_label_prob_gap": float((branch_py - final_py).abs().mean().item()),
            })
    summarized = {}
    for name, rows in result.items():
        summarized[name] = {}
        for key in rows[0]:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            summarized[name][f"{key}_mean"] = float(values.mean())
            summarized[name][f"{key}_std"] = float(values.std(ddof=0))
    return summarized


def prefix_parameters(model, branch_name):
    prefixes = PREFIXES[branch_name]
    return [parameter for name, parameter in model.named_parameters() if name.startswith(prefixes)]


def flattened_gradient(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(parameter) if grad is None else grad).detach().cpu().flatten()
        for parameter, grad in zip(parameters, grads)
    ])


def common_reference_gradients(model, heads, dataloader, device, max_batches):
    sums = {
        branch: {"branch": None, "final": None, "samples": 0}
        for branch in ("b1", "b2", "b3")
    }
    model.eval()
    heads.eval()
    for batch_idx, (x, target) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        features = raw_features(model, x)
        losses = {name: F.cross_entropy(heads[name](features[name]), target) for name in HEADS}
        batch_size = int(target.numel())
        for branch in ("b1", "b2", "b3"):
            parameters = prefix_parameters(model, branch)
            branch_grad = flattened_gradient(losses[branch], parameters)
            final_grad = flattened_gradient(losses["final"], parameters)
            weight = float(batch_size)
            if sums[branch]["branch"] is None:
                sums[branch]["branch"] = weight * branch_grad
                sums[branch]["final"] = weight * final_grad
            else:
                sums[branch]["branch"] += weight * branch_grad
                sums[branch]["final"] += weight * final_grad
            sums[branch]["samples"] += batch_size
    for branch in sums:
        count = max(int(sums[branch]["samples"]), 1)
        sums[branch]["branch"] /= count
        sums[branch]["final"] /= count
    return sums


def cosine_and_ratio(left, right, eps=1e-12):
    left_norm = float(torch.linalg.vector_norm(left).item())
    right_norm = float(torch.linalg.vector_norm(right).item())
    cosine = float(torch.dot(left, right).item() / (left_norm * right_norm + eps))
    cosine = max(-1.0, min(1.0, cosine))
    return cosine, left_norm / max(right_norm, eps)


def summarize_gradient_geometry(client_gradients):
    within = {}
    between = {}
    for branch in ("b1", "b2", "b3"):
        cosines, ratios = [], []
        for gradients in client_gradients:
            cosine, ratio = cosine_and_ratio(
                gradients[branch]["branch"], gradients[branch]["final"]
            )
            cosines.append(cosine)
            ratios.append(ratio)
        within[branch] = {
            "branch_final_cosine_mean": float(np.mean(cosines)),
            "branch_final_cosine_std": float(np.std(cosines)),
            "branch_final_norm_ratio_mean": float(np.mean(ratios)),
            "branch_final_norm_ratio_std": float(np.std(ratios)),
        }
        between[branch] = {}
        for loss_name in ("branch", "final"):
            vectors = torch.stack([gradients[branch][loss_name] for gradients in client_gradients])
            mean = vectors.mean(0)
            mean_norm_sq = float(torch.sum(mean * mean).item())
            centered = vectors - mean.unsqueeze(0)
            divergence = float(torch.sum(centered * centered, dim=1).mean().item())
            vector_norms = torch.linalg.vector_norm(vectors, dim=1)
            mean_norm = math.sqrt(max(mean_norm_sq, 0.0))
            cosines_to_mean = ((vectors @ mean) / (vector_norms * mean_norm + 1e-12)).clamp(-1.0, 1.0)
            between[branch][loss_name] = {
                "client_count": int(vectors.size(0)),
                "divergence": divergence,
                "relative_divergence": divergence / (mean_norm_sq + 1e-12),
                "mean_gradient_norm": float(vector_norms.mean().item()),
                "aggregated_gradient_norm": mean_norm,
                "cosine_to_mean": float(cosines_to_mean.mean().item()),
            }
    return {"within_client_branch_final": within, "between_client": between}


def mean_std_client_accuracy(client_metrics):
    result = {}
    for name in HEADS:
        values = np.asarray([metrics[name]["acc_pct"] for metrics in client_metrics], dtype=np.float64)
        result[name] = {
            "shared_probe_acc_pct_mean": float(values.mean()),
            "shared_probe_acc_pct_std": float(values.std(ddof=0)),
            "shared_probe_acc_pct_min": float(values.min()),
            "shared_probe_acc_pct_max": float(values.max()),
        }
    return result


def global_depth_accuracy_gap(global_metrics):
    final_acc = float(global_metrics["final"]["acc"])
    result = {}
    for name in ("b1", "b2", "b3"):
        branch_acc = float(global_metrics[name]["acc"])
        result[name] = {
            "final_minus_branch_acc_pct": 100.0 * (final_acc - branch_acc),
            "branch_to_final_error_ratio": (1.0 - branch_acc) / max(1.0 - final_acc, 1e-12),
        }
    return result


def main():
    args = parse_args()
    seed_everything(args.seed)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable.")
    device = torch.device(args.device)

    args.in_channels = 3
    args.num_classes = 100
    train_dataset, _, test_dataset = get_global_dataset(args)
    # Probe fitting should be deterministic and should not use crop/flip noise.
    import copy
    probe_train_base = copy.copy(train_dataset)
    if hasattr(probe_train_base, "transform") and hasattr(test_dataset, "transform"):
        probe_train_base.transform = test_dataset.transform
    probe_train, probe_count = class_balanced_subset(
        probe_train_base, 100, args.probe_samples_per_class, args.seed
    )
    common_test, common_count = class_balanced_subset(
        test_dataset, 100, args.common_samples_per_class, args.seed + 1
    )
    train_loader = DataLoader(
        probe_train, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    common_loader = DataLoader(
        common_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    global_model, global_checkpoint = load_model(
        args.global_checkpoint, "global_model", device
    )
    train_features, train_targets = extract_frozen_features(global_model, train_loader, device)
    heads = train_global_linear_probes(train_features, train_targets, args, device)
    del train_features, train_targets
    global_metrics, _ = evaluate_shared_heads(global_model, heads, test_loader, device)

    manifest_path = Path(args.client_checkpoint_dir) / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    client_metrics = []
    client_common_tensors = []
    client_gradients = []
    client_ids = []
    for filename in manifest["files"]:
        checkpoint_path = Path(args.client_checkpoint_dir) / filename
        client_model, client_checkpoint = load_model(checkpoint_path, "client_model", device)
        client_id = int(client_checkpoint["client_id"])
        metrics, _ = evaluate_shared_heads(client_model, heads, test_loader, device)
        _, common_tensors = evaluate_shared_heads(client_model, heads, common_loader, device, collect=True)
        gradients = common_reference_gradients(
            client_model, heads, common_loader, device, args.gradient_batches
        )
        client_ids.append(client_id)
        client_metrics.append(metrics)
        client_common_tensors.append(common_tensors)
        client_gradients.append(gradients)
        client_model.to("cpu")
        del client_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "experiment": "iid_full_participation_client_count_representation_diagnostic",
        "dataset": args.dataset,
        "global_checkpoint": os.path.abspath(args.global_checkpoint),
        "client_checkpoint_dir": os.path.abspath(args.client_checkpoint_dir),
        "n_clients": int(global_checkpoint.get("args", {}).get("n_clients", manifest["n_clients"])),
        "samples_per_client": 50000.0 / float(manifest["n_clients"]),
        "saved_client_ids": client_ids,
        "probe_definition": "strict GAP + linear heads trained only on frozen aggregated-global raw features",
        "probe_train_samples": probe_count,
        "common_reference_samples": common_count,
        "probe_epochs": args.probe_epochs,
        "gradient_batches": args.gradient_batches,
        "global_frozen_probe": global_metrics,
        "global_frozen_probe_depth_gap": global_depth_accuracy_gap(global_metrics),
        "postlocal_client_shared_probe_accuracy": mean_std_client_accuracy(client_metrics),
        "postlocal_client_prediction_dispersion": client_prediction_dispersion(client_common_tensors),
        "postlocal_within_client_depth_gap": within_client_depth_metrics(client_common_tensors),
        "postlocal_gradient_geometry": summarize_gradient_geometry(client_gradients),
        "seed": args.seed,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
