#!/usr/bin/env python3
"""Evaluate hard-label separability of frozen ResNet18-BYOT branch representations.

This is an *analysis-only* central probe.  It never changes the FL checkpoint:
for each selected B1/B2/B3 depth, the checkpoint's trunk through that branch
point is frozen and a fresh copy of the branch's private prediction path is
trained on a class-balanced global reference split.  The probes are evaluated
on the global test split.  This tests whether shallow branch representations
can support hard-label classification; it is not an additional FL method.
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# The script is invoked from scripts/experiments/analysis, so make repository
# imports work regardless of the caller's current directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import get_global_dataset
from models.resnet_byot import multi_resnet18_kd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a final/best BYOT checkpoint.")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--datadir", default="./data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--probe_epochs", type=int, default=30)
    parser.add_argument("--probe_lr", type=float, default=0.05)
    parser.add_argument("--probe_weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--branches", default="1,2,3",
        help="Comma-separated shallow branch depths to probe; supported values: 1,2,3.",
    )
    parser.add_argument(
        "--samples_per_class", type=int, default=500,
        help="Maximum class-balanced reference samples per class; 0 uses every training sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, help="JSON path for the probe metrics.")
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
    raise ValueError("Cannot read labels from the global reference dataset.")


def class_balanced_subset(dataset, num_classes, samples_per_class, seed):
    if samples_per_class <= 0:
        return dataset, len(dataset)

    labels = dataset_targets(dataset)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(num_classes):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) == 0:
            raise ValueError(f"Reference split has no examples for class {class_id}.")
        rng.shuffle(indices)
        selected.extend(indices[:samples_per_class].tolist())
    rng.shuffle(selected)
    return Subset(dataset, selected), len(selected)


def reset_parameters(module):
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def parse_branch_indices(raw):
    values = [value.strip() for value in str(raw).split(",") if value.strip()]
    if not values:
        raise ValueError("--branches must contain at least one branch id from 1,2,3.")
    indices = []
    for value in values:
        branch_idx = int(value)
        if branch_idx not in (1, 2, 3):
            raise ValueError("--branches only supports 1,2,3.")
        if branch_idx not in indices:
            indices.append(branch_idx)
    return indices


def branch_modules(model, branch_idx):
    return (
        getattr(model, f"bottleneck{branch_idx}_1"),
        getattr(model, f"avgpool{branch_idx}"),
        getattr(model, f"middle_fc{branch_idx}"),
    )


def frozen_branch_logits(model, x, branch_idx):
    """Run a frozen source prefix and one trainable private branch prediction path."""
    with torch.no_grad():
        features = model.conv1(x)
        features = model.bn1(features)
        features = model.relu(features)
        features = model.layer1(features)
        if branch_idx >= 2:
            features = model.layer2(features)
        if branch_idx >= 3:
            features = model.layer3(features)
    bottleneck, avgpool, classifier = branch_modules(model, branch_idx)
    logits = bottleneck(features)
    logits = avgpool(logits)
    logits = torch.flatten(logits, 1)
    return classifier(logits)


def configure_fresh_branch_probe(model, branch_idx):
    """Reset and unfreeze only the selected branch's private prediction path."""
    bottleneck, _, classifier = branch_modules(model, branch_idx)
    reset_parameters(bottleneck)
    reset_parameters(classifier)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (bottleneck, classifier):
        for parameter in module.parameters():
            parameter.requires_grad = True


def train_branch_probe(model, reference_loader, device, branch_idx, args):
    configure_fresh_branch_probe(model, branch_idx)
    bottleneck, _, classifier = branch_modules(model, branch_idx)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.probe_lr,
        momentum=0.9,
        weight_decay=args.probe_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.probe_epochs, 1)
    )

    for _ in range(args.probe_epochs):
        model.eval()
        bottleneck.train()
        classifier.train()
        for x, target in reference_loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            logits = frozen_branch_logits(model, x, branch_idx)
            F.cross_entropy(logits, target).backward()
            optimizer.step()
        scheduler.step()


def init_metric_sums():
    return {
        "correct": 0,
        "nll": 0.0,
        "true_label_prob": 0.0,
        "entropy_norm": 0.0,
        "margin": 0.0,
    }


def accumulate_metric_sums(sums, logits, target, num_classes):
    probs = F.softmax(logits, dim=1)
    true_prob = probs.gather(1, target.unsqueeze(1)).squeeze(1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1) / float(np.log(max(num_classes, 2)))
    target_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    competitor = logits.masked_fill(
        F.one_hot(target, num_classes=num_classes).bool(), float("-inf")
    ).max(dim=1).values
    sums["correct"] += logits.argmax(dim=1).eq(target).sum().item()
    sums["nll"] += F.cross_entropy(logits, target, reduction="sum").item()
    sums["true_label_prob"] += true_prob.sum().item()
    sums["entropy_norm"] += entropy.sum().item()
    sums["margin"] += (target_logit - competitor).sum().item()


def finalize_metric_sums(sums, total):
    accuracy = sums["correct"] / max(total, 1)
    return {
        "acc": accuracy,
        "acc_pct": 100.0 * accuracy,
        "nll": sums["nll"] / max(total, 1),
        "true_label_prob": sums["true_label_prob"] / max(total, 1),
        "entropy_norm": sums["entropy_norm"] / max(total, 1),
        "margin": sums["margin"] / max(total, 1),
    }


@torch.no_grad()
def evaluate_branch_probe(model, dataloader, device, num_classes, branch_idx):
    model.eval()
    total = 0
    branch_sums = init_metric_sums()
    teacher_sums = init_metric_sums()

    for x, target in dataloader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        outputs = model(x)
        if not (isinstance(outputs, tuple) and len(outputs) == 8):
            raise RuntimeError("The frozen branch probe requires the ResNet18-BYOT output contract.")
        teacher_logits = outputs[0]
        branch_logits = frozen_branch_logits(model, x, branch_idx)
        batch_size = target.size(0)
        accumulate_metric_sums(branch_sums, branch_logits, target, num_classes)
        accumulate_metric_sums(teacher_sums, teacher_logits, target, num_classes)
        total += batch_size

    branch_metrics = finalize_metric_sums(branch_sums, total)
    teacher_metrics = finalize_metric_sums(teacher_sums, total)
    branch_metrics["to_teacher_error_ratio"] = (
        (1.0 - branch_metrics["acc"]) / max(1.0 - teacher_metrics["acc"], 1e-12)
    )
    return branch_metrics, teacher_metrics, total


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
    device = torch.device(args.device)
    num_classes = 10 if args.dataset == "cifar10" else 100

    # get_global_dataset expects these attributes for its dataset transforms.
    args.in_channels = 3
    global_train_dataset, _, global_test_dataset = get_global_dataset(args)
    # The diagnostic should not depend on random crop/flip noise.  Make a
    # shallow dataset copy so the global FL training dataset remains untouched.
    reference_base_dataset = copy.copy(global_train_dataset)
    if hasattr(reference_base_dataset, "transform") and hasattr(global_test_dataset, "transform"):
        reference_base_dataset.transform = global_test_dataset.transform
    reference_dataset, reference_count = class_balanced_subset(
        reference_base_dataset, num_classes, args.samples_per_class, args.seed
    )
    reference_loader = DataLoader(
        reference_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=device.type == "cuda", num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        global_test_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=device.type == "cuda", num_workers=args.num_workers,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("global_model", checkpoint)
    model = multi_resnet18_kd(num_classes=num_classes, in_channels=3).to(device)
    model.load_state_dict(state_dict, strict=True)
    branch_indices = parse_branch_indices(args.branches)
    branch_probes = {}
    teacher_metrics = None
    test_samples = 0
    for branch_idx in branch_indices:
        train_branch_probe(model, reference_loader, device, branch_idx, args)
        branch_metrics, current_teacher_metrics, test_samples = evaluate_branch_probe(
            model, test_loader, device, num_classes, branch_idx
        )
        branch_probes[f"B{branch_idx}"] = branch_metrics
        if teacher_metrics is None:
            teacher_metrics = current_teacher_metrics

    result = {
        "probe_type": "frozen_shallow_representation_fresh_branch_heads",
        "checkpoint": os.path.abspath(args.checkpoint),
        "dataset": args.dataset,
        "num_classes": num_classes,
        "reference_samples": reference_count,
        "samples_per_class": args.samples_per_class,
        "probe_epochs": args.probe_epochs,
        "probe_lr": args.probe_lr,
        "tested_branches": [f"B{branch_idx}" for branch_idx in branch_indices],
        "test_samples": test_samples,
        "teacher": teacher_metrics,
        "branch_probes": branch_probes,
        "seed": args.seed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
