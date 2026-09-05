#!/usr/bin/env python3
"""Fit strict linear probes to frozen raw-trunk representations.

The source checkpoint must be a model trained without branch supervision when
the goal is to isolate what final-CE backpropagation alone placed in the
intermediate representations.  B1/B2/B3/final features are global-average
pooled once and cached.  Only one fresh ``nn.Linear`` classifier per depth is
optimized; no backbone, batch-normalization, bottleneck, or nonlinear probe
parameter is trained.
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.analysis.local_data_size_internal_probe import (
    DATASET_NUM_CLASSES,
    DEPTHS,
    class_balanced_subset,
    extract_features,
    load_datasets,
    load_global_model,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_NUM_CLASSES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--datadir", default="./data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--probe_epochs", type=int, default=30)
    parser.add_argument("--probe_lr", type=float, default=0.1)
    parser.add_argument("--probe_weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--probe_samples_per_class",
        type=int,
        default=0,
        help="Train samples per class; 0 uses the full official train set.",
    )
    parser.add_argument(
        "--test_samples_per_class",
        type=int,
        default=0,
        help="Test samples per class; 0 uses the full official test set.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def atomic_json_dump(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        delete=False,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        temporary_path = handle.name
    os.replace(temporary_path, output_path)


def fit_heads(features, targets, args, device):
    dimensions = {depth: int(features[depth].shape[1]) for depth in DEPTHS}
    heads = nn.ModuleDict(
        {
            depth: nn.Linear(dimensions[depth], args.num_classes)
            for depth in DEPTHS
        }
    ).to(device)
    dataset = TensorDataset(*(features[depth] for depth in DEPTHS), targets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.SGD(
        heads.parameters(),
        lr=args.probe_lr,
        momentum=0.9,
        weight_decay=args.probe_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.probe_epochs, 1)
    )

    for epoch in range(args.probe_epochs):
        heads.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in loader:
            feature_batch = {
                depth: batch[index].to(device, non_blocking=True)
                for index, depth in enumerate(DEPTHS)
            }
            target = batch[-1].to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            # The heads are independent. Averaging preserves the optimization
            # convention used by the repository's earlier strict probes.
            loss = sum(
                F.cross_entropy(heads[depth](feature_batch[depth]), target)
                for depth in DEPTHS
            ) / float(len(DEPTHS))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(target.numel())
            sample_count += int(target.numel())
        scheduler.step()
        print(
            f"probe epoch {epoch + 1:03d}/{args.probe_epochs:03d} "
            f"loss={loss_sum / max(sample_count, 1):.6f}",
            flush=True,
        )

    heads.eval()
    return heads, dimensions


@torch.no_grad()
def evaluate_heads(heads, features, targets, device):
    metrics = {}
    for depth in DEPTHS:
        correct = 0
        loss_sum = 0.0
        sample_count = 0
        feature_chunks = features[depth].split(4096)
        target_chunks = targets.split(4096)
        for feature_chunk, target_chunk in zip(feature_chunks, target_chunks):
            feature_chunk = feature_chunk.to(device, non_blocking=True)
            target_chunk = target_chunk.to(device, non_blocking=True).long()
            logits = heads[depth](feature_chunk)
            correct += int(logits.argmax(dim=1).eq(target_chunk).sum().item())
            loss_sum += float(
                F.cross_entropy(logits, target_chunk, reduction="sum").item()
            )
            sample_count += int(target_chunk.numel())
        metrics[depth] = {
            "accuracy": correct / max(sample_count, 1),
            "accuracy_pct": 100.0 * correct / max(sample_count, 1),
            "nll": loss_sum / max(sample_count, 1),
            "samples": sample_count,
        }
    return metrics


def main():
    args = parse_args()
    if args.probe_epochs <= 0:
        raise ValueError("--probe_epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    started_at = time.time()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.num_classes = DATASET_NUM_CLASSES[args.dataset]
    train_dataset, test_dataset = load_datasets(args)

    # Disable train-time augmentation for a deterministic representation probe.
    probe_train_base = copy.copy(train_dataset)
    probe_train_base.transform = test_dataset.transform
    probe_train, probe_train_count = class_balanced_subset(
        probe_train_base,
        args.probe_samples_per_class,
        args.seed,
        args.num_classes,
    )
    probe_test, probe_test_count = class_balanced_subset(
        test_dataset,
        args.test_samples_per_class,
        args.seed + 1,
        args.num_classes,
    )
    train_loader = DataLoader(
        probe_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        probe_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, checkpoint = load_global_model(args.checkpoint, args.dataset, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    feature_started_at = time.time()
    train_features, train_targets = extract_features(model, train_loader, device)
    feature_seconds = time.time() - feature_started_at

    fit_started_at = time.time()
    heads, dimensions = fit_heads(train_features, train_targets, args, device)
    fit_seconds = time.time() - fit_started_at
    del train_features, train_targets

    eval_started_at = time.time()
    test_features, test_targets = extract_features(model, test_loader, device)
    metrics = evaluate_heads(heads, test_features, test_targets, device)
    eval_seconds = time.time() - eval_started_at

    checkpoint_args = checkpoint.get("args", {})
    payload = {
        "format_version": 1,
        "experiment": "strict_linear_representation_probe",
        "definition": (
            "Frozen raw ResNet trunk feature at each depth, global average "
            "pooling, and one freshly fitted nn.Linear classifier. No branch "
            "bottleneck, activation, batch normalization, or backbone update."
        ),
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_dataset": (
            checkpoint_args.get("dataset")
            if isinstance(checkpoint_args, dict)
            else None
        ),
        "global_round": int(checkpoint.get("round", -1)),
        "global_completed_rounds": int(
            checkpoint.get(
                "completed_rounds", int(checkpoint.get("round", -1)) + 1
            )
        ),
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "depths": list(DEPTHS),
        "head_dimensions": dimensions,
        "probe_train_split": f"official_{args.dataset}_train_no_augmentation",
        "probe_train_samples": int(probe_train_count),
        "probe_samples_per_class": int(args.probe_samples_per_class),
        "evaluation_split": f"official_{args.dataset}_test",
        "evaluation_samples": int(probe_test_count),
        "test_samples_per_class": int(args.test_samples_per_class),
        "probe_epochs": int(args.probe_epochs),
        "probe_lr": float(args.probe_lr),
        "probe_weight_decay": float(args.probe_weight_decay),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "metrics": metrics,
        "runtime_seconds": {
            "train_feature_extraction": feature_seconds,
            "linear_head_fitting": fit_seconds,
            "test_feature_extraction_and_evaluation": eval_seconds,
            "total": time.time() - started_at,
        },
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
