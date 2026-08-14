#!/usr/bin/env python3
"""Distinguish frozen-probe misalignment from loss of linear class information.

For every local-model fork, this diagnostic evaluates the round-start global
linear probes and then refits one linear probe per depth on the frozen local
features.  Both probe families are evaluated on the same official test set.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from local_data_size_internal_probe import (
    DEPTHS,
    atomic_json_dump,
    build_heads,
    class_balanced_subset,
    dataset_targets,
    extract_features,
    fit_linear_probes,
    load_datasets,
    load_global_model,
    local_subset_indices,
    logit_metrics,
    multi_resnet18_kd,
    seed_everything,
    train_teacher_only,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global_checkpoint", required=True)
    parser.add_argument("--global_probe_checkpoint", required=True)
    parser.add_argument(
        "--allow_relocated_global_checkpoint", action="store_true",
        help=(
            "Allow the checkpoint/probe pair to be copied under a different "
            "absolute repository path; dataset and completed round are still checked."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument("--datadir", default="./data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--sample_size", type=int, required=True)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--train_budget", choices=("steps", "epochs"), default="steps")
    parser.add_argument("--local_steps", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--local_batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--probe_epochs", type=int, default=30)
    parser.add_argument("--probe_lr", type=float, default=0.1)
    parser.add_argument("--probe_weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--probe_samples_per_class", type=int, default=0,
        help="Local-refit probe samples per class; 0 uses the full train set.",
    )
    parser.add_argument(
        "--test_samples_per_class", type=int, default=0,
        help="Test samples per class; must match the global probe checkpoint.",
    )
    return parser.parse_args()


def accuracy_fields(logits):
    return {
        depth: float(logits["sanity"][depth]["accuracy_pct"])
        for depth in DEPTHS
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable.")
    device = torch.device(args.device)

    train_dataset, test_dataset = load_datasets(args)
    global_probe = torch.load(args.global_probe_checkpoint, map_location="cpu")
    if global_probe.get("dataset") != args.dataset:
        raise ValueError(
            f"Global probe dataset is {global_probe.get('dataset')}, "
            f"but requested dataset is {args.dataset}."
        )
    expected_global = os.path.abspath(args.global_checkpoint)
    recorded_global = os.path.abspath(global_probe["global_checkpoint"])
    checkpoint_relocated = expected_global != recorded_global
    if checkpoint_relocated and not args.allow_relocated_global_checkpoint:
        raise ValueError(
            f"Global probe was fitted for {recorded_global}, not {expected_global}. "
            "Use --allow_relocated_global_checkpoint only when this is an exact "
            "copy of the original checkpoint on another server."
        )
    expected_test_samples = int(global_probe.get("reference_samples_per_class", 0))
    if args.test_samples_per_class != expected_test_samples:
        raise ValueError(
            "test_samples_per_class must match the global probe checkpoint: "
            f"expected {expected_test_samples}, got {args.test_samples_per_class}."
        )

    reference_test, reference_count = class_balanced_subset(
        test_dataset,
        args.test_samples_per_class,
        int(global_probe.get("reference_selection_seed", 1)),
        args.num_classes,
    )
    reference_loader = DataLoader(
        reference_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    # Probe fitting uses the same full train images and non-augmented transform
    # as the round-start global probes.  The local-training subset remains
    # augmented and is used only for local model optimization.
    probe_train_base = copy.copy(train_dataset)
    probe_train_base.transform = test_dataset.transform
    probe_train, probe_train_count = class_balanced_subset(
        probe_train_base, args.probe_samples_per_class,
        int(global_probe.get("seed", 3407)), args.num_classes,
    )
    probe_train_loader = DataLoader(
        probe_train, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    base_model, checkpoint = load_global_model(
        args.global_checkpoint, args.dataset, device
    )
    checkpoint_completed_rounds = int(
        checkpoint.get(
            "completed_rounds", int(checkpoint.get("round", -1)) + 1
        )
    )
    probe_completed_rounds = int(global_probe.get("global_completed_rounds", -1))
    if checkpoint_completed_rounds != probe_completed_rounds:
        raise ValueError(
            "Checkpoint/probe completed-round mismatch: "
            f"checkpoint={checkpoint_completed_rounds}, probe={probe_completed_rounds}."
        )
    base_state = {
        key: value.detach().cpu().clone()
        for key, value in base_model.state_dict().items()
    }
    base_model.to("cpu")
    del base_model

    frozen_heads = build_heads(
        global_probe["head_dimensions"], global_probe["heads_state_dict"],
        args.num_classes, device,
    )
    global_logits = global_probe["global_reference_metrics"]["logits"]
    global_accuracy = accuracy_fields(global_logits)

    client_results = []
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
            subset, batch_size=args.local_batch_size, shuffle=True,
            drop_last=(
                args.train_budget == "steps"
                and len(subset) >= args.local_batch_size
            ),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda", generator=generator,
        )

        model = multi_resnet18_kd(
            num_classes=args.num_classes, in_channels=3
        ).to(device)
        model.load_state_dict(base_state, strict=True)
        training = train_teacher_only(model, local_loader, args, device)
        model.eval()

        # Extract each local feature set once.  Probe optimization thereafter
        # updates only four linear heads; the local backbone remains frozen.
        test_features, test_targets = extract_features(
            model, reference_loader, device
        )
        frozen_logits = logit_metrics(
            test_features, test_targets, frozen_heads, args.num_classes, device
        )
        probe_features, probe_targets = extract_features(
            model, probe_train_loader, device
        )
        seed_everything(client_seed + 104_729)
        refit_heads, _ = fit_linear_probes(
            probe_features, probe_targets, args, device
        )
        refit_train_logits = logit_metrics(
            probe_features, probe_targets, refit_heads, args.num_classes, device
        )
        refit_test_logits = logit_metrics(
            test_features, test_targets, refit_heads, args.num_classes, device
        )

        frozen_accuracy = accuracy_fields(frozen_logits)
        refit_accuracy = accuracy_fields(refit_test_logits)
        comparison = {
            depth: {
                "global_baseline_accuracy_pct": global_accuracy[depth],
                "frozen_accuracy_pct": frozen_accuracy[depth],
                "refit_accuracy_pct": refit_accuracy[depth],
                "recovery_accuracy_points": (
                    refit_accuracy[depth] - frozen_accuracy[depth]
                ),
                "refit_minus_global_accuracy_points": (
                    refit_accuracy[depth] - global_accuracy[depth]
                ),
                "refit_train_accuracy_pct": float(
                    refit_train_logits["sanity"][depth]["accuracy_pct"]
                ),
            }
            for depth in DEPTHS
        }

        local_labels = dataset_targets(train_dataset)[indices]
        counts = np.bincount(local_labels, minlength=args.num_classes)
        client_results.append({
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
            "postlocal_metrics": {
                "probe_comparison": comparison,
                "frozen_probe_logits": frozen_logits,
                "refit_probe_logits": refit_test_logits,
            },
            "delta_from_round_start_global": {},
        })
        print(
            f"client={client_id:02d} n={args.sample_size} "
            + " ".join(
                f"{depth}:frozen={frozen_accuracy[depth]:.2f},"
                f"refit={refit_accuracy[depth]:.2f},"
                f"recovery={comparison[depth]['recovery_accuracy_points']:+.2f}"
                for depth in DEPTHS
            ),
            flush=True,
        )

        model.to("cpu")
        refit_heads.to("cpu")
        del model, refit_heads, test_features, test_targets
        del probe_features, probe_targets, refit_train_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "format_version": 1,
        "experiment": "local_probe_refit_diagnostic",
        "research_question": (
            "frozen-global-probe misalignment versus loss of linearly "
            "decodable class information"
        ),
        "dataset": args.dataset,
        "num_classes": int(args.num_classes),
        "global_checkpoint": expected_global,
        "global_checkpoint_original_probe_path": recorded_global,
        "global_checkpoint_relocated": bool(checkpoint_relocated),
        "global_completed_rounds": int(
            checkpoint_completed_rounds
        ),
        "global_probe_checkpoint": os.path.abspath(args.global_probe_checkpoint),
        "probe_protocol": {
            "backbone_frozen_during_refit": True,
            "depths": list(DEPTHS),
            "train_split": f"official_{args.dataset}_train",
            "train_transform": "official test/evaluation transform",
            "train_samples": int(probe_train_count),
            "samples_per_class": int(args.probe_samples_per_class),
            "epochs": int(args.probe_epochs),
            "lr": float(args.probe_lr),
            "weight_decay": float(args.probe_weight_decay),
            "test_split": f"official_{args.dataset}_test",
            "test_samples": int(reference_count),
        },
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
        "client_results": client_results,
    }
    atomic_json_dump(payload, args.output)
    print(f"saved refit diagnostic: {args.output}", flush=True)


if __name__ == "__main__":
    main()
