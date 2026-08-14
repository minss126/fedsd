#!/usr/bin/env python3
"""Validate independently trained, horizon-normalized FL endpoint checkpoints."""

import argparse
import json
from pathlib import Path

import torch


def parse_csv(raw, cast=str):
    return [cast(token.strip()) for token in raw.split(",") if token.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--datasets", default="cifar10,cifar100")
    parser.add_argument("--rounds", default="10,50,100")
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument(
        "--method", default="teacher_only_independent_endpoint"
    )
    parser.add_argument("--expected_lr", type=float, default=0.1)
    parser.add_argument("--expected_eta_min", type=float, default=0.0)
    parser.add_argument("--manifest_output")
    return parser.parse_args()


def checkpoint_path(root, dataset, rounds, completed, seed, method):
    setting = (
        f"{dataset}_resnet18/iid/clients_20/fedavg/seed{seed}/"
        f"budget_r{rounds:04d}"
    )
    return root / setting / f"{method}_round{completed:04d}.pt"


def require_equal(actual, expected, label, path):
    if actual != expected:
        raise ValueError(
            f"{label} mismatch in {path}: expected {expected!r}, got {actual!r}"
        )


def main():
    args = parse_args()
    root = Path(args.checkpoint_root).resolve()
    datasets = parse_csv(args.datasets)
    rounds = parse_csv(args.rounds, int)
    if not rounds or any(value <= 0 for value in rounds):
        raise ValueError("--rounds must contain positive endpoint budgets.")

    records = []
    for dataset in datasets:
        reference_initial = None
        reference_initial_path = None
        for total_rounds in rounds:
            initial_path = checkpoint_path(
                root, dataset, total_rounds, 0, args.global_seed, args.method
            )
            final_path = checkpoint_path(
                root, dataset, total_rounds, total_rounds,
                args.global_seed, args.method,
            )
            for path in (initial_path, final_path):
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(path)

            initial = torch.load(initial_path, map_location="cpu")
            final = torch.load(final_path, map_location="cpu")
            for payload, path, completed in (
                (initial, initial_path, 0),
                (final, final_path, total_rounds),
            ):
                require_equal(
                    int(payload.get("completed_rounds", -1)), completed,
                    "completed_rounds", path,
                )
                metadata = payload.get("args", {})
                require_equal(
                    int(metadata.get("round", -1)), total_rounds,
                    "cosine horizon / args.round", path,
                )
                require_equal(metadata.get("scheduler"), "cosine", "scheduler", path)
                if abs(float(metadata.get("lr", -1.0)) - args.expected_lr) > 1e-12:
                    raise ValueError(f"initial LR mismatch in {path}")
                if abs(
                    float(metadata.get("eta_min", -1.0)) - args.expected_eta_min
                ) > 1e-12:
                    raise ValueError(f"eta_min mismatch in {path}")
                require_equal(int(metadata.get("seed", -1)), args.global_seed, "seed", path)
                require_equal(int(metadata.get("n_clients", -1)), 20, "n_clients", path)
                require_equal(metadata.get("partition"), "iid", "partition", path)

            initial_state = initial["global_model"]
            if reference_initial is None:
                reference_initial = initial_state
                reference_initial_path = initial_path
            else:
                if set(initial_state) != set(reference_initial):
                    raise ValueError(
                        f"Initialization keys differ: {reference_initial_path} vs {initial_path}"
                    )
                unequal = [
                    key for key in initial_state
                    if not torch.equal(initial_state[key], reference_initial[key])
                ]
                if unequal:
                    raise ValueError(
                        f"Independent budgets do not share an identical initialization "
                        f"for {dataset}; first differing tensor: {unequal[0]}"
                    )

            records.append({
                "dataset": dataset,
                "global_seed": args.global_seed,
                "training_budget_rounds": total_rounds,
                "scheduler": "cosine",
                "cosine_horizon": total_rounds,
                "initial_lr": args.expected_lr,
                "eta_min": args.expected_eta_min,
                "initial_checkpoint": str(initial_path),
                "final_checkpoint": str(final_path),
                "final_accuracy": final.get("accuracy"),
            })

    manifest = {
        "design": "independent_horizon_normalized_final_endpoints",
        "question": (
            "whether within-client local-sample-size trends reproduce across "
            "independent final FL training budgets"
        ),
        "initialization_control": (
            "round 0 is a sanity control and is not counted as a trained endpoint"
        ),
        "records": records,
    }
    if args.manifest_output:
        output = Path(args.manifest_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
