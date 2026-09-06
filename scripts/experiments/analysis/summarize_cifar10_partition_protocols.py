#!/usr/bin/env python3
"""Summarize corrected-unbalanced and balanced-500 CIFAR-10 runs.

The script reports late-round averages from the same trajectories.  Therefore
last-100 evaluation is an analysis choice, not an additional training run.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np


DEFAULT_ROOTS = (
    "logs/analysis/logs_cifar10_js_granularity_min10_seeds012",
    "logs/analysis/logs_cifar10_min10_baselines_seeds012",
    "logs/analysis/logs_cifar10_balanced500_pilot_seed0",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS))
    parser.add_argument(
        "--output",
        default="logs/analysis/cifar10_partition_protocol_summary.csv",
    )
    return parser.parse_args()


def protocol(path: Path) -> str:
    text = str(path)
    if "balanced500" in text:
        return "balanced500"
    if "min10" in text:
        return "unbalanced_min10"
    return "unknown"


def method_and_seed(path: Path):
    match = re.search(r"/runs/([^/]+)/cifar10/[^/]+/seed(\d+)/", str(path))
    if not match:
        return None
    method = {"js": "js_branch"}.get(match.group(1), match.group(1))
    return method, int(match.group(2))


def summarize(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    acc = np.asarray(payload["acc_global"], dtype=float)
    if len(acc) < 100:
        raise ValueError(f"Expected at least 100 rounds: {path}")
    parsed = method_and_seed(path)
    if parsed is None:
        return None
    method, seed = parsed
    alpha = [x for x in payload.get("byot_effective_alpha_mean", []) if x is not None]
    alpha = np.asarray(alpha, dtype=float)
    return {
        "protocol": protocol(path),
        "method": method,
        "seed": seed,
        "rounds": len(acc),
        "last30_acc": float(acc[-30:].mean()),
        "last30_round_std": float(acc[-30:].std()),
        "last100_acc": float(acc[-100:].mean()),
        "last100_round_std": float(acc[-100:].std()),
        "best_acc": float(acc.max()),
        "final_acc": float(acc[-1]),
        "last30_effective_lambda": (
            float(alpha[-30:].mean()) if method != "plain" and len(alpha) >= 30 else ""
        ),
        "path": str(path),
    }


def main():
    args = parse_args()
    rows = []
    for root_text in args.roots:
        root = Path(root_text)
        for path in root.glob("**/*.pkl"):
            row = summarize(path)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: (row["protocol"], row["method"], row["seed"]))
    if not rows:
        raise SystemExit("No matching completed pickle logs were found.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("protocol,method,n_seeds,last30_mean,last30_seed_std,last100_mean,last100_seed_std,best_mean")
    groups = {}
    for row in rows:
        groups.setdefault((row["protocol"], row["method"]), []).append(row)
    for (proto, method), group in sorted(groups.items()):
        last30 = np.asarray([row["last30_acc"] for row in group])
        last100 = np.asarray([row["last100_acc"] for row in group])
        best = np.asarray([row["best_acc"] for row in group])
        print(
            f"{proto},{method},{len(group)},"
            f"{last30.mean():.3f},{last30.std():.3f},"
            f"{last100.mean():.3f},{last100.std():.3f},{best.mean():.3f}"
        )
    print(f"Wrote per-run summary: {output}")


if __name__ == "__main__":
    main()

