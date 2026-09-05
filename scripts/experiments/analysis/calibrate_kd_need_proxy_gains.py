#!/usr/bin/env python3
"""Calibrate KD-need proxy gains to a common pre-local mean gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRIC_FIELDS = {
    "js": "branch_teacher_js_normalized",
    "advantage": "teacher_label_advantage_positive",
    "combined": "need_js_x_label_advantage",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--target-proxy", default="js", choices=tuple(METRIC_FIELDS),
        help="Raw proxy whose mean defines the common mapped-gate mean.",
    )
    return parser.parse_args()


def load_values(root: Path) -> tuple[dict[str, np.ndarray], dict]:
    values = {name: [] for name in METRIC_FIELDS}
    datasets, partitions, rounds = set(), set(), set()
    files = sorted(root.rglob("client_branch_metrics.csv"))
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["stage"] != "client_pre_local":
                    continue
                datasets.add(row["dataset"])
                partitions.add(
                    "iid" if row["partition"] == "iid" else f"beta={float(row['beta']):g}"
                )
                rounds.add(int(row["communication_round"]))
                for name, field in METRIC_FIELDS.items():
                    values[name].append(float(row[field]))
    if not files or not values["js"]:
        raise SystemExit(f"No pre-local stage-1 metrics found below {root}")
    arrays = {name: np.asarray(items, dtype=np.float64) for name, items in values.items()}
    counts = {len(items) for items in arrays.values()}
    if len(counts) != 1:
        raise ValueError("Proxy metric columns have inconsistent row counts.")
    metadata = {
        "input_files": len(files),
        "rows": int(next(iter(counts))),
        "datasets": sorted(datasets),
        "partitions": sorted(partitions),
        "communication_rounds": sorted(rounds),
        "stage": "client_pre_local",
    }
    return arrays, metadata


def gain_for_target(values: np.ndarray, target: float) -> float:
    if target <= 0.0:
        return 0.0
    low, high = 0.0, 1.0
    while float(np.minimum(1.0, high * values).mean()) < target:
        high *= 2.0
        if high > 1e9:
            raise ValueError("Cannot calibrate a nonzero target from zero proxy values.")
    for _ in range(100):
        midpoint = 0.5 * (low + high)
        mapped_mean = float(np.minimum(1.0, midpoint * values).mean())
        if mapped_mean < target:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def main():
    args = parse_args()
    arrays, metadata = load_values(Path(args.input_root))
    target = float(arrays[args.target_proxy].mean())
    result = {
        "calibration": "mean-matched clipped linear branch-need gates",
        "mapping": "gate = min(1, gain * raw_proxy)",
        "target_proxy": args.target_proxy,
        "target_gate_mean": target,
        **metadata,
        "proxies": {},
    }
    for name, values in arrays.items():
        gain = gain_for_target(values, target)
        mapped = np.minimum(1.0, gain * values)
        result["proxies"][name] = {
            "raw_mean": float(values.mean()),
            "gain": float(gain),
            "mapped_mean": float(mapped.mean()),
            "mapped_median": float(np.median(mapped)),
            "mapped_q90": float(np.quantile(mapped, 0.9)),
            "saturation_fraction": float(np.mean(mapped >= 1.0 - 1e-12)),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

