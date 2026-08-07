#!/usr/bin/env python3
"""Aggregate within-client local-data-size probe jobs across clients and seeds."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def flatten_numeric(value, prefix=""):
    flattened = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numeric(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened[prefix] = float(value)
    return flattened


def t_critical_95(sample_count):
    # Two-sided Student-t critical values for 95% intervals.  The normal value
    # is a sufficiently accurate fallback once the seed count is large.
    table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
        12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
        17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093,
    }
    return table.get(sample_count, 1.96)


def statistics(values, student_t_interval=False):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "ci95_half_width": None}
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    critical = t_critical_95(int(array.size)) if student_t_interval else 1.96
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": std,
        "ci95_half_width": float(critical * std / math.sqrt(array.size)) if array.size > 1 else 0.0,
    }


def main():
    args = parse_args()
    paths = sorted(Path(args.input_root).glob("sample_*/seed_*/metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No completed metrics.json jobs found under {args.input_root}")

    pooled = defaultdict(lambda: defaultdict(list))
    per_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    provenance = []
    ignored_prefixes = (
        "client_id", "client_seed", "subset_seed", "local_samples",
        "training.wall_time_seconds",
    )
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            job = json.load(handle)
        sample_size = int(job["sample_size"])
        sampling_seed = int(job["sampling_seed"])
        provenance.append({
            "path": str(path.resolve()),
            "sample_size": sample_size,
            "sampling_seed": sampling_seed,
            "clients": len(job["client_results"]),
        })
        for client in job["client_results"]:
            selected = {
                "local_data": {
                    "observed_classes": client["observed_classes"],
                    "class_count_std": client["local_class_count_std"],
                },
                "training": client["training"],
                "postlocal": client["postlocal_metrics"],
                "delta_from_global": client["delta_from_round_start_global"],
            }
            for metric, value in flatten_numeric(selected).items():
                if metric.startswith(ignored_prefixes):
                    continue
                pooled[sample_size][metric].append(value)
                per_seed[sample_size][metric][sampling_seed].append(value)

    summary = {}
    csv_rows = []
    for sample_size in sorted(pooled):
        sample_summary = {}
        for metric in sorted(pooled[sample_size]):
            client_stats = statistics(pooled[sample_size][metric])
            seed_means = [
                float(np.mean(values))
                for _, values in sorted(per_seed[sample_size][metric].items())
            ]
            seed_stats = statistics(seed_means, student_t_interval=True)
            sample_summary[metric] = {
                "pooled_clients": client_stats,
                "seed_macro": seed_stats,
                "seed_means": {
                    str(seed): float(np.mean(values))
                    for seed, values in sorted(per_seed[sample_size][metric].items())
                },
            }
            csv_rows.append({
                "sample_size": sample_size,
                "metric": metric,
                "pooled_client_count": client_stats["count"],
                "pooled_client_mean": client_stats["mean"],
                "pooled_client_std": client_stats["std"],
                "seed_count": seed_stats["count"],
                "seed_macro_mean": seed_stats["mean"],
                "seed_macro_std": seed_stats["std"],
                "seed_macro_ci95_half_width": seed_stats["ci95_half_width"],
            })
        summary[str(sample_size)] = sample_summary

    payload = {
        "experiment": "within_client_local_data_size_probe",
        "aggregation": {
            "pooled_clients": "all local forks pooled within a sample-size condition",
            "seed_macro": "client mean computed within each seed, then summarized across seeds",
            "recommended_error_bar": "seed_macro_ci95_half_width",
            "ci_note": "two-sided Student-t 95% CI over seed-level client means; also show individual seed values",
        },
        "jobs": provenance,
        "summary_by_sample_size": summary,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(csv_rows[0].keys())
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"aggregated {len(paths)} jobs -> {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
