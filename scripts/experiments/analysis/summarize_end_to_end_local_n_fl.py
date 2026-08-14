#!/usr/bin/env python3
"""Summarize final pre-aggregation logit/CKA metrics for local-n FL runs."""

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
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def t_critical_95(count):
    return {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
    }.get(count, 1.96)


def stats(values, student_t=False):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "ci95_half_width": None}
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    critical = t_critical_95(int(array.size)) if student_t else 1.96
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": std,
        "ci95_half_width": (
            float(critical * std / math.sqrt(array.size)) if array.size > 1 else 0.0
        ),
    }


def subtract_flattened(post, baseline):
    post_flat = flatten_numeric(post)
    baseline_flat = flatten_numeric(baseline)
    return {
        key: value - baseline_flat[key]
        for key, value in post_flat.items()
        if key in baseline_flat
    }


def final_record(payload):
    records = payload.get("records", [])
    if not records:
        raise ValueError("Result has no post-local records.")
    return max(records, key=lambda item: int(item["round"]))


def main():
    args = parse_args()
    paths = sorted(Path(args.input_root).glob("sample_*/seed_*/*_postlocal_internal_geometry.json"))
    if not paths:
        raise FileNotFoundError(f"No end-to-end result JSON found under {args.input_root}")

    client_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    global_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    provenance = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        config = payload["args"]
        sample_size = int(config["client_samples_per_client"])
        seed = int(config["seed"])
        record = final_record(payload)
        global_logits = record.get("round_start_global_logits", {})
        global_cka = record.get("round_start_global_cka", {})

        global_selected = {
            "global.logits": global_logits,
            "global.cka": global_cka,
        }
        for metric, value in flatten_numeric(global_selected).items():
            global_values[sample_size][metric][seed].append(value)

        for client in record["clients"]:
            selected = {
                "postlocal.logits": client.get("logits", {}),
                "postlocal.cka": client.get("cka", {}),
            }
            flattened = flatten_numeric(selected)
            logit_delta = subtract_flattened(client.get("logits", {}), global_logits)
            cka_delta = subtract_flattened(client.get("cka", {}), global_cka)
            flattened.update({f"delta.logits.{key}": value for key, value in logit_delta.items()})
            flattened.update({f"delta.cka.{key}": value for key, value in cka_delta.items()})
            for metric, value in flattened.items():
                client_values[sample_size][metric][seed].append(value)

        provenance.append({
            "path": str(path.resolve()),
            "dataset": config["dataset"],
            "budget": (
                "fixed_step" if int(config.get("local_steps_per_round", 0)) > 0
                else "fixed_epoch"
            ),
            "sample_size": sample_size,
            "seed": seed,
            "completed_round": int(record["round"]) + 1,
            "measured_clients": int(record["measured_client_count"]),
            "total_unique_train_samples": int(sample_size * int(config["n_clients"])),
        })

    summary, rows = {}, []
    for sample_size in sorted(set(client_values) | set(global_values)):
        metrics = {}
        for source in (client_values, global_values):
            for metric, seed_map in source[sample_size].items():
                seed_means = {
                    str(seed): float(np.mean(values))
                    for seed, values in sorted(seed_map.items())
                }
                pooled = [value for values in seed_map.values() for value in values]
                seed_summary = stats(list(seed_means.values()), student_t=True)
                pooled_summary = stats(pooled)
                metrics[metric] = {
                    "pooled_observations": pooled_summary,
                    "seed_macro": seed_summary,
                    "seed_means": seed_means,
                }
                rows.append({
                    "sample_size": sample_size,
                    "metric": metric,
                    "pooled_count": pooled_summary["count"],
                    "pooled_mean": pooled_summary["mean"],
                    "pooled_std": pooled_summary["std"],
                    "seed_count": seed_summary["count"],
                    "seed_macro_mean": seed_summary["mean"],
                    "seed_macro_std": seed_summary["std"],
                    "seed_macro_ci95_half_width": seed_summary["ci95_half_width"],
                })
        summary[str(sample_size)] = metrics

    payload = {
        "experiment": "end_to_end_fl_local_sample_size",
        "measurement": "last-round post-local/pre-aggregation models",
        "aggregation": {
            "postlocal": "client mean within seed, followed by macro mean and t-95% CI across seeds",
            "global": "round-start global value for each independently trained local-n trajectory",
            "delta": "paired post-local client value minus its trajectory-specific round-start global value",
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
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"aggregated {len(paths)} runs -> {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
