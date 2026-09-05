#!/usr/bin/env python3
"""Aggregate strict-linear probe JSON files into JSON, CSV, and Markdown."""

import argparse
import csv
import json
import statistics
from pathlib import Path


DEPTHS = ("b1", "b2", "b3", "final")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_root.glob("**/strict_linear_probe_metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No strict-linear metrics found under {input_root}")

    runs = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        row = {
            "dataset": payload["dataset"],
            "seed": int(payload["seed"]),
            "checkpoint": payload["checkpoint"],
            "probe_train_samples": int(payload["probe_train_samples"]),
            "evaluation_samples": int(payload["evaluation_samples"]),
            "runtime_seconds": float(payload["runtime_seconds"]["total"]),
        }
        for depth in DEPTHS:
            row[f"{depth}_accuracy_pct"] = float(
                payload["metrics"][depth]["accuracy_pct"]
            )
        runs.append(row)

    aggregate = []
    for dataset in sorted({row["dataset"] for row in runs}):
        selected = [row for row in runs if row["dataset"] == dataset]
        item = {"dataset": dataset, "num_runs": len(selected), "depths": {}}
        for depth in DEPTHS:
            values = [row[f"{depth}_accuracy_pct"] for row in selected]
            item["depths"][depth] = {
                "mean_accuracy_pct": statistics.fmean(values),
                "std_accuracy_pct": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        aggregate.append(item)

    summary = {"runs": runs, "aggregate": aggregate}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    fieldnames = [
        "dataset",
        "seed",
        *(f"{depth}_accuracy_pct" for depth in DEPTHS),
        "probe_train_samples",
        "evaluation_samples",
        "runtime_seconds",
        "checkpoint",
    ]
    with (output_dir / "per_run.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runs)

    lines = [
        "# Strict linear representation probe",
        "",
        "Accuracy is measured by GAP + a fresh linear classifier on each frozen raw-trunk feature.",
        "",
        "| Dataset | Runs | B1 | B2 | B3 | Final |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in aggregate:
        cells = []
        for depth in DEPTHS:
            stats = item["depths"][depth]
            cells.append(
                f"{stats['mean_accuracy_pct']:.3f} ± {stats['std_accuracy_pct']:.3f}"
            )
        lines.append(
            f"| {item['dataset']} | {item['num_runs']} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")
    markdown = "\n".join(lines)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
