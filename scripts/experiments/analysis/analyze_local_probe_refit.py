#!/usr/bin/env python3
"""Export compact tables and plots for local-probe refit diagnostics."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dxfl-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEPTHS = ("b1", "b2", "b3", "final")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root", default="logs/analysis/logs_local_probe_refit"
    )
    parser.add_argument(
        "--output_dir", default="logs/analysis/local_probe_refit_analysis"
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def stat(summary, size, metric):
    return summary[str(size)][metric]["seed_macro"]


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = sorted(input_root.glob("*/round_*/fixed_*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No refit summaries under {input_root}")

    rows = []
    for path in summaries:
        budget = path.parent.name
        round_name = path.parent.parent.name
        dataset = path.parent.parent.parent.name
        completed_rounds = int(round_name.split("_")[-1])
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)["summary_by_sample_size"]
        sizes = sorted(int(value) for value in summary)
        for size in sizes:
            for depth in DEPTHS:
                prefix = f"postlocal.probe_comparison.{depth}."
                global_value = stat(
                    summary, size, prefix + "global_baseline_accuracy_pct"
                )
                frozen = stat(summary, size, prefix + "frozen_accuracy_pct")
                refit = stat(summary, size, prefix + "refit_accuracy_pct")
                recovery = stat(
                    summary, size, prefix + "recovery_accuracy_points"
                )
                rows.append({
                    "dataset": dataset,
                    "round": completed_rounds,
                    "budget": budget,
                    "sample_size": size,
                    "branch": depth,
                    "global_accuracy_pct": global_value["mean"],
                    "frozen_accuracy_pct": frozen["mean"],
                    "frozen_ci95": frozen["ci95_half_width"],
                    "refit_accuracy_pct": refit["mean"],
                    "refit_ci95": refit["ci95_half_width"],
                    "recovery_accuracy_points": recovery["mean"],
                    "recovery_ci95": recovery["ci95_half_width"],
                })

        figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        for axis, depth in zip(axes.flat, DEPTHS):
            selected = [
                row for row in rows
                if row["dataset"] == dataset
                and row["round"] == completed_rounds
                and row["budget"] == budget
                and row["branch"] == depth
            ]
            axis.errorbar(
                sizes, [row["frozen_accuracy_pct"] for row in selected],
                yerr=[row["frozen_ci95"] for row in selected],
                marker="o", capsize=3, label="Global frozen probe",
            )
            axis.errorbar(
                sizes, [row["refit_accuracy_pct"] for row in selected],
                yerr=[row["refit_ci95"] for row in selected],
                marker="s", capsize=3, label="Local-refit probe",
            )
            axis.axhline(
                selected[0]["global_accuracy_pct"], color="black",
                linestyle="--", linewidth=1, label="Global baseline",
            )
            axis.set_title(depth.upper())
            axis.set_xscale("log")
            axis.set_xticks(sizes)
            axis.set_xticklabels([str(size) for size in sizes])
            axis.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8)
        figure.supxlabel("Local training samples")
        figure.supylabel("Linear-probe test accuracy (%)")
        figure.suptitle(
            f"{dataset.upper()} / R={completed_rounds} / "
            f"{budget.replace('_', '-')}"
        )
        figure.tight_layout()
        stem = f"{dataset}_round_{completed_rounds:04d}_{budget}_probe_accuracy"
        figure.savefig(output_dir / f"{stem}.png", dpi=args.dpi, bbox_inches="tight")
        figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(figure)

    csv_path = output_dir / "local_probe_refit_accuracy.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows and plots to {output_dir}")


if __name__ == "__main__":
    main()
