#!/usr/bin/env python3
"""Plot the fixed-checkpoint, fixed-local-epoch motivation diagnostics."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dxfl-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEPTHS = ("b1", "b2", "b3", "final")
PAIRS = ("b1-b2", "b1-b3", "b2-b3", "b1-final", "b2-final", "b3-final")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default="logs/analysis/logs_local_data_size_internal_probe_epochs/summary.json",
    )
    parser.add_argument(
        "--output_dir",
        default="logs/analysis/checkpoint_fixed_epoch_motivation",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def stat(summary, sample_size, key):
    return summary[str(sample_size)][key]["seed_macro"]


def save(fig, output_dir, name, dpi):
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    with open(args.summary, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload["summary_by_sample_size"]
    sample_sizes = sorted(map(int, summary))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    colors = dict(zip(DEPTHS, plt.get_cmap("tab10")(np.arange(len(DEPTHS)))))
    for metric in ("within_class_variance", "between_class_variance"):
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), sharex=True)
        for axis, depth in zip(axes.flat, DEPTHS):
            post_key = f"postlocal.feature_geometry.{depth}.{metric}"
            delta_key = f"delta_from_global.feature_geometry.{depth}.{metric}"
            post = [stat(summary, size, post_key) for size in sample_sizes]
            delta = [stat(summary, size, delta_key) for size in sample_sizes]
            y = np.asarray([item["mean"] for item in post])
            ci = np.asarray([item["ci95_half_width"] for item in post])
            baseline = float(y[0] - delta[0]["mean"])
            axis.errorbar(
                sample_sizes, y, yerr=ci, marker="o", capsize=3,
                linewidth=1.8, color=colors[depth], label="Post-local mean ± 95% CI",
            )
            axis.axhline(
                baseline, color="#555555", linestyle="--", linewidth=1.3,
                label="Fixed global checkpoint",
            )
            axis.set_xscale("log")
            axis.set_xticks(sample_sizes, [f"{size:,}" for size in sample_sizes])
            axis.set_title(depth.upper())
            axis.set_ylabel(metric.replace("_", " ").title())
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=8)
            for size, post_item, delta_item in zip(sample_sizes, post, delta):
                rows.append({
                    "category": "feature",
                    "depth_or_pair": depth,
                    "metric": metric,
                    "local_samples": size,
                    "postlocal_mean": post_item["mean"],
                    "postlocal_ci95_half_width": post_item["ci95_half_width"],
                    "delta_from_global_mean": delta_item["mean"],
                    "delta_from_global_ci95_half_width": delta_item["ci95_half_width"],
                    "global_reference": baseline,
                })
        for axis in axes[-1]:
            axis.set_xlabel("Unique local training samples (5 local epochs)")
        fig.suptitle(f"Checkpoint pre-aggregation {metric.replace('_', ' ')}", y=1.01)
        save(fig, output_dir, f"feature_{metric}", args.dpi)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.35), sharex=True)
    logit_specs = (
        (
            "off_diagonal_centered_cosine_mean",
            "Mean pairwise centered-logit cosine",
            "Cross-depth logit alignment",
        ),
        (
            "directional_variance_mean",
            "Logit directional variance",
            "Cross-depth logit dispersion",
        ),
    )
    for axis, (metric, ylabel, title) in zip(axes, logit_specs):
        post_key = f"postlocal.logits.depth_summary.{metric}"
        delta_key = f"delta_from_global.depth_summary.{metric}"
        post = [stat(summary, size, post_key) for size in sample_sizes]
        delta = [stat(summary, size, delta_key) for size in sample_sizes]
        y = np.asarray([item["mean"] for item in post])
        ci = np.asarray([item["ci95_half_width"] for item in post])
        baseline = float(y[0] - delta[0]["mean"])
        axis.errorbar(sample_sizes, y, yerr=ci, marker="o", capsize=3,
                      linewidth=1.8, color="#2468b4", label="Post-local mean ± 95% CI")
        axis.axhline(baseline, color="#555555", linestyle="--", linewidth=1.3,
                     label="Fixed global checkpoint")
        axis.set_xscale("log")
        axis.set_xticks(sample_sizes, [f"{size:,}" for size in sample_sizes])
        axis.set_xlabel("Unique local training samples (5 local epochs)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
        for size, post_item, delta_item in zip(sample_sizes, post, delta):
            rows.append({
                "category": "logit_aggregate",
                "depth_or_pair": "all_pairs",
                "metric": metric,
                "local_samples": size,
                "postlocal_mean": post_item["mean"],
                "postlocal_ci95_half_width": post_item["ci95_half_width"],
                "delta_from_global_mean": delta_item["mean"],
                "delta_from_global_ci95_half_width": delta_item["ci95_half_width"],
                "global_reference": baseline,
            })
    save(fig, output_dir, "logit_aggregate", args.dpi)

    fig, axis = plt.subplots(figsize=(7.5, 4.9))
    pair_colors = plt.get_cmap("tab10")(np.arange(len(PAIRS)))
    for color, pair in zip(pair_colors, PAIRS):
        key = f"delta_from_global.pairwise_logits.{pair}.centered_logit_cosine_mean"
        values = [stat(summary, size, key) for size in sample_sizes]
        means = [item["mean"] for item in values]
        axis.plot(sample_sizes, means, marker="o", markersize=3.7,
                  linewidth=1.65, color=color, label=pair.upper())
        for size, item in zip(sample_sizes, values):
            rows.append({
                "category": "logit_pair",
                "depth_or_pair": pair,
                "metric": "centered_logit_cosine_mean",
                "local_samples": size,
                "postlocal_mean": "",
                "postlocal_ci95_half_width": "",
                "delta_from_global_mean": item["mean"],
                "delta_from_global_ci95_half_width": item["ci95_half_width"],
                "global_reference": "",
            })
    axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2)
    axis.set_xscale("log")
    axis.set_xticks(sample_sizes, [f"{size:,}" for size in sample_sizes])
    axis.set_xlabel("Unique local training samples (5 local epochs)")
    axis.set_ylabel("Change from global in centered-logit cosine")
    axis.set_title("Which depth pairs drive cross-depth alignment?")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, output_dir, "logit_pairwise_delta", args.dpi)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), sharex=True)
    for axis, depth in zip(axes.flat, DEPTHS):
        key = f"postlocal.logits.sanity.{depth}.accuracy_pct"
        values = [stat(summary, size, key) for size in sample_sizes]
        y = np.asarray([item["mean"] for item in values])
        ci = np.asarray([item["ci95_half_width"] for item in values])
        # The frozen probe checkpoint is shared across conditions. Its test
        # accuracy is recovered from the raw job metadata below when present.
        raw_root = Path(args.summary).parent
        first_job = next(raw_root.glob("sample_*/seed_*/metrics.json"))
        with first_job.open("r", encoding="utf-8") as handle:
            baseline_payload = json.load(handle)
        baseline = float(
            baseline_payload["global_reference_metrics"]["logits"]["sanity"][depth][
                "accuracy_pct"
            ]
        )
        axis.errorbar(
            sample_sizes, y, yerr=ci, marker="o", capsize=3,
            linewidth=1.8, color=colors[depth], label="Post-local mean ± 95% CI",
        )
        axis.axhline(
            baseline, color="#555555", linestyle="--", linewidth=1.3,
            label="Fixed global checkpoint",
        )
        axis.set_xscale("log")
        axis.set_xticks(sample_sizes, [f"{size:,}" for size in sample_sizes])
        axis.set_title(depth.upper())
        axis.set_ylabel("Frozen-probe test accuracy (%)")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
        for size, item in zip(sample_sizes, values):
            rows.append({
                "category": "logit_sanity",
                "depth_or_pair": depth,
                "metric": "accuracy_pct",
                "local_samples": size,
                "postlocal_mean": item["mean"],
                "postlocal_ci95_half_width": item["ci95_half_width"],
                "delta_from_global_mean": item["mean"] - baseline,
                "delta_from_global_ci95_half_width": item["ci95_half_width"],
                "global_reference": baseline,
            })
    for axis in axes[-1]:
        axis.set_xlabel("Unique local training samples (5 local epochs)")
    fig.suptitle("Frozen-probe sanity check", y=1.01)
    save(fig, output_dir, "logit_probe_accuracy_sanity", args.dpi)

    with (output_dir / "checkpoint_fixed_epoch_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows and 5 PNG/PDF figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
