#!/usr/bin/env python3
"""Create tables and plots for the fixed-step/fixed-epoch logit diagnostics."""

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


DEPTH_PAIRS = ("b1-b2", "b1-b3", "b1-final", "b2-b3", "b2-final", "b3-final")
SETTINGS = (
    ("fixed_step", "Fixed step", "logs/analysis/logs_local_data_size_internal_probe"),
    ("fixed_epoch", "Fixed epoch", "logs/analysis/logs_local_data_size_internal_probe_epochs"),
)
AGGREGATE_METRICS = (
    (
        "off_diagonal_centered_cosine_mean",
        "Mean pairwise centered-logit cosine",
        "Cosine mean",
    ),
    (
        "directional_variance_mean",
        "Logit directional variance across depths",
        "Directional variance",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument(
        "--output_dir", default="logs/analysis/local_data_size_logit_analysis"
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_summaries(repo_root):
    loaded = {}
    for key, title, relative_root in SETTINGS:
        path = repo_root / relative_root / "summary.json"
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        loaded[key] = {
            "title": title,
            "summary": payload["summary_by_sample_size"],
            "path": path,
        }
    return loaded


def seed_stat(sample_summary, key):
    return sample_summary[key]["seed_macro"]


def save_figure(fig, output_dir, stem, dpi):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(repo_root)

    aggregate_rows = []
    pair_rows = []
    for setting_key, setting in summaries.items():
        summary = setting["summary"]
        for sample_size in sorted(map(int, summary)):
            sample = summary[str(sample_size)]
            for metric, _, _ in AGGREGATE_METRICS:
                post_key = f"postlocal.logits.depth_summary.{metric}"
                delta_key = f"delta_from_global.depth_summary.{metric}"
                post = seed_stat(sample, post_key)
                delta = seed_stat(sample, delta_key)
                aggregate_rows.append({
                    "setting": setting_key,
                    "sample_size": sample_size,
                    "metric": metric,
                    "seed_count": post["count"],
                    "postlocal_mean": post["mean"],
                    "postlocal_std_across_seeds": post["std"],
                    "postlocal_ci95_half_width": post["ci95_half_width"],
                    "delta_from_global_mean": delta["mean"],
                    "delta_from_global_ci95_half_width": delta["ci95_half_width"],
                    "global_reference": post["mean"] - delta["mean"],
                })
            for pair in DEPTH_PAIRS:
                post_key = f"postlocal.logits.pairwise.{pair}.centered_logit_cosine_mean"
                delta_key = f"delta_from_global.pairwise_logits.{pair}.centered_logit_cosine_mean"
                post = seed_stat(sample, post_key)
                delta = seed_stat(sample, delta_key)
                pair_rows.append({
                    "setting": setting_key,
                    "sample_size": sample_size,
                    "pair": pair,
                    "seed_count": post["count"],
                    "postlocal_mean": post["mean"],
                    "postlocal_ci95_half_width": post["ci95_half_width"],
                    "delta_from_global_mean": delta["mean"],
                    "delta_from_global_ci95_half_width": delta["ci95_half_width"],
                    "global_reference": post["mean"] - delta["mean"],
                })

    with (output_dir / "aggregate_logit_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with (output_dir / "pairwise_logit_cosines.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    fixed_epoch_aggregate = [
        item for item in aggregate_rows if item["setting"] == "fixed_epoch"
    ]
    fixed_epoch_pairs = [
        item for item in pair_rows if item["setting"] == "fixed_epoch"
    ]
    with (output_dir / "fixed_epoch_aggregate_logit_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fixed_epoch_aggregate[0]))
        writer.writeheader()
        writer.writerows(fixed_epoch_aggregate)
    with (output_dir / "fixed_epoch_pairwise_logit_cosines.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fixed_epoch_pairs[0]))
        writer.writeheader()
        writer.writerows(fixed_epoch_pairs)

    # Aggregate metrics: columns are training budgets, rows are the two
    # algebraically linked depth-disagreement summaries.
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), sharex=True)
    for column, (setting_key, setting_title, _) in enumerate(SETTINGS):
        for row, (metric, y_label, short_title) in enumerate(AGGREGATE_METRICS):
            axis = axes[row, column]
            selected = [
                item for item in aggregate_rows
                if item["setting"] == setting_key and item["metric"] == metric
            ]
            selected.sort(key=lambda item: item["sample_size"])
            x = np.asarray([item["sample_size"] for item in selected])
            y = np.asarray([item["postlocal_mean"] for item in selected])
            ci = np.asarray([item["postlocal_ci95_half_width"] for item in selected])
            baseline = float(selected[0]["global_reference"])
            axis.errorbar(x, y, yerr=ci, marker="o", capsize=3, linewidth=1.8,
                          color="#2468b4", label="Post-local mean ± 95% CI")
            axis.axhline(baseline, linestyle="--", linewidth=1.4, color="#555555",
                         label="Round-start global")
            axis.set_xscale("log")
            axis.set_xticks(x, [f"{value:,}" for value in x])
            axis.grid(True, alpha=0.25)
            axis.set_title(f"{setting_title}: {short_title}")
            axis.set_ylabel(y_label)
            if row == 1:
                axis.set_xlabel("Unique local training samples")
            axis.legend(frameon=False, fontsize=8)
    save_figure(fig, output_dir, "aggregate_logit_metrics", args.dpi)

    # Paper-facing fixed-epoch-only aggregate figure.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35), sharex=True)
    for axis, (metric, y_label, short_title) in zip(axes, AGGREGATE_METRICS):
        selected = [
            item for item in fixed_epoch_aggregate if item["metric"] == metric
        ]
        selected.sort(key=lambda item: item["sample_size"])
        x = np.asarray([item["sample_size"] for item in selected])
        y = np.asarray([item["postlocal_mean"] for item in selected])
        ci = np.asarray([item["postlocal_ci95_half_width"] for item in selected])
        baseline = float(selected[0]["global_reference"])
        axis.errorbar(x, y, yerr=ci, marker="o", capsize=3, linewidth=1.8,
                      color="#2468b4", label="Post-local mean ± 95% CI")
        axis.axhline(baseline, linestyle="--", linewidth=1.4, color="#555555",
                     label="Round-start global")
        axis.set_xscale("log")
        axis.set_xticks(x, [f"{value:,}" for value in x])
        axis.set_xlabel("Unique local training samples (5 local epochs)")
        axis.set_ylabel(y_label)
        axis.set_title(f"Fixed epoch: {short_title}")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    save_figure(fig, output_dir, "fixed_epoch_aggregate_logit_metrics", args.dpi)

    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(DEPTH_PAIRS)))
    for value_name, y_label, stem, add_zero_line in (
        (
            "postlocal_mean",
            "Centered-logit cosine",
            "pairwise_logit_cosine_absolute",
            False,
        ),
        (
            "delta_from_global_mean",
            "Change in centered-logit cosine",
            "pairwise_logit_cosine_delta",
            True,
        ),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), sharex=True)
        for axis, (setting_key, setting_title, _) in zip(axes, SETTINGS):
            for color, pair in zip(colors, DEPTH_PAIRS):
                selected = [
                    item for item in pair_rows
                    if item["setting"] == setting_key and item["pair"] == pair
                ]
                selected.sort(key=lambda item: item["sample_size"])
                x = [item["sample_size"] for item in selected]
                y = [item[value_name] for item in selected]
                axis.plot(x, y, marker="o", markersize=3.5, linewidth=1.6,
                          color=color, label=pair.upper())
            if add_zero_line:
                axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2)
            axis.set_xscale("log")
            axis.set_xticks(x, [f"{value:,}" for value in x])
            axis.set_title(setting_title)
            axis.set_xlabel("Unique local training samples")
            axis.set_ylabel(y_label)
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=8, ncol=2)
        save_figure(fig, output_dir, stem, args.dpi)

    # Fixed-epoch pair decomposition, shown as a change from the common
    # round-start global checkpoint so the data-size effect is visible.
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    for color, pair in zip(colors, DEPTH_PAIRS):
        selected = [item for item in fixed_epoch_pairs if item["pair"] == pair]
        selected.sort(key=lambda item: item["sample_size"])
        x = [item["sample_size"] for item in selected]
        y = [item["delta_from_global_mean"] for item in selected]
        axis.plot(x, y, marker="o", markersize=3.8, linewidth=1.7,
                  color=color, label=pair.upper())
    axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2)
    axis.set_xscale("log")
    axis.set_xticks(x, [f"{value:,}" for value in x])
    axis.set_xlabel("Unique local training samples (5 local epochs)")
    axis.set_ylabel("Change in centered-logit cosine")
    axis.set_title("Fixed epoch: pairwise change from round-start global")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    save_figure(fig, output_dir, "fixed_epoch_pairwise_logit_cosine_delta", args.dpi)

    print(f"Wrote logit analysis tables and figures to {output_dir}")


if __name__ == "__main__":
    main()
