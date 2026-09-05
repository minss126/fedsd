#!/usr/bin/env python3
"""Merge and plot aggregation-induced degradation pilot outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "Plain": "#6B7280",
    "Adaptive": "#F28E2B",
}
METHOD_MARKERS = {"Plain": "o", "Adaptive": "s"}
STAGE_ORDER = ["B1", "B2", "B3"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", required=True,
        help="Root containing run-local aggregation-damage CSV directories.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _read_recursive(input_root: Path, output_dir: Path, name: str) -> pd.DataFrame:
    frames = []
    output_dir = output_dir.resolve()
    for path in sorted(input_root.rglob(name)):
        if path.parent.resolve() == output_dir:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frame["source_csv"] = str(path)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    value_columns = [column for column in merged.columns if column != "source_csv"]
    return merged.drop_duplicates(subset=value_columns, keep="last")


def _ordered(values, preferred):
    found = list(dict.fromkeys(values))
    ordered = [value for value in preferred if value in found]
    ordered.extend(value for value in found if value not in ordered)
    return ordered


def _style_axis(axis):
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _shared_legend(figure, axes, methods):
    handles, labels = [], []
    for axis in np.asarray(axes).reshape(-1):
        current_handles, current_labels = axis.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    wanted = [method for method in methods if method in labels]
    ordered_handles = [handles[labels.index(label)] for label in wanted]
    if ordered_handles:
        figure.legend(
            ordered_handles, wanted, loc="upper center", ncol=len(wanted),
            frameon=False, bbox_to_anchor=(0.5, 1.01),
        )


def plot_round_metric(model: pd.DataFrame, metric: str, output: Path, dpi: int):
    partitions = _ordered(
        model["partition"].unique(), ["IID", "Gamma = 0.1", "Beta = 0.1"]
    )
    epochs = sorted(int(value) for value in model["local_epoch"].unique())
    methods = _ordered(model["method"].unique(), ["Plain", "Adaptive"])
    figure, axes = plt.subplots(
        len(partitions), len(epochs),
        figsize=(4.2 * len(epochs), 3.1 * len(partitions)),
        squeeze=False, sharex=True,
    )
    for row, partition in enumerate(partitions):
        for column, epoch in enumerate(epochs):
            axis = axes[row, column]
            subset = model[
                (model["partition"] == partition)
                & (model["local_epoch"].astype(int) == epoch)
            ]
            for method in methods:
                selected = subset[subset["method"] == method]
                if selected.empty:
                    continue
                grouped = selected.groupby("round")[metric].agg(["mean", "std"]).reset_index()
                axis.plot(
                    grouped["round"], grouped["mean"],
                    label=method,
                    color=METHOD_COLORS.get(method),
                    marker=METHOD_MARKERS.get(method, "o"),
                    linewidth=1.9, markersize=4.5,
                )
                if grouped["std"].notna().any():
                    std = grouped["std"].fillna(0.0)
                    axis.fill_between(
                        grouped["round"], grouped["mean"] - std,
                        grouped["mean"] + std,
                        color=METHOD_COLORS.get(method), alpha=0.14,
                    )
            axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.65)
            axis.set_title(f"{partition} · Local epoch {epoch}", fontsize=10.5)
            if row == len(partitions) - 1:
                axis.set_xlabel("Completed communication rounds")
            if column == 0:
                axis.set_ylabel(metric.replace("_", " "))
            _style_axis(axis)
    _shared_legend(figure, axes, methods)
    figure.suptitle(metric.replace("_", " ").title(), y=1.035, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_stage_metric(summary: pd.DataFrame, metric: str, output: Path, dpi: int):
    subset = summary[summary["replacement_source"] == "post_aggregation_global"].copy()
    if subset.empty:
        return
    partitions = _ordered(
        subset["partition"].unique(), ["IID", "Gamma = 0.1", "Beta = 0.1"]
    )
    epochs = sorted(int(value) for value in subset["local_epoch"].unique())
    methods = _ordered(subset["method"].unique(), ["Plain", "Adaptive"])
    figure, axes = plt.subplots(
        len(partitions), len(epochs),
        figsize=(4.2 * len(epochs), 3.1 * len(partitions)),
        squeeze=False, sharex=True,
    )
    x = np.arange(len(STAGE_ORDER))
    for row, partition in enumerate(partitions):
        for column, epoch in enumerate(epochs):
            axis = axes[row, column]
            current = subset[
                (subset["partition"] == partition)
                & (subset["local_epoch"].astype(int) == epoch)
            ]
            for method in methods:
                selected = current[current["method"] == method]
                means, errors = [], []
                for stage in STAGE_ORDER:
                    values = selected[selected["stage"] == stage][metric].astype(float)
                    means.append(float(values.mean()) if len(values) else np.nan)
                    errors.append(float(values.std(ddof=0)) if len(values) > 1 else 0.0)
                axis.errorbar(
                    x, means, yerr=errors,
                    label=method,
                    color=METHOD_COLORS.get(method),
                    marker=METHOD_MARKERS.get(method, "o"),
                    linewidth=1.9, markersize=5, capsize=3,
                )
            axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.65)
            axis.set_xticks(x, STAGE_ORDER)
            axis.set_title(f"{partition} · Local epoch {epoch}", fontsize=10.5)
            if row == len(partitions) - 1:
                axis.set_xlabel("Replaced backbone stage")
            if column == 0:
                axis.set_ylabel(metric.replace("_", " "))
            _style_axis(axis)
    _shared_legend(figure, axes, methods)
    figure.suptitle(
        "Post-aggregation Stage Replacement Damage", y=1.035, fontweight="bold"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_local_epoch_model_damage(model: pd.DataFrame, output: Path, dpi: int):
    metrics = ["aggregation_loss_gap", "aggregation_acc_gap"]
    partitions = _ordered(
        model["partition"].unique(), ["IID", "Gamma = 0.1", "Beta = 0.1"]
    )
    methods = _ordered(model["method"].unique(), ["Plain", "Adaptive"])
    figure, axes = plt.subplots(
        len(partitions), len(metrics),
        figsize=(8.8, 3.15 * len(partitions)), squeeze=False,
    )
    for row, partition in enumerate(partitions):
        for column, metric in enumerate(metrics):
            axis = axes[row, column]
            current = model[model["partition"] == partition]
            for method in methods:
                selected = current[current["method"] == method]
                grouped = selected.groupby("local_epoch")[metric].agg(["mean", "std"]).reset_index()
                axis.errorbar(
                    grouped["local_epoch"], grouped["mean"],
                    yerr=grouped["std"].fillna(0.0),
                    label=method, color=METHOD_COLORS.get(method),
                    marker=METHOD_MARKERS.get(method, "o"),
                    linewidth=1.9, markersize=5, capsize=3,
                )
            axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.65)
            axis.set_title(f"{partition} · {metric.replace('_', ' ')}", fontsize=10.5)
            axis.set_xlabel("Local epochs")
            axis.set_ylabel(metric.replace("_", " "))
            axis.set_xticks(sorted(int(value) for value in model["local_epoch"].unique()))
            _style_axis(axis)
    _shared_legend(figure, axes, methods)
    figure.suptitle("Local Epoch vs Model-level Aggregation Damage", y=1.025, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_local_epoch_intermediate_damage(summary: pd.DataFrame, output: Path, dpi: int):
    subset = summary[summary["replacement_source"] == "post_aggregation_global"].copy()
    if subset.empty:
        return
    metrics = [
        "replacement_loss_damage_mean",
        "replacement_acc_damage_mean",
    ]
    partitions = _ordered(
        subset["partition"].unique(), ["IID", "Gamma = 0.1", "Beta = 0.1"]
    )
    methods = _ordered(subset["method"].unique(), ["Plain", "Adaptive"])
    figure, axes = plt.subplots(
        len(partitions), len(metrics),
        figsize=(8.8, 3.15 * len(partitions)), squeeze=False,
    )
    for row, partition in enumerate(partitions):
        for column, metric in enumerate(metrics):
            axis = axes[row, column]
            current = subset[subset["partition"] == partition]
            for method in methods:
                selected = current[current["method"] == method]
                grouped = selected.groupby("local_epoch")[metric].agg(["mean", "std"]).reset_index()
                axis.errorbar(
                    grouped["local_epoch"], grouped["mean"],
                    yerr=grouped["std"].fillna(0.0),
                    label=method, color=METHOD_COLORS.get(method),
                    marker=METHOD_MARKERS.get(method, "o"),
                    linewidth=1.9, markersize=5, capsize=3,
                )
            axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.65)
            axis.set_title(f"{partition} · {metric.replace('_', ' ')}", fontsize=10.5)
            axis.set_xlabel("Local epochs")
            axis.set_ylabel(metric.replace("_", " "))
            axis.set_xticks(sorted(int(value) for value in subset["local_epoch"].unique()))
            _style_axis(axis)
    _shared_legend(figure, axes, methods)
    figure.suptitle(
        "Local Epoch vs Mean Intermediate Replacement Damage",
        y=1.025, fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _read_recursive(input_root, output_dir, "model_level.csv")
    raw = _read_recursive(input_root, output_dir, "block_replacement_raw.csv")
    summary = _read_recursive(input_root, output_dir, "block_replacement_summary.csv")
    if model.empty or raw.empty or summary.empty:
        raise SystemExit(
            "Could not find non-empty model_level.csv, block_replacement_raw.csv, "
            "and block_replacement_summary.csv under the input root."
        )

    for frame in (model, raw, summary):
        frame.drop(columns=["source_csv"], inplace=True, errors="ignore")
    model.to_csv(output_dir / "model_level.csv", index=False)
    raw.to_csv(output_dir / "block_replacement_raw.csv", index=False)
    summary.to_csv(output_dir / "block_replacement_summary.csv", index=False)

    plot_round_metric(
        model, "aggregation_loss_gap",
        output_dir / "round_vs_aggregation_loss_gap.png", args.dpi,
    )
    plot_round_metric(
        model, "aggregation_acc_gap",
        output_dir / "round_vs_aggregation_acc_gap.png", args.dpi,
    )
    plot_stage_metric(
        summary, "replacement_loss_damage_mean",
        output_dir / "stage_vs_replacement_loss_damage.png", args.dpi,
    )
    plot_stage_metric(
        summary, "replacement_acc_damage_mean",
        output_dir / "stage_vs_replacement_acc_damage.png", args.dpi,
    )
    plot_local_epoch_model_damage(
        model, output_dir / "local_epoch_vs_aggregation_damage.png", args.dpi,
    )
    plot_local_epoch_intermediate_damage(
        summary,
        output_dir / "local_epoch_vs_mean_intermediate_replacement_damage.png",
        args.dpi,
    )
    print(f"Merged CSVs and plots: {output_dir}")


if __name__ == "__main__":
    main()
