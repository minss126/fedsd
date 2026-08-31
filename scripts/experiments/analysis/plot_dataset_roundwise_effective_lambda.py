#!/usr/bin/env python3
"""Plot compact round-wise effective-lambda curves for three datasets."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dxfl-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARTITIONS = (
    ("iid", "IID", "#244A73"),
    ("beta_0.5", r"$\beta=0.5$", "#6E9EC4"),
    ("beta_0.3", r"$\beta=0.3$", "#D28A8A"),
    ("beta_0.1", r"$\beta=0.1$", "#9E3D46"),
)


@dataclass(frozen=True)
class DatasetRun:
    key: str
    label: str
    root: Path
    filename: str
    warmup: int
    smooth_window: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/adaptive_lambda/presentation"),
    )
    parser.add_argument(
        "--stem",
        default="dataset_roundwise_effective_lambda_1x3",
    )
    parser.add_argument("--last-window", type=int, default=30)
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Overlay faint unsmoothed curves behind the moving averages.",
    )
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def load_effective_lambda(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    values = payload.get("byot_effective_alpha_mean")
    if values is None:
        raise KeyError(f"Missing byot_effective_alpha_mean in {path}")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"Invalid effective-lambda series in {path}")
    return array


def dataset_manifest(repo_root: Path) -> tuple[DatasetRun, ...]:
    adaptive = repo_root / "logs/lambda/adaptive"
    return (
        DatasetRun(
            key="cifar100",
            label="CIFAR-100",
            root=adaptive / "logs_soft_adaptive_tuning_stage1",
            filename="soft_b_tkd1p00_lmax1p00_warm250_tau0p85.pkl",
            warmup=250,
            smooth_window=15,
        ),
        DatasetRun(
            key="tinyimagenet",
            label="TinyImageNet",
            root=adaptive / "logs_extension_short_horizon_warmup50/tinyimagenet",
            filename="soft_b_tkd1p00_lmax1p00_warm50_tau0p85_r100.pkl",
            warmup=50,
            smooth_window=7,
        ),
        DatasetRun(
            key="imagenet100_64",
            label="ImageNet100-64",
            root=adaptive / "logs_extension_short_horizon_warmup50/imagenet100_64",
            filename="soft_b_tkd1p00_lmax1p00_warm50_tau0p85_r100.pkl",
            warmup=50,
            smooth_window=7,
        ),
    )


def compact_upper_limit(maximum: float) -> float:
    """Add a small headroom and round to a readable 0.05 increment."""
    return max(0.05, float(np.ceil((maximum * 1.08) / 0.05) * 0.05))


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = dataset_manifest(repo_root)
    all_series: dict[str, dict[str, np.ndarray]] = {}
    csv_rows: list[dict[str, object]] = []

    for dataset in datasets:
        all_series[dataset.key] = {}
        expected_length: int | None = None
        for partition, partition_label, _ in PARTITIONS:
            path = dataset.root / partition / "fedavg" / dataset.filename
            if not path.is_file():
                raise FileNotFoundError(path)
            values = load_effective_lambda(path)
            if expected_length is None:
                expected_length = len(values)
            elif len(values) != expected_length:
                raise ValueError(
                    f"Round counts differ for {dataset.label}: "
                    f"{expected_length} vs {len(values)} in {path}"
                )
            all_series[dataset.key][partition] = values
            smoothed = moving_average(values, dataset.smooth_window)
            for round_index, (raw, smooth) in enumerate(
                zip(values, smoothed), start=1
            ):
                csv_rows.append(
                    {
                        "dataset": dataset.label,
                        "partition": partition_label.replace("$", "").replace(
                            "\\beta", "beta"
                        ),
                        "round": round_index,
                        "mean_effective_lambda": float(raw),
                        "smoothed_effective_lambda": float(smooth),
                    }
                )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8))
    # Reserve separate vertical bands for the title, common legend, and panels.
    # Manual spacing avoids the legend/title collision produced by constrained_layout.
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.15,
        top=0.72,
        wspace=0.22,
    )

    for ax, dataset in zip(axes, datasets):
        dataset_values = all_series[dataset.key]
        rounds_total = len(next(iter(dataset_values.values())))
        rounds = np.arange(1, rounds_total + 1)
        maximum = max(float(np.max(values)) for values in dataset_values.values())

        ax.axvspan(
            1,
            dataset.warmup,
            color="#E5E7EB",
            alpha=0.58,
            linewidth=0,
            zorder=0,
        )
        ax.axvline(
            dataset.warmup,
            color="#6B7280",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )

        summary_lines = []
        for partition, label, color in PARTITIONS:
            values = dataset_values[partition]
            smoothed = moving_average(values, dataset.smooth_window)
            last_n = min(args.last_window, len(values))
            last_mean = float(np.mean(values[-last_n:]))
            summary_label = label.replace("$", "").replace("\\beta", "β")
            summary_lines.append(f"{summary_label}: {last_mean:.3f}")

            if args.show_raw:
                ax.plot(
                    rounds,
                    values,
                    color=color,
                    alpha=0.12,
                    linewidth=0.65,
                    zorder=2,
                )
            ax.plot(
                rounds,
                smoothed,
                color=color,
                linewidth=2.0,
                label=label,
                zorder=3,
            )

        ax.text(
            0.975,
            0.035,
            "Last-{}\n{}".format(args.last_window, "\n".join(summary_lines)),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.2,
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#D1D5DB",
                "alpha": 0.91,
            },
            zorder=5,
        )
        ax.text(
            dataset.warmup,
            0.975,
            f"Warm-up {dataset.warmup}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=8,
            color="#6B7280",
        )

        ax.set_title(dataset.label, fontweight="bold", pad=8)
        ax.set_xlabel("Communication round")
        ax.set_xlim(1, rounds_total)
        ax.set_ylim(0, compact_upper_limit(maximum))
        ax.margins(x=0)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.75, alpha=0.72)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9CA3AF")
        ax.spines["bottom"].set_color("#9CA3AF")

    axes[0].set_ylabel(r"Mean effective $\lambda$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.875),
    )
    fig.suptitle(
        r"Round-wise Mean Effective $\lambda$ across Datasets",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    stem = output_dir / args.stem
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    csv_path = stem.with_suffix(".csv")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"SVG: {svg_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
