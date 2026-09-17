#!/usr/bin/env python3
"""Plot round-wise effective-lambda dynamics from final JS-client runs.

The script creates a publication-oriented 1x3 dataset-wise trajectory figure.

Defaults correspond to the canonical seed-0, no-feature, T_KD=1,
lambda_max=1, tau=0.85 experiments.  No training is performed.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dxfl-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARTITIONS = (
    ("iid", "IID", "#2B2B2B", "-"),
    ("beta_0.3", r"$\beta=0.3$", "#4C78A8", "-"),
    ("beta_0.1", r"$\beta=0.1$", "#F58518", "-"),
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    rounds: int
    warmup: int
    smooth: int


DATASETS = (
    DatasetSpec("cifar100", "CIFAR-100", 500, 250, 15),
    DatasetSpec("tinyimagenet", "TinyImageNet", 100, 50, 5),
    DatasetSpec("imagenet100_64", "ImageNet100-64", 100, 50, 5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--extension-root",
        type=Path,
        default=Path("logs/lambda/final/logs_js_client_ofat_extensions_seed0"),
        help="Root containing final TinyImageNet/ImageNet100-64 dataset runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/adaptive_lambda/final_effective_lambda"),
    )
    parser.add_argument(
        "--show-raw", action="store_true", help="Overlay faint unsmoothed curves."
    )
    return parser.parse_args()


def absolute(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def resolve_run(repo_root: Path, extension_root: Path, dataset: str, partition: str) -> Path:
    """Resolve one canonical adaptive run without silently mixing configurations."""
    if dataset == "cifar100":
        if partition in {"iid", "beta_0.1"}:
            return (
                repo_root
                / "logs/lambda/adaptive/logs_js_granularity_temperature_canonical_no_feature"
                / partition
                / "seed0/js_client"
                / f"cifar100_{partition}_js_client_tkd1p00_canonical_nofeat_seed0_r500.pkl"
            )
        if partition == "beta_0.3":
            return (
                repo_root
                / "logs/lambda/adaptive/logs_js_client_lmax_crosscheck_canonical_no_feature"
                / "cifar100/beta_0.3/seed0/lmax1p00"
                / "cifar100_beta_0.3_js_client_lmax1p00_tau0p85_tkd1p00_canonical_nofeat_seed0_r500.pkl"
            )

    run_dir = (
        extension_root
        / "dataset"
        / dataset
        / dataset
        / partition
        / "seed0/adaptive_js_client"
    )
    matches = sorted(run_dir.glob("*_js_client_lmax1p00_tau0p85_tkd1p00_nofeat_canonical_seed0_*.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one canonical run under {run_dir}, found {len(matches)}"
        )
    return matches[0]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def effective_series(payload: dict, source: Path) -> np.ndarray:
    values = np.asarray(payload.get("byot_effective_alpha_mean", []), dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid byot_effective_alpha_mean in {source}")
    return values


def client_means(payload: dict, window: int, source: Path) -> list[dict[str, float | int]]:
    rounds = payload.get("byot_effective_alpha_client_stats", [])
    if not rounds or not any(rounds):
        raise ValueError(f"No client-wise effective-lambda stats in {source}")

    values: dict[int, list[float]] = defaultdict(list)
    for round_stats in rounds[max(0, len(rounds) - window) :]:
        for client_id, stats in (round_stats or {}).items():
            values[int(client_id)].append(float(stats["mean"]))
    return [
        {
            "client_id": client_id,
            "mean_effective_lambda": float(np.mean(client_values)),
            "observations": len(client_values),
        }
        for client_id, client_values in sorted(values.items())
        if client_values
    ]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def save_all(fig: plt.Figure, stem: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"dpi": 300} if suffix == ".png" else {}
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)


def plot_roundwise(
    payloads: dict[str, dict[str, tuple[dict, Path]]],
    output_dir: Path,
    show_raw: bool,
) -> Path:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.25))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.16, top=0.78, wspace=0.24)

    csv_rows: list[dict[str, object]] = []
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        ax.axvspan(1, dataset.warmup, color="#E5E7EB", alpha=0.62, linewidth=0)
        ax.axvline(dataset.warmup, color="#6B7280", linestyle="--", linewidth=1.0)
        maximum = 0.0
        for partition, label, color, linestyle in PARTITIONS:
            payload, source = payloads[dataset.key][partition]
            values = effective_series(payload, source)
            maximum = max(maximum, float(values.max()))
            rounds = np.arange(1, len(values) + 1)
            smooth = moving_average(values, dataset.smooth)
            if show_raw:
                ax.plot(rounds, values, color=color, alpha=0.13, linewidth=0.7)
            ax.plot(
                rounds,
                smooth,
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=label,
            )
            for round_number, (raw, smoothed) in enumerate(zip(values, smooth), 1):
                csv_rows.append(
                    {
                        "dataset": dataset.label,
                        "partition": partition,
                        "round": round_number,
                        "mean_effective_lambda": float(raw),
                        "smoothed_effective_lambda": float(smoothed),
                        "source": str(source),
                    }
                )

        upper = max(0.05, float(np.ceil(maximum * 1.08 / 0.05) * 0.05))
        ax.set_ylim(0, upper)
        ax.set_xlim(1, dataset.rounds)
        ax.set_title(dataset.label, fontweight="bold", pad=8)
        ax.set_xlabel("Communication round")
        if index == 0:
            ax.set_ylabel(r"Mean effective $\lambda$")
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=True,
        columnspacing=2.1,
        handlelength=3.0,
        handletextpad=0.8,
        borderpad=0.65,
    )
    legend.get_frame().set_edgecolor("#4B5563")
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_alpha(1.0)

    stem = output_dir / "final_effective_lambda_roundwise"
    save_all(fig, stem)
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    return stem


def plot_clientwise(
    payloads: dict[str, dict[str, tuple[dict, Path]]], output_dir: Path, last_window: int
) -> Path:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 3.9))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.19, top=0.72, wspace=0.30)
    csv_rows: list[dict[str, object]] = []

    for index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        distributions = []
        labels = []
        colors = []
        rows_by_partition = []
        for partition, label, color, _ in PARTITIONS:
            payload, source = payloads[dataset.key][partition]
            rows = client_means(payload, last_window, source)
            vals = np.asarray([float(row["mean_effective_lambda"]) for row in rows])
            distributions.append(vals)
            labels.append(f"{label}\n$n={len(vals)}$")
            colors.append(color)
            rows_by_partition.append(rows)
            for row in rows:
                csv_rows.append(
                    {
                        "dataset": dataset.label,
                        "partition": partition,
                        **row,
                        "source": str(source),
                    }
                )

        positions = np.arange(len(distributions), 0, -1)
        boxes = ax.boxplot(
            distributions,
            positions=positions,
            widths=0.42,
            vert=False,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 1.8},
            whiskerprops={"color": "#64748B", "linewidth": 1.0},
            capprops={"color": "#64748B", "linewidth": 1.0},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.78)

        for position, vals, rows, color in zip(positions, distributions, rows_by_partition, colors):
            ids = np.asarray([int(row["client_id"]) for row in rows], dtype=np.uint64)
            jitter = ((ids * 2654435761 % 1009) / 1008.0 - 0.5) * 0.32
            ax.scatter(vals, position + jitter, s=13, color=color, alpha=0.34, edgecolor="white", linewidth=0.25)
            mean = float(vals.mean())
            ax.scatter(mean, position, marker="D", s=38, color="#111827", edgecolor="white", linewidth=0.7, zorder=4)
            ax.text(mean, position + 0.19, f"{mean:.3f}", fontsize=8, ha="center", va="bottom")

        combined = np.concatenate(distributions)
        span = max(float(combined.max() - combined.min()), 0.05)
        ax.set_xlim(max(0, float(combined.min()) - 0.08 * span), float(combined.max()) + 0.12 * span)
        ax.set_yticks(positions, labels)
        ax.set_title(dataset.label, fontweight="bold", pad=8)
        ax.set_xlabel(rf"Client mean effective $\lambda$ (last {last_window} rounds)")
        if index == 0:
            ax.set_ylabel("Partition")
        ax.grid(axis="x", color="#CBD5E1", linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(r"Client-wise Effective $\lambda$ Distribution", fontsize=17, fontweight="bold", y=0.98)
    stem = output_dir / "final_effective_lambda_clientwise"
    save_all(fig, stem)
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    return stem


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    extension_root = absolute(repo_root, args.extension_root)
    output_dir = absolute(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads: dict[str, dict[str, tuple[dict, Path]]] = {}
    for dataset in DATASETS:
        payloads[dataset.key] = {}
        for partition, _, _, _ in PARTITIONS:
            path = resolve_run(repo_root, extension_root, dataset.key, partition)
            if not path.is_file():
                raise FileNotFoundError(path)
            payloads[dataset.key][partition] = (load_pickle(path), path)
            print(f"[load] {dataset.label:14s} {partition:8s} {path}")

    round_stem = plot_roundwise(payloads, output_dir, args.show_raw)
    print(f"\nRound-wise: {round_stem}.png/.pdf/.svg/.csv")


if __name__ == "__main__":
    main()
