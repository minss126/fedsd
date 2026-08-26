#!/usr/bin/env python3
"""Plot round-wise mean effective lambda for the final CIFAR-100 adaptive runs."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dxfl-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG_ROOT = Path(
    "logs/lambda/adaptive/logs_soft_adaptive_tuning_stage1"
)
DEFAULT_OUTPUT_DIR = Path("analysis/adaptive_lambda")
RUN_NAME = "soft_b_tkd1p00_lmax1p00_warm250_tau0p85.pkl"
PARTITIONS = (
    ("iid", "IID"),
    ("beta_0.5", r"$\beta=0.5$"),
    ("beta_0.3", r"$\beta=0.3$"),
    ("beta_0.1", r"$\beta=0.1$"),
)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centered moving average without shortening the series."""
    if window <= 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def load_effective_lambda(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        result = pickle.load(handle)
    if "byot_effective_alpha_mean" not in result:
        raise KeyError(f"Missing byot_effective_alpha_mean in {path}")
    values = np.asarray(result["byot_effective_alpha_mean"], dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Invalid effective-lambda series in {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--warmup-rounds", type=int, default=250)
    args = parser.parse_args()

    series: dict[str, tuple[str, np.ndarray]] = {}
    for partition, label in PARTITIONS:
        path = args.log_root / partition / "fedavg" / RUN_NAME
        if not path.is_file():
            raise FileNotFoundError(path)
        series[partition] = (label, load_effective_lambda(path))

    lengths = {len(values) for _, values in series.values()}
    if len(lengths) != 1:
        raise ValueError(f"Round counts differ across partitions: {sorted(lengths)}")
    rounds_total = lengths.pop()
    rounds = np.arange(1, rounds_total + 1)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.1), constrained_layout=True)

    ax.axvspan(
        1,
        args.warmup_rounds,
        color="#dbeafe",
        alpha=0.42,
        linewidth=0,
        label="Warm-up phase",
        zorder=0,
    )
    ax.axvline(
        args.warmup_rounds,
        color="#475569",
        linestyle="--",
        linewidth=1.25,
        alpha=0.85,
        zorder=1,
    )
    ax.text(
        args.warmup_rounds - 7,
        0.025,
        f"Warm-up ends (round {args.warmup_rounds})",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=9,
        color="#475569",
    )

    colors = ("#2563eb", "#16a34a", "#f59e0b", "#dc2626")
    last30_rows = []
    for color, (_, (label, values)) in zip(colors, series.items()):
        smoothed = moving_average(values, args.smooth_window)
        last30 = float(np.mean(values[-30:]))
        last30_rows.append((label, last30))

        ax.plot(rounds, values, color=color, alpha=0.13, linewidth=0.7, zorder=2)
        ax.plot(
            rounds,
            smoothed,
            color=color,
            linewidth=2.25,
            label=f"{label}  (last-30: {last30:.3f})",
            zorder=3,
        )
        ax.scatter(
            rounds[-1], smoothed[-1], s=28, color=color, edgecolor="white",
            linewidth=0.7, zorder=4
        )

    ax.set_title("CIFAR-100: Round-wise Mean Effective $\lambda$")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Mean effective $\lambda$")
    ax.set_xlim(1, rounds_total)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.65)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.6, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, ncol=2)
    ax.text(
        0.995,
        0.015,
        f"Bold lines: {args.smooth_window}-round centered moving average; faint lines: raw values",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#64748b",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "cifar100_roundwise_effective_lambda"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"PNG: {stem.with_suffix('.png')}")
    print(f"PDF: {stem.with_suffix('.pdf')}")
    for label, last30 in last30_rows:
        print(f"{label}: last-30 mean effective lambda = {last30:.6f}")


if __name__ == "__main__":
    main()
