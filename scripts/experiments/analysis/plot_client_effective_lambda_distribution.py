#!/usr/bin/env python3
"""Plot last-window client-wise effective-lambda distributions."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dxfl-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARTITIONS = (
    ("iid", "IID", "#2563eb"),
    ("beta_0.1", r"$\beta=0.1$", "#dc2626"),
)


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def collect_client_means(data: dict, window: int) -> list[dict]:
    alpha_rounds = data.get("byot_effective_alpha_client_stats", [])
    reliability_rounds = data.get("byot_client_reliability_stats", [])
    skew_rounds = data.get("byot_prediction_entropy_client_stats", [])
    if not alpha_rounds or not any(alpha_rounds):
        raise ValueError(
            "No client-wise effective-lambda records found. "
            "Run the client-distribution experiment with the updated logger."
        )

    start = max(0, len(alpha_rounds) - window)
    by_client: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"alpha": [], "reliability": [], "prediction_entropy": []}
    )

    for round_idx in range(start, len(alpha_rounds)):
        alpha_stats = alpha_rounds[round_idx] or {}
        reliability_stats = (
            reliability_rounds[round_idx]
            if round_idx < len(reliability_rounds) and reliability_rounds[round_idx]
            else {}
        )
        skew_stats = (
            skew_rounds[round_idx]
            if round_idx < len(skew_rounds) and skew_rounds[round_idx]
            else {}
        )
        for client_id_raw, stats in alpha_stats.items():
            client_id = int(client_id_raw)
            by_client[client_id]["alpha"].append(float(stats["mean"]))

            r_stats = reliability_stats.get(client_id, reliability_stats.get(str(client_id), {}))
            if "reliability" in r_stats:
                by_client[client_id]["reliability"].append(float(r_stats["reliability"]))

            b_stats = skew_stats.get(client_id, skew_stats.get(str(client_id), {}))
            if "prediction_entropy" in b_stats:
                by_client[client_id]["prediction_entropy"].append(
                    float(b_stats["prediction_entropy"])
                )

    rows = []
    for client_id in sorted(by_client):
        values = by_client[client_id]
        if not values["alpha"]:
            continue
        rows.append(
            {
                "client_id": client_id,
                "mean_effective_lambda": float(np.mean(values["alpha"])),
                "observations": len(values["alpha"]),
                "mean_reliability": (
                    float(np.mean(values["reliability"]))
                    if values["reliability"] else None
                ),
                "mean_prediction_entropy": (
                    float(np.mean(values["prediction_entropy"]))
                    if values["prediction_entropy"] else None
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis/adaptive_lambda")
    )
    args = parser.parse_args()

    partition_rows: dict[str, tuple[str, str, list[dict]]] = {}
    for partition, label, color in PARTITIONS:
        path = args.log_root / partition / "fedavg" / f"{args.run_name}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        partition_rows[partition] = (
            label,
            color,
            collect_client_means(load_pickle(path), args.window),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "cifar100_client_effective_lambda_last30.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "partition",
            "client_id",
            "mean_effective_lambda",
            "observations",
            "mean_reliability",
            "mean_prediction_entropy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for partition, (_, _, rows) in partition_rows.items():
            for row in rows:
                writer.writerow({"partition": partition, **row})

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(8.4, 6.2), constrained_layout=True)

    distributions = []
    labels = []
    colors = []
    rows_in_order = []
    for _, (label, color, rows) in partition_rows.items():
        distributions.append(np.asarray([row["mean_effective_lambda"] for row in rows]))
        labels.append(f"{label}\n({len(rows)} observed clients)")
        colors.append(color)
        rows_in_order.append(rows)

    box = ax.boxplot(
        distributions,
        positions=np.arange(1, len(distributions) + 1),
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 2.0},
        whiskerprops={"color": "#475569", "linewidth": 1.2},
        capprops={"color": "#475569", "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor(color)

    for position, (values, rows, color) in enumerate(
        zip(distributions, rows_in_order, colors), start=1
    ):
        client_ids = np.asarray([row["client_id"] for row in rows], dtype=np.uint32)
        # Deterministic client-id jitter keeps the plot reproducible.
        jitter = ((client_ids * 2654435761 % 1009) / 1008.0 - 0.5) * 0.34
        ax.scatter(
            position + jitter,
            values,
            s=22,
            color=color,
            alpha=0.48,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        mean_value = float(np.mean(values))
        ax.scatter(
            position,
            mean_value,
            marker="D",
            s=62,
            color="#111827",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.text(
            position + 0.27,
            mean_value,
            f"mean={mean_value:.3f}",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#111827",
        )

    ax.set_title(f"CIFAR-100: Client-wise Effective $\lambda$ Distribution")
    ax.set_ylabel(f"Per-client mean effective $\lambda$ over last {args.window} rounds")
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set_xlim(0.45, len(labels) + 0.65)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.01,
        0.015,
        "Each point is one client; diamond denotes the client-macro mean.\n"
        "Clients not selected during the window are omitted.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#64748b",
    )

    stem = args.output_dir / "cifar100_client_effective_lambda_last30"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"PNG: {stem.with_suffix('.png')}")
    print(f"PDF: {stem.with_suffix('.pdf')}")
    print(f"CSV: {csv_path}")
    for partition, (label, _, rows) in partition_rows.items():
        values = [row["mean_effective_lambda"] for row in rows]
        observations = [row["observations"] for row in rows]
        print(
            f"{partition}: clients={len(rows)}, lambda_mean={np.mean(values):.6f}, "
            f"lambda_median={np.median(values):.6f}, "
            f"observations/client={np.mean(observations):.2f}"
        )


if __name__ == "__main__":
    main()
