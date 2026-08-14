#!/usr/bin/env python3
"""Compare fixed-checkpoint CIFAR-10/100 within-model logit diagnostics."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dxfl-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAIRS = ("b1-b2", "b1-b3", "b2-b3", "b1-final", "b2-final", "b3-final")
DATASETS = (
    ("cifar10", "CIFAR-10", "#2878b5"),
    ("cifar100", "CIFAR-100", "#d9534f"),
)
BUDGETS = (("fixed_step", "Fixed step"), ("fixed_epoch", "Fixed epoch"))
ROOTS = {
    ("cifar10", "fixed_step"): (
        "logs/analysis/logs_cifar10_fixed_checkpoint_motivation/fixed_step"
    ),
    ("cifar10", "fixed_epoch"): (
        "logs/analysis/logs_cifar10_fixed_checkpoint_motivation/fixed_epoch"
    ),
    ("cifar100", "fixed_step"): (
        "logs/analysis/logs_local_data_size_internal_probe"
    ),
    ("cifar100", "fixed_epoch"): (
        "logs/analysis/logs_local_data_size_internal_probe_epochs"
    ),
}
AGGREGATES = (
    (
        "off_diagonal_centered_cosine_mean",
        "Mean pairwise centered-logit cosine",
        "pairwise_cosine_dataset_comparison",
    ),
    (
        "directional_variance_mean",
        "Directional logit variance",
        "directional_variance_dataset_comparison",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument(
        "--output_dir",
        default="logs/analysis/cifar10_cifar100_checkpoint_logit_comparison",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def seed_stat(summary, sample_size, metric):
    return summary[str(sample_size)][metric]["seed_macro"]


def save_figure(fig, output_dir, stem, dpi):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{value:.4f}"


def paired_endpoint_delta(summary, metric, low=100, high=2500):
    low_seeds = summary[str(low)][metric]["seed_means"]
    high_seeds = summary[str(high)][metric]["seed_means"]
    seeds = sorted(set(low_seeds) & set(high_seeds), key=int)
    differences = np.asarray(
        [float(high_seeds[seed]) - float(low_seeds[seed]) for seed in seeds],
        dtype=np.float64,
    )
    std = float(differences.std(ddof=1)) if differences.size > 1 else 0.0
    # Two-sided Student-t 95% critical value for three paired seed means.
    critical = 4.303 if differences.size == 3 else 1.96
    return {
        "count": int(differences.size),
        "mean": float(differences.mean()),
        "std": std,
        "ci95_half_width": (
            float(critical * std / math.sqrt(differences.size))
            if differences.size > 1 else 0.0
        ),
    }


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for key, relative_root in ROOTS.items():
        path = repo_root / relative_root / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            summaries[key] = json.load(handle)["summary_by_sample_size"]

    aggregate_rows = []
    pair_rows = []
    global_rows = []
    endpoint_rows = []
    for dataset_key, dataset_title, _ in DATASETS:
        for budget_key, budget_title in BUDGETS:
            summary = summaries[(dataset_key, budget_key)]
            sizes = sorted(map(int, summary))
            for metric, _, _ in AGGREGATES:
                post_key = f"postlocal.logits.depth_summary.{metric}"
                delta_key = f"delta_from_global.depth_summary.{metric}"
                for sample_size in sizes:
                    post = seed_stat(summary, sample_size, post_key)
                    delta = seed_stat(summary, sample_size, delta_key)
                    aggregate_rows.append({
                        "dataset": dataset_key,
                        "budget": budget_key,
                        "sample_size": sample_size,
                        "metric": metric,
                        "mean": post["mean"],
                        "seed_macro_ci95_half_width": post["ci95_half_width"],
                        "delta_from_global_mean": delta["mean"],
                        "delta_from_global_ci95_half_width": delta["ci95_half_width"],
                        "global_reference": post["mean"] - delta["mean"],
                    })
                endpoint = paired_endpoint_delta(summary, post_key)
                endpoint_rows.append({
                    "dataset": dataset_key,
                    "budget": budget_key,
                    "category": "aggregate",
                    "metric": metric,
                    "delta_definition": "value_at_2500_minus_value_at_100",
                    "paired_seed_count": endpoint["count"],
                    "delta_mean": endpoint["mean"],
                    "delta_std_across_paired_seeds": endpoint["std"],
                    "delta_ci95_half_width": endpoint["ci95_half_width"],
                })
            for pair in PAIRS:
                post_key = f"postlocal.logits.pairwise.{pair}.centered_logit_cosine_mean"
                delta_key = (
                    f"delta_from_global.pairwise_logits.{pair}."
                    "centered_logit_cosine_mean"
                )
                for sample_size in sizes:
                    post = seed_stat(summary, sample_size, post_key)
                    delta = seed_stat(summary, sample_size, delta_key)
                    pair_rows.append({
                        "dataset": dataset_key,
                        "budget": budget_key,
                        "sample_size": sample_size,
                        "pair": pair,
                        "mean": post["mean"],
                        "seed_macro_ci95_half_width": post["ci95_half_width"],
                        "delta_from_global_mean": delta["mean"],
                        "delta_from_global_ci95_half_width": delta["ci95_half_width"],
                        "global_reference": post["mean"] - delta["mean"],
                    })
                endpoint = paired_endpoint_delta(summary, post_key)
                endpoint_rows.append({
                    "dataset": dataset_key,
                    "budget": budget_key,
                    "category": "branch_pair_cosine",
                    "metric": pair,
                    "delta_definition": "value_at_2500_minus_value_at_100",
                    "paired_seed_count": endpoint["count"],
                    "delta_mean": endpoint["mean"],
                    "delta_std_across_paired_seeds": endpoint["std"],
                    "delta_ci95_half_width": endpoint["ci95_half_width"],
                })

        # The fixed-step and fixed-epoch runs share one checkpoint. Use the
        # fixed-step summary once to avoid duplicating global-reference rows.
        summary = summaries[(dataset_key, "fixed_step")]
        first_size = min(map(int, summary))
        for metric, _, _ in AGGREGATES:
            post = seed_stat(
                summary, first_size, f"postlocal.logits.depth_summary.{metric}"
            )["mean"]
            delta = seed_stat(
                summary, first_size, f"delta_from_global.depth_summary.{metric}"
            )["mean"]
            global_rows.append({
                "dataset": dataset_key,
                "category": "aggregate",
                "metric": metric,
                "value": post - delta,
            })
        for pair in PAIRS:
            post = seed_stat(
                summary,
                first_size,
                f"postlocal.logits.pairwise.{pair}.centered_logit_cosine_mean",
            )["mean"]
            delta = seed_stat(
                summary,
                first_size,
                (
                    f"delta_from_global.pairwise_logits.{pair}."
                    "centered_logit_cosine_mean"
                ),
            )["mean"]
            global_rows.append({
                "dataset": dataset_key,
                "category": "pairwise_cosine",
                "metric": pair,
                "value": post - delta,
            })

    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output_dir / "branch_pair_cosines.csv", pair_rows)
    write_csv(output_dir / "global_checkpoint_metrics.csv", global_rows)
    write_csv(output_dir / "endpoint_deltas_2500_minus_100.csv", endpoint_rows)

    # 1-2. Dataset comparison within fixed-step and fixed-epoch panels.
    for metric, y_label, stem in AGGREGATES:
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.65), sharex=True, sharey=True)
        for axis, (budget_key, budget_title) in zip(axes, BUDGETS):
            for dataset_key, dataset_title, color in DATASETS:
                selected = [
                    row for row in aggregate_rows
                    if row["dataset"] == dataset_key
                    and row["budget"] == budget_key
                    and row["metric"] == metric
                ]
                selected.sort(key=lambda row: row["sample_size"])
                x = np.asarray([row["sample_size"] for row in selected])
                y = np.asarray([row["mean"] for row in selected])
                ci = np.asarray(
                    [row["seed_macro_ci95_half_width"] for row in selected]
                )
                baseline = float(selected[0]["global_reference"])
                axis.errorbar(
                    x, y, yerr=ci, marker="o", markersize=4.2, capsize=3,
                    linewidth=1.9, color=color, label=dataset_title,
                )
                axis.axhline(
                    baseline, color=color, linestyle="--", alpha=0.65,
                    linewidth=1.25, label=f"{dataset_title} global",
                )
            axis.set_xscale("log")
            axis.set_xticks(x, [f"{value:,}" for value in x])
            axis.set_xlabel("Unique local training samples")
            axis.set_title(budget_title)
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=8, ncol=2)
        axes[0].set_ylabel(y_label)
        fig.suptitle(f"CIFAR-10 vs CIFAR-100: {y_label}", y=1.01)
        save_figure(fig, output_dir, stem, args.dpi)

    # 3. Required panel order: C10 step, C10 epoch, C100 step, C100 epoch.
    panel_order = (
        ("cifar10", "fixed_step", "CIFAR-10 — Fixed step"),
        ("cifar10", "fixed_epoch", "CIFAR-10 — Fixed epoch"),
        ("cifar100", "fixed_step", "CIFAR-100 — Fixed step"),
        ("cifar100", "fixed_epoch", "CIFAR-100 — Fixed epoch"),
    )
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(PAIRS)))
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 8.6), sharex=True, sharey=True)
    for axis, (dataset_key, budget_key, title) in zip(axes.flat, panel_order):
        for color, pair in zip(colors, PAIRS):
            selected = [
                row for row in pair_rows
                if row["dataset"] == dataset_key
                and row["budget"] == budget_key
                and row["pair"] == pair
            ]
            selected.sort(key=lambda row: row["sample_size"])
            x = [row["sample_size"] for row in selected]
            y = [row["mean"] for row in selected]
            axis.plot(
                x, y, marker="o", markersize=3.8, linewidth=1.65,
                color=color, label=pair.upper(),
            )
        axis.set_xscale("log")
        axis.set_xticks(x, [f"{value:,}" for value in x])
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=7.8, ncol=2)
    for axis in axes[1]:
        axis.set_xlabel("Unique local training samples")
    for axis in axes[:, 0]:
        axis.set_ylabel("Centered-logit cosine")
    fig.suptitle("Branch-pair cosine by dataset and local-training budget", y=1.005)
    save_figure(fig, output_dir, "branch_pair_cosine_comparison", args.dpi)

    # Endpoint-effect summary: paired change from n=100 to n=2,500.
    condition_order = (
        ("cifar10", "fixed_step", "Step\nC10", "#2878b5"),
        ("cifar100", "fixed_step", "Step\nC100", "#d9534f"),
        ("cifar10", "fixed_epoch", "Epoch\nC10", "#63a8d3"),
        ("cifar100", "fixed_epoch", "Epoch\nC100", "#e58b87"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 4.65))
    for axis, (metric, y_label, _) in zip(axes, AGGREGATES):
        selected = [
            next(
                row for row in endpoint_rows
                if row["dataset"] == dataset_key
                and row["budget"] == budget_key
                and row["category"] == "aggregate"
                and row["metric"] == metric
            )
            for dataset_key, budget_key, _, _ in condition_order
        ]
        positions = np.arange(len(selected))
        values = [row["delta_mean"] for row in selected]
        errors = [row["delta_ci95_half_width"] for row in selected]
        axis.bar(
            positions, values, yerr=errors, capsize=4,
            color=[item[3] for item in condition_order], alpha=0.9,
        )
        axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.1)
        axis.set_xticks(positions, [item[2] for item in condition_order])
        axis.set_ylabel(f"Δ {y_label}\n(n=2,500 minus n=100)")
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].set_title("Mean pairwise cosine endpoint change")
    axes[1].set_title("Directional variance endpoint change")
    save_figure(fig, output_dir, "aggregate_endpoint_delta_2500_minus_100", args.dpi)

    matrix = np.zeros((len(PAIRS), len(condition_order)), dtype=np.float64)
    for row_index, pair in enumerate(PAIRS):
        for column_index, (dataset_key, budget_key, _, _) in enumerate(condition_order):
            matrix[row_index, column_index] = next(
                row["delta_mean"] for row in endpoint_rows
                if row["dataset"] == dataset_key
                and row["budget"] == budget_key
                and row["category"] == "branch_pair_cosine"
                and row["metric"] == pair
            )
    limit = float(np.abs(matrix).max())
    fig, axis = plt.subplots(figsize=(7.8, 5.4))
    image = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(condition_order)), [item[2].replace("\n", " ") for item in condition_order])
    axis.set_yticks(np.arange(len(PAIRS)), [pair.upper() for pair in PAIRS])
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index, row_index, f"{value:+.4f}",
                ha="center", va="center",
                color="white" if abs(value) > 0.55 * limit else "black",
                fontsize=9,
            )
    axis.set_title("Branch-pair cosine Δ (n=2,500 minus n=100)")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Paired endpoint difference")
    save_figure(fig, output_dir, "branch_pair_endpoint_delta_heatmap", args.dpi)

    # Additional global-checkpoint comparison. This exposes whether the depth
    # gap exists before the controlled post-checkpoint local update.
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    positions = np.arange(len(PAIRS))
    width = 0.36
    for offset, (dataset_key, dataset_title, color) in zip((-0.5, 0.5), DATASETS):
        values = [
            next(
                row["value"] for row in global_rows
                if row["dataset"] == dataset_key
                and row["category"] == "pairwise_cosine"
                and row["metric"] == pair
            )
            for pair in PAIRS
        ]
        axis.bar(
            positions + offset * width, values, width=width,
            color=color, alpha=0.88, label=dataset_title,
        )
    axis.set_xticks(positions, [pair.upper() for pair in PAIRS])
    axis.set_ylabel("Global-checkpoint centered-logit cosine")
    axis.set_title("Cross-depth discrepancy already present at the global checkpoint")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False)
    save_figure(fig, output_dir, "global_checkpoint_pairwise_cosine", args.dpi)

    aggregate_lookup = {
        (row["dataset"], row["metric"]): row["value"]
        for row in global_rows if row["category"] == "aggregate"
    }
    pair_lookup = {
        (row["dataset"], row["metric"]): row["value"]
        for row in global_rows if row["category"] == "pairwise_cosine"
    }
    report = [
        "# CIFAR-10 / CIFAR-100 fixed-checkpoint logit comparison",
        "",
        "Values are seed-macro means. Error terms in CSV/figures are two-sided ",
        "Student-t 95% CIs over three seed-level client means.",
        "",
    ]
    for metric, title, _ in AGGREGATES:
        report.extend([f"## {title}", ""])
        report.append(
            "| Local samples | Fixed step: CIFAR-10 | Fixed step: CIFAR-100 | "
            "Fixed epoch: CIFAR-10 | Fixed epoch: CIFAR-100 |"
        )
        report.append("|---:|---:|---:|---:|---:|")
        for sample_size in sorted(map(int, summaries[("cifar10", "fixed_step")])):
            cells = []
            for dataset_key, budget_key in (
                ("cifar10", "fixed_step"), ("cifar100", "fixed_step"),
                ("cifar10", "fixed_epoch"), ("cifar100", "fixed_epoch"),
            ):
                selected = next(
                    row for row in aggregate_rows
                    if row["dataset"] == dataset_key
                    and row["budget"] == budget_key
                    and row["sample_size"] == sample_size
                    and row["metric"] == metric
                )
                cells.append(
                    f"{fmt(selected['mean'])} ± "
                    f"{fmt(selected['seed_macro_ci95_half_width'])}"
                )
            report.append(f"| {sample_size:,} | " + " | ".join(cells) + " |")
        delta_cells = []
        for dataset_key, budget_key in (
            ("cifar10", "fixed_step"), ("cifar100", "fixed_step"),
            ("cifar10", "fixed_epoch"), ("cifar100", "fixed_epoch"),
        ):
            endpoint = next(
                row for row in endpoint_rows
                if row["dataset"] == dataset_key
                and row["budget"] == budget_key
                and row["category"] == "aggregate"
                and row["metric"] == metric
            )
            delta_cells.append(
                f"{endpoint['delta_mean']:+.4f} ± "
                f"{endpoint['delta_ci95_half_width']:.4f}"
            )
        report.append("| **Δ (2,500 − 100)** | " + " | ".join(delta_cells) + " |")
        report.append("")

    report.extend([
        "## Branch-pair cosine endpoint change",
        "",
        "All entries are paired `value(n=2,500) − value(n=100)` differences.",
        "",
        "| Pair | Fixed step: CIFAR-10 | Fixed step: CIFAR-100 | Fixed epoch: CIFAR-10 | Fixed epoch: CIFAR-100 |",
        "|---|---:|---:|---:|---:|",
    ])
    for pair in PAIRS:
        cells = []
        for dataset_key, budget_key in (
            ("cifar10", "fixed_step"), ("cifar100", "fixed_step"),
            ("cifar10", "fixed_epoch"), ("cifar100", "fixed_epoch"),
        ):
            endpoint = next(
                row for row in endpoint_rows
                if row["dataset"] == dataset_key
                and row["budget"] == budget_key
                and row["category"] == "branch_pair_cosine"
                and row["metric"] == pair
            )
            cells.append(
                f"{endpoint['delta_mean']:+.4f} ± "
                f"{endpoint['delta_ci95_half_width']:.4f}"
            )
        report.append(f"| {pair.upper()} | " + " | ".join(cells) + " |")
    report.append("")

    report.extend([
        "## Global checkpoint aggregate metrics",
        "",
        "| Dataset | Mean pairwise cosine | Directional variance |",
        "|---|---:|---:|",
    ])
    for dataset_key, dataset_title, _ in DATASETS:
        report.append(
            f"| {dataset_title} | "
            f"{fmt(aggregate_lookup[(dataset_key, 'off_diagonal_centered_cosine_mean')])} | "
            f"{fmt(aggregate_lookup[(dataset_key, 'directional_variance_mean')])} |"
        )
    report.extend([
        "",
        "## Global checkpoint pairwise cosine",
        "",
        "| Pair | CIFAR-10 | CIFAR-100 | CIFAR-100 − CIFAR-10 |",
        "|---|---:|---:|---:|",
    ])
    for pair in PAIRS:
        c10 = pair_lookup[("cifar10", pair)]
        c100 = pair_lookup[("cifar100", pair)]
        report.append(f"| {pair.upper()} | {fmt(c10)} | {fmt(c100)} | {c100-c10:+.4f} |")
    report.extend([
        "",
        "Directional variance is algebraically redundant with the aggregate cosine ",
        "under the current centered/unit-normalized four-depth definition: ",
        "`variance = 3/4 × (1 − mean pairwise cosine)`.",
        "",
    ])
    (output_dir / "comparison_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"Wrote comparison tables and figures to {output_dir}")


if __name__ == "__main__":
    main()
