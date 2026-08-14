#!/usr/bin/env python3
"""Summarize the round-checkpoint logit/CKA motivation experiment."""

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


DATASETS = (("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100"))
BUDGETS = (("fixed_step", "Fixed step"), ("fixed_epoch", "Fixed epoch"))
ROUNDS = (0, 10, 50, 100)
SIZES = (100, 250, 500, 1000, 2500)
PAIRS = ("b1-b2", "b1-b3", "b2-b3", "b1-final", "b2-final", "b3-final")
METRICS = (
    (
        "mean_logit_cosine",
        "postlocal.logits.depth_summary.off_diagonal_centered_cosine_mean",
        "Mean pairwise centered-logit cosine",
    ),
    (
        "directional_logit_variance",
        "postlocal.logits.depth_summary.directional_variance_mean",
        "Directional logit variance",
    ),
    (
        "mean_linear_cka",
        "postlocal.cka.depth_summary.off_diagonal_linear_cka_mean",
        "Mean pairwise linear CKA",
    ),
)
ROUND_COLORS = {0: "#6c757d", 10: "#2a9d8f", 50: "#e9c46a", 100: "#e76f51"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root",
        default="logs/analysis/logs_round_checkpoint_logit_cka",
    )
    parser.add_argument(
        "--output_dir",
        default="logs/analysis/round_checkpoint_logit_cka_analysis",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def nested_get(mapping, dotted_key):
    value = mapping
    for part in dotted_key.split("."):
        value = value[part]
    return value


def summary_stat(summary, size, key):
    return summary[str(size)][key]["seed_macro"]


def paired_endpoint(summary, key):
    low = summary["100"][key]["seed_means"]
    high = summary["2500"][key]["seed_means"]
    seeds = sorted(set(low) & set(high), key=int)
    differences = np.asarray([high[seed] - low[seed] for seed in seeds], dtype=float)
    std = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    critical = 4.303 if len(differences) == 3 else 1.96
    return {
        "mean": float(differences.mean()),
        "ci95_half_width": critical * std / math.sqrt(len(differences)),
        "seed_count": len(differences),
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output_dir, stem, dpi):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    input_root = (repo_root / args.input_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    globals_by_round = {}
    metric_files = []
    for dataset, _ in DATASETS:
        for completed_rounds in ROUNDS:
            round_dir = input_root / dataset / f"round_{completed_rounds:04d}"
            representative = (
                round_dir / "fixed_step" / "sample_100" / "seed_0" / "metrics.json"
            )
            with representative.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            assert payload["metrics"] == "logits_cka"
            assert payload["global_completed_rounds"] == completed_rounds
            assert payload["reference_test_samples"] == 10000
            globals_by_round[(dataset, completed_rounds)] = payload[
                "global_reference_metrics"
            ]
            for budget, _ in BUDGETS:
                summary_path = round_dir / budget / "summary.json"
                with summary_path.open("r", encoding="utf-8") as handle:
                    summaries[(dataset, completed_rounds, budget)] = json.load(handle)[
                        "summary_by_sample_size"
                    ]
                metric_files.extend((round_dir / budget).glob("sample_*/seed_*/metrics.json"))

    if len(metric_files) != 240:
        raise RuntimeError(f"Expected 240 metrics files, found {len(metric_files)}")

    aggregate_rows = []
    pair_rows = []
    endpoint_rows = []
    global_rows = []
    for dataset, _ in DATASETS:
        for completed_rounds in ROUNDS:
            global_metrics = globals_by_round[(dataset, completed_rounds)]
            global_rows.append({
                "dataset": dataset,
                "completed_rounds": completed_rounds,
                "mean_logit_cosine": nested_get(
                    global_metrics,
                    "logits.depth_summary.off_diagonal_centered_cosine_mean",
                ),
                "directional_logit_variance": nested_get(
                    global_metrics, "logits.depth_summary.directional_variance_mean"
                ),
                "mean_linear_cka": nested_get(
                    global_metrics, "cka.depth_summary.off_diagonal_linear_cka_mean"
                ),
                **{
                    f"probe_accuracy_{branch}_pct": nested_get(
                        global_metrics, f"logits.sanity.{branch}.accuracy_pct"
                    )
                    for branch in ("b1", "b2", "b3", "final")
                },
            })
            for budget, _ in BUDGETS:
                summary = summaries[(dataset, completed_rounds, budget)]
                for metric_name, key, _ in METRICS:
                    endpoint = paired_endpoint(summary, key)
                    endpoint_rows.append({
                        "dataset": dataset,
                        "completed_rounds": completed_rounds,
                        "budget": budget,
                        "category": "aggregate",
                        "metric": metric_name,
                        "delta_2500_minus_100": endpoint["mean"],
                        "paired_seed_ci95_half_width": endpoint["ci95_half_width"],
                    })
                    for size in SIZES:
                        stat = summary_stat(summary, size, key)
                        if metric_name == "mean_logit_cosine":
                            delta_key = (
                                "delta_from_global.depth_summary."
                                "off_diagonal_centered_cosine_mean"
                            )
                        elif metric_name == "directional_logit_variance":
                            delta_key = (
                                "delta_from_global.depth_summary.directional_variance_mean"
                            )
                        else:
                            delta_key = (
                                "delta_from_global.cka_depth_summary."
                                "off_diagonal_linear_cka_mean"
                            )
                        delta_stat = summary_stat(summary, size, delta_key)
                        aggregate_rows.append({
                            "dataset": dataset,
                            "completed_rounds": completed_rounds,
                            "budget": budget,
                            "sample_size": size,
                            "metric": metric_name,
                            "postlocal_mean": stat["mean"],
                            "seed_macro_ci95_half_width": stat["ci95_half_width"],
                            "delta_from_global_mean": delta_stat["mean"],
                            "delta_from_global_ci95_half_width": delta_stat[
                                "ci95_half_width"
                            ],
                        })
                for representation in ("logit", "cka"):
                    for pair in PAIRS:
                        if representation == "logit":
                            key = (
                                f"postlocal.logits.pairwise.{pair}."
                                "centered_logit_cosine_mean"
                            )
                            delta_key = (
                                f"delta_from_global.pairwise_logits.{pair}."
                                "centered_logit_cosine_mean"
                            )
                        else:
                            key = f"postlocal.cka.pairwise.{pair}.linear_cka"
                            delta_key = f"delta_from_global.pairwise_cka.{pair}.linear_cka"
                        endpoint = paired_endpoint(summary, key)
                        endpoint_rows.append({
                            "dataset": dataset,
                            "completed_rounds": completed_rounds,
                            "budget": budget,
                            "category": f"{representation}_pair",
                            "metric": pair,
                            "delta_2500_minus_100": endpoint["mean"],
                            "paired_seed_ci95_half_width": endpoint[
                                "ci95_half_width"
                            ],
                        })
                        for size in SIZES:
                            stat = summary_stat(summary, size, key)
                            delta_stat = summary_stat(summary, size, delta_key)
                            pair_rows.append({
                                "dataset": dataset,
                                "completed_rounds": completed_rounds,
                                "budget": budget,
                                "sample_size": size,
                                "representation": representation,
                                "pair": pair,
                                "postlocal_mean": stat["mean"],
                                "seed_macro_ci95_half_width": stat[
                                    "ci95_half_width"
                                ],
                                "delta_from_global_mean": delta_stat["mean"],
                                "delta_from_global_ci95_half_width": delta_stat[
                                    "ci95_half_width"
                                ],
                            })

    write_csv(output_dir / "global_checkpoint_metrics.csv", global_rows)
    write_csv(output_dir / "aggregate_postlocal_metrics.csv", aggregate_rows)
    write_csv(output_dir / "branch_pair_postlocal_metrics.csv", pair_rows)
    write_csv(output_dir / "endpoint_deltas_2500_minus_100.csv", endpoint_rows)

    # Global checkpoint evolution: separates representation maturation from local effects.
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    global_metric_specs = (
        ("mean_logit_cosine", "Mean centered-logit cosine"),
        ("directional_logit_variance", "Directional logit variance"),
        ("mean_linear_cka", "Mean linear CKA"),
    )
    for axis, (metric, title) in zip(axes, global_metric_specs):
        for dataset, dataset_title in DATASETS:
            rows = [row for row in global_rows if row["dataset"] == dataset]
            axis.plot(
                [row["completed_rounds"] for row in rows],
                [row[metric] for row in rows],
                marker="o", linewidth=2, label=dataset_title,
            )
        axis.set_xlabel("Completed FL rounds")
        axis.set_ylabel(title)
        axis.set_xticks(ROUNDS)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    save_figure(fig, output_dir, "global_checkpoint_evolution", args.dpi)

    panel_order = (
        ("cifar10", "fixed_step", "CIFAR-10 — fixed step"),
        ("cifar100", "fixed_step", "CIFAR-100 — fixed step"),
        ("cifar10", "fixed_epoch", "CIFAR-10 — fixed epoch"),
        ("cifar100", "fixed_epoch", "CIFAR-100 — fixed epoch"),
    )
    for metric_name, _, y_label in METRICS:
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.3), sharex=True, sharey=True)
        for axis, (dataset, budget, title) in zip(axes.flat, panel_order):
            for completed_rounds in ROUNDS:
                rows = [
                    row for row in aggregate_rows
                    if row["dataset"] == dataset and row["budget"] == budget
                    and row["completed_rounds"] == completed_rounds
                    and row["metric"] == metric_name
                ]
                rows.sort(key=lambda row: row["sample_size"])
                x = np.asarray([row["sample_size"] for row in rows])
                y = np.asarray([row["postlocal_mean"] for row in rows])
                ci = np.asarray([row["seed_macro_ci95_half_width"] for row in rows])
                color = ROUND_COLORS[completed_rounds]
                axis.plot(
                    x, y, marker="o", markersize=3.8, linewidth=1.8,
                    color=color, label=f"round {completed_rounds}",
                )
                axis.fill_between(x, y - ci, y + ci, color=color, alpha=0.10)
            axis.set_xscale("log")
            axis.set_xticks(SIZES, [f"{size:,}" for size in SIZES])
            axis.set_title(title)
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=8)
        for axis in axes[1]:
            axis.set_xlabel("Unique local samples")
        for axis in axes[:, 0]:
            axis.set_ylabel(y_label)
        save_figure(fig, output_dir, metric_name, args.dpi)

    # Endpoint effects resolve the otherwise dense branch-pair curves.
    for category, stem, title in (
        ("logit_pair", "logit_pair_endpoint_delta", "Centered-logit cosine"),
        ("cka_pair", "cka_pair_endpoint_delta", "Linear CKA"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.1), sharex=True, sharey=True)
        for axis, (dataset, budget, panel_title) in zip(axes.flat, panel_order):
            matrix = np.zeros((len(PAIRS), len(ROUNDS)))
            for row_index, pair in enumerate(PAIRS):
                for col_index, completed_rounds in enumerate(ROUNDS):
                    matrix[row_index, col_index] = next(
                        row["delta_2500_minus_100"] for row in endpoint_rows
                        if row["dataset"] == dataset and row["budget"] == budget
                        and row["completed_rounds"] == completed_rounds
                        and row["category"] == category and row["metric"] == pair
                    )
            limit = max(0.01, float(np.abs(matrix).max()))
            image = axis.imshow(
                matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto"
            )
            axis.set_xticks(np.arange(len(ROUNDS)), ROUNDS)
            axis.set_yticks(np.arange(len(PAIRS)), [pair.upper() for pair in PAIRS])
            axis.set_title(panel_title)
            for row_index in range(matrix.shape[0]):
                for col_index in range(matrix.shape[1]):
                    value = matrix[row_index, col_index]
                    axis.text(
                        col_index, row_index, f"{value:+.3f}", ha="center", va="center",
                        color="white" if abs(value) > 0.55 * limit else "black",
                        fontsize=7.8,
                    )
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label(f"Δ {title}")
        for axis in axes[1]:
            axis.set_xlabel("Completed FL rounds")
        fig.suptitle(f"{title}: paired n=2,500 minus n=100", y=1.01)
        save_figure(fig, output_dir, stem, args.dpi)

    report = [
        "# Round-checkpoint logit + CKA result summary",
        "",
        "All post-local entries are seed-macro means over three sampling seeds; each seed mean averages ten local forks. Endpoint error is a paired two-sided Student-t 95% CI across the three seed means.",
        "",
        "## Global checkpoint before controlled local training",
        "",
        "| Dataset | Rounds | Mean logit cosine | Directional variance | Mean CKA | Probe acc. B1/B2/B3/Final (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        acc = "/".join(
            f"{row[f'probe_accuracy_{branch}_pct']:.1f}"
            for branch in ("b1", "b2", "b3", "final")
        )
        report.append(
            f"| {row['dataset'].upper()} | {row['completed_rounds']} | "
            f"{row['mean_logit_cosine']:.4f} | {row['directional_logit_variance']:.4f} | "
            f"{row['mean_linear_cka']:.4f} | {acc} |"
        )

    for metric_name, _, title in METRICS:
        report.extend([
            "", f"## Post-local {title}", "",
            "| Dataset / budget | Round | Global | n=100 | n=250 | n=500 | n=1,000 | n=2,500 | Δ(2,500−100), paired 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for dataset, _ in DATASETS:
            for budget, _ in BUDGETS:
                for completed_rounds in ROUNDS:
                    global_row = next(
                        row for row in global_rows
                        if row["dataset"] == dataset
                        and row["completed_rounds"] == completed_rounds
                    )
                    rows = [
                        row for row in aggregate_rows
                        if row["dataset"] == dataset and row["budget"] == budget
                        and row["completed_rounds"] == completed_rounds
                        and row["metric"] == metric_name
                    ]
                    rows.sort(key=lambda row: row["sample_size"])
                    endpoint = next(
                        row for row in endpoint_rows
                        if row["dataset"] == dataset and row["budget"] == budget
                        and row["completed_rounds"] == completed_rounds
                        and row["category"] == "aggregate"
                        and row["metric"] == metric_name
                    )
                    report.append(
                        f"| {dataset.upper()} / {budget} | {completed_rounds} | "
                        f"{global_row[metric_name]:.4f} | "
                        + " | ".join(f"{row['postlocal_mean']:.4f}" for row in rows)
                        + f" | {endpoint['delta_2500_minus_100']:+.4f} ± "
                        f"{endpoint['paired_seed_ci95_half_width']:.4f} |"
                    )

    for category, title in (
        ("logit_pair", "Branch-pair centered-logit cosine endpoint effects"),
        ("cka_pair", "Branch-pair linear CKA endpoint effects"),
    ):
        report.extend(["", f"## {title}", ""])
        for dataset, _ in DATASETS:
            for budget, _ in BUDGETS:
                report.extend([
                    f"### {dataset.upper()} / {budget}", "",
                    "| Round | " + " | ".join(pair.upper() for pair in PAIRS) + " |",
                    "|---:|" + "---:|" * len(PAIRS),
                ])
                for completed_rounds in ROUNDS:
                    cells = []
                    for pair in PAIRS:
                        row = next(
                            row for row in endpoint_rows
                            if row["dataset"] == dataset and row["budget"] == budget
                            and row["completed_rounds"] == completed_rounds
                            and row["category"] == category and row["metric"] == pair
                        )
                        cells.append(
                            f"{row['delta_2500_minus_100']:+.4f} ± "
                            f"{row['paired_seed_ci95_half_width']:.4f}"
                        )
                    report.append(f"| {completed_rounds} | " + " | ".join(cells) + " |")
                report.append("")

    report.extend([
        "## Interpretation notes",
        "",
        "- Directional variance is not independent of the reported mean cosine: with four centered, unit-normalized branch logit vectors it equals `3/4 × (1 − mean pairwise cosine)`.",
        "- Fixed step isolates unique-sample diversity at a common 100 optimizer steps (5,000 processed examples). Fixed epoch intentionally changes the optimizer-step count with n: 10/25/50/100/250 steps for n=100/250/500/1,000/2,500.",
        "- Round 0 uses checkpoint-specific probes fitted on random-initialization features. Those features are not semantically empty, but this condition is an initialization control rather than a trained-FL endpoint.",
        "- Absolute post-local values mix the round-specific global geometry and the subsequent local update. Use `delta_from_global_*` columns in the CSV files when the question is specifically how much local training changed that checkpoint.",
        "",
    ])
    (output_dir / "result_summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Validated {len(metric_files)} jobs and wrote results to {output_dir}")


if __name__ == "__main__":
    main()
