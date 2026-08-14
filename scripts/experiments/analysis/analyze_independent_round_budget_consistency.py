#!/usr/bin/env python3
"""Analyze within-round sample-size consistency across independent FL budgets."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


DATASETS = ("cifar10", "cifar100")
ROUNDS = (0, 10, 50, 100)
TRAINED_ROUNDS = (10, 50, 100)
BUDGETS = ("fixed_step", "fixed_epoch")
SIZES = (100, 250, 500, 1000, 2500)
PAIRS = ("b1-b2", "b1-b3", "b2-b3", "b1-final", "b2-final", "b3-final")
AGGREGATES = (
    (
        "mean_logit_cosine",
        "postlocal.logits.depth_summary.off_diagonal_centered_cosine_mean",
        1,
    ),
    (
        "directional_logit_variance",
        "postlocal.logits.depth_summary.directional_variance_mean",
        -1,
    ),
    (
        "mean_linear_cka",
        "postlocal.cka.depth_summary.off_diagonal_linear_cka_mean",
        1,
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root",
        default="logs/analysis/logs_independent_round_budget_logit_cka",
    )
    parser.add_argument(
        "--output_dir",
        default="logs/analysis/independent_round_budget_logit_cka_analysis",
    )
    return parser.parse_args()


def spearman(x, y):
    ranked_x = rankdata(np.asarray(x, dtype=float))
    ranked_y = rankdata(np.asarray(y, dtype=float))
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def paired_endpoint(summary, key):
    low = summary["100"][key]["seed_means"]
    high = summary["2500"][key]["seed_means"]
    seeds = sorted(set(low) & set(high), key=int)
    values = np.asarray([high[seed] - low[seed] for seed in seeds], dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    critical = 4.303 if len(values) == 3 else 1.96
    return float(values.mean()), float(critical * std / math.sqrt(len(values)))


def monotonic_label(values):
    increasing = all(left <= right for left, right in zip(values, values[1:]))
    decreasing = all(left >= right for left, right in zip(values, values[1:]))
    if increasing:
        return "increasing"
    if decreasing:
        return "decreasing"
    return "non_monotonic"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_nested_subsets(input_root):
    """Ensure every checkpoint sees the same client subsets for a condition."""
    reference = {}
    checked_files = 0
    for dataset in DATASETS:
        for completed_rounds in ROUNDS:
            for budget in BUDGETS:
                for size in SIZES:
                    for seed in (0, 1, 2):
                        path = (
                            input_root / dataset / f"round_{completed_rounds:04d}"
                            / budget / f"sample_{size}" / f"seed_{seed}" / "metrics.json"
                        )
                        with path.open("r", encoding="utf-8") as handle:
                            payload = json.load(handle)
                        if payload.get("metrics") != "logits_cka":
                            raise ValueError(f"Incorrect metrics mode in {path}")
                        if int(payload.get("global_completed_rounds", -1)) != completed_rounds:
                            raise ValueError(f"Checkpoint-round mismatch in {path}")
                        if float(payload["local_training"]["lr"]) != 0.01:
                            raise ValueError(f"Local LR is not fixed at 0.01 in {path}")
                        if any(
                            "subset_indices_sha256" not in client
                            for client in payload["client_results"]
                        ):
                            raise ValueError(
                                f"Exact local-subset hashes are missing in {path}"
                            )
                        signature = tuple(
                            (
                                int(client["client_id"]),
                                int(client["subset_seed"]),
                                str(client.get("subset_indices_sha256", "missing")),
                                int(client["observed_classes"]),
                                int(client["local_class_count_min"]),
                                int(client["local_class_count_max"]),
                                round(float(client["local_class_count_std"]), 12),
                            )
                            for client in payload["client_results"]
                        )
                        key = (dataset, budget, size, seed)
                        if key not in reference:
                            reference[key] = signature
                        elif reference[key] != signature:
                            raise ValueError(
                                f"Local subset metadata changed across rounds for {key}"
                            )
                        checked_files += 1
    return checked_files


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    input_root = (repo_root / args.input_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checked_files = validate_nested_subsets(input_root)
    if checked_files != 240:
        raise RuntimeError(f"Expected 240 metrics files, validated {checked_files}")

    summaries = {}
    for dataset in DATASETS:
        for completed_rounds in ROUNDS:
            for budget in BUDGETS:
                path = (
                    input_root / dataset / f"round_{completed_rounds:04d}"
                    / budget / "summary.json"
                )
                with path.open("r", encoding="utf-8") as handle:
                    summaries[(dataset, completed_rounds, budget)] = json.load(handle)[
                        "summary_by_sample_size"
                    ]

    aggregate_rows = []
    pair_rows = []
    for dataset in DATASETS:
        for completed_rounds in ROUNDS:
            for budget in BUDGETS:
                summary = summaries[(dataset, completed_rounds, budget)]
                for metric, key, desired_sign in AGGREGATES:
                    values = [
                        summary[str(size)][key]["seed_macro"]["mean"] for size in SIZES
                    ]
                    endpoint, endpoint_ci = paired_endpoint(summary, key)
                    rho = spearman(SIZES, values)
                    aggregate_rows.append({
                        "dataset": dataset,
                        "completed_rounds": completed_rounds,
                        "checkpoint_role": (
                            "initialization_control" if completed_rounds == 0
                            else "independent_final_endpoint"
                        ),
                        "local_budget": budget,
                        "metric": metric,
                        **{f"n_{size}": value for size, value in zip(SIZES, values)},
                        "spearman_rho": rho,
                        "monotonicity": monotonic_label(values),
                        "endpoint_delta_2500_minus_100": endpoint,
                        "endpoint_delta_ci95_half_width": endpoint_ci,
                        "desired_direction_sign": desired_sign,
                        "matches_desired_overall_direction": bool(rho * desired_sign > 0),
                        "endpoint_ci_excludes_zero": bool(abs(endpoint) > endpoint_ci),
                    })

                for representation in ("logit", "cka"):
                    for pair in PAIRS:
                        key = (
                            f"postlocal.logits.pairwise.{pair}.centered_logit_cosine_mean"
                            if representation == "logit"
                            else f"postlocal.cka.pairwise.{pair}.linear_cka"
                        )
                        values = [
                            summary[str(size)][key]["seed_macro"]["mean"]
                            for size in SIZES
                        ]
                        endpoint, endpoint_ci = paired_endpoint(summary, key)
                        rho = spearman(SIZES, values)
                        pair_rows.append({
                            "dataset": dataset,
                            "completed_rounds": completed_rounds,
                            "checkpoint_role": (
                                "initialization_control" if completed_rounds == 0
                                else "independent_final_endpoint"
                            ),
                            "local_budget": budget,
                            "representation": representation,
                            "pair": pair,
                            **{f"n_{size}": value for size, value in zip(SIZES, values)},
                            "spearman_rho": rho,
                            "monotonicity": monotonic_label(values),
                            "endpoint_delta_2500_minus_100": endpoint,
                            "endpoint_delta_ci95_half_width": endpoint_ci,
                            "matches_desired_overall_direction": bool(rho > 0),
                            "endpoint_ci_excludes_zero": bool(abs(endpoint) > endpoint_ci),
                        })

    write_csv(output_dir / "within_round_aggregate_consistency.csv", aggregate_rows)
    write_csv(output_dir / "within_round_pair_consistency.csv", pair_rows)

    report = [
        "# Independent round-budget consistency analysis",
        "",
        "This report asks whether the local-sample-size trend reproduces within each independently trained final endpoint. Round 0 is reported only as an initialization control.",
        "",
        "All endpoints use horizon-normalized cosine training (`cosine horizon = total FL rounds`). Local diagnostics use the same LR=0.01 and identical nested subsets across checkpoints.",
        "",
        "## Aggregate within-round trends",
        "",
        "`rho` is Spearman correlation over n={100,250,500,1000,2500}. Desired signs are positive for cosine/CKA and negative for directional variance.",
        "",
        "| Dataset | R | Budget | Metric | Values at n=100/250/500/1000/2500 | rho | Monotonicity | Δ2500−100 ± 95% CI | Desired? |",
        "|---|---:|---|---|---|---:|---|---:|---|",
    ]
    for row in aggregate_rows:
        values = "/".join(f"{row[f'n_{size}']:.4f}" for size in SIZES)
        report.append(
            f"| {row['dataset'].upper()} | {row['completed_rounds']} | "
            f"{row['local_budget']} | {row['metric']} | {values} | "
            f"{row['spearman_rho']:+.2f} | {row['monotonicity']} | "
            f"{row['endpoint_delta_2500_minus_100']:+.4f} ± "
            f"{row['endpoint_delta_ci95_half_width']:.4f} | "
            f"{'yes' if row['matches_desired_overall_direction'] else 'no'} |"
        )

    report.extend([
        "",
        "## Branch-pair direction counts",
        "",
        "Each entry counts pairs with positive Spearman correlation; strict counts require monotonic increase at every adjacent sample size.",
        "",
        "| Dataset | R | Budget | Logit positive / strict | CKA positive / strict |",
        "|---|---:|---|---:|---:|",
    ])
    for dataset in DATASETS:
        for completed_rounds in ROUNDS:
            for budget in BUDGETS:
                cells = []
                for representation in ("logit", "cka"):
                    selected = [
                        row for row in pair_rows
                        if row["dataset"] == dataset
                        and row["completed_rounds"] == completed_rounds
                        and row["local_budget"] == budget
                        and row["representation"] == representation
                    ]
                    positive = sum(row["spearman_rho"] > 0 for row in selected)
                    strict = sum(row["monotonicity"] == "increasing" for row in selected)
                    cells.append(f"{positive}/6 / {strict}/6")
                report.append(
                    f"| {dataset.upper()} | {completed_rounds} | {budget} | "
                    f"{cells[0]} | {cells[1]} |"
                )

    report.extend([
        "",
        "## Robustness decision rule",
        "",
        "- Primary representation evidence: positive mean CKA rho and positive endpoint Δ at independently trained R=10/50/100.",
        "- Strong reproduction: the sign is shared across all trained endpoints and datasets, with branch-final pairs showing the same direction.",
        "- Partial reproduction: only mature endpoints reproduce the sign; report the result as training-stage dependent.",
        "- Logit cosine and CKA answer different questions. Directional variance is algebraically redundant with mean cosine and is not counted as independent confirmation.",
        "",
        f"Validated {checked_files} job files and identical local-subset metadata across all checkpoint conditions.",
        "",
    ])
    (output_dir / "independent_budget_consistency_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        f"Validated {checked_files} jobs and wrote within-round consistency results "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
