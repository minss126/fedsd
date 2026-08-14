#!/usr/bin/env python3
"""Export every absolute local-sample-size value from the round checkpoint analysis."""

import argparse
import csv
from pathlib import Path


ROUNDS = (0, 10, 50, 100)
SIZES = (100, 250, 500, 1000, 2500)
PAIRS = ("b1-b2", "b1-b3", "b2-b3", "b1-final", "b2-final", "b3-final")
CONDITIONS = (
    ("cifar10", "fixed_step", "CIFAR-10 / fixed-step"),
    ("cifar100", "fixed_step", "CIFAR-100 / fixed-step"),
    ("cifar10", "fixed_epoch", "CIFAR-10 / fixed-epoch"),
    ("cifar100", "fixed_epoch", "CIFAR-100 / fixed-epoch"),
)
AGGREGATES = (
    ("mean_logit_cosine", "Mean centered-logit cosine"),
    ("directional_logit_variance", "Directional logit variance"),
    ("mean_linear_cka", "Mean linear CKA"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis_dir",
        default="logs/analysis/round_checkpoint_logit_cka_analysis",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values_for(rows, criteria):
    selected = [
        row for row in rows
        if all(str(row[key]) == str(value) for key, value in criteria.items())
    ]
    selected.sort(key=lambda row: int(row["sample_size"]))
    if [int(row["sample_size"]) for row in selected] != list(SIZES):
        raise ValueError(f"Incomplete sample sizes for {criteria}")
    values = [float(row["postlocal_mean"]) for row in selected]
    globals_ = [
        float(row["postlocal_mean"]) - float(row["delta_from_global_mean"])
        for row in selected
    ]
    if max(globals_) - min(globals_) > 1e-6:
        raise ValueError(f"Global baseline changed across sample sizes: {criteria}")
    return globals_[0], values


def append_table_header(report, first_column):
    report.append(
        f"| {first_column} | Global | n=100 | n=250 | n=500 | n=1,000 | n=2,500 |"
    )
    report.append("|---|---:|---:|---:|---:|---:|---:|")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    analysis_dir = (repo_root / args.analysis_dir).resolve()
    output = (
        Path(args.output).resolve() if args.output
        else analysis_dir / "all_absolute_values_by_round.md"
    )
    aggregate_rows = read_csv(analysis_dir / "aggregate_postlocal_metrics.csv")
    pair_rows = read_csv(analysis_dir / "branch_pair_postlocal_metrics.csv")

    report = [
        "# All absolute values by checkpoint round",
        "",
        "These are seed-macro post-local means from the original single 100-round trajectory experiment. Values are absolute metrics, not n=2,500 minus n=100 deltas.",
        "",
        "Sample-size order is always 100, 250, 500, 1,000, 2,500. Round 0 is an initialization control; rounds 10 and 50 are intermediate checkpoints under a cosine horizon of 100.",
        "",
    ]
    for completed_rounds in ROUNDS:
        report.extend([f"## Round {completed_rounds}", ""])
        for metric, title in AGGREGATES:
            report.extend([f"### {title}", ""])
            append_table_header(report, "Condition")
            for dataset, budget, label in CONDITIONS:
                global_value, values = values_for(aggregate_rows, {
                    "dataset": dataset,
                    "completed_rounds": completed_rounds,
                    "budget": budget,
                    "metric": metric,
                })
                report.append(
                    f"| {label} | {global_value:.4f} | "
                    + " | ".join(f"{value:.4f}" for value in values)
                    + " |"
                )
            report.append("")

        for representation, title in (
            ("logit", "Branch-pair centered-logit cosine"),
            ("cka", "Branch-pair linear CKA"),
        ):
            report.extend([f"### {title}", ""])
            append_table_header(report, "Condition / pair")
            for dataset, budget, label in CONDITIONS:
                for pair in PAIRS:
                    global_value, values = values_for(pair_rows, {
                        "dataset": dataset,
                        "completed_rounds": completed_rounds,
                        "budget": budget,
                        "representation": representation,
                        "pair": pair,
                    })
                    report.append(
                        f"| {label} / {pair.upper()} | {global_value:.4f} | "
                        + " | ".join(f"{value:.4f}" for value in values)
                        + " |"
                    )
                report.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote all absolute values to {output}")


if __name__ == "__main__":
    main()
