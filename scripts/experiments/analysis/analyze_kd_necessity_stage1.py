#!/usr/bin/env python3
"""Aggregate and plot stage-1 branch-wise KD-necessity diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "branch_teacher_js_normalized",
    "teacher_label_advantage_positive",
    "need_js_x_label_advantage",
    "teacher_true_label_probability",
    "branch_true_label_probability",
    "teacher_entropy_normalized",
    "branch_entropy_normalized",
    "teacher_accuracy",
    "branch_accuracy",
    "teacher_branch_top1_agreement",
    "existing_effective_lambda_mean",
    "existing_teacher_reliability_raw",
)
PRIMARY_METRICS = (
    "branch_teacher_js_normalized",
    "teacher_label_advantage_positive",
    "need_js_x_label_advantage",
)
GROUP_FIELDS = (
    "dataset",
    "partition",
    "beta",
    "stage",
    "communication_round",
    "branch",
)
BRANCH_ORDER = {"B1": 0, "B2": 1, "B3": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_rows(root: Path) -> list[dict]:
    paths = sorted(root.rglob("client_branch_metrics.csv"))
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                parsed = dict(row)
                parsed["seed"] = int(row["seed"])
                parsed["client_id"] = int(row["client_id"])
                parsed["communication_round"] = int(row["communication_round"])
                parsed["samples"] = int(row["samples"])
                for metric in METRICS:
                    parsed[metric] = float(row[metric])
                rows.append(parsed)
    if not rows:
        raise SystemExit(f"No client_branch_metrics.csv files found below {root}")
    return rows


def group_key(row: dict, fields) -> tuple:
    return tuple(row[field] for field in fields)


def mean(values):
    return sum(values) / len(values)


def seed_macros(rows: list[dict]) -> list[dict]:
    fields = GROUP_FIELDS + ("seed",)
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row, fields)].append(row)
    output = []
    for key, client_rows in groups.items():
        record = dict(zip(fields, key))
        record["clients"] = len({row["client_id"] for row in client_rows})
        for metric in METRICS:
            # The client is the analysis unit. Do not let a larger client's
            # sample count silently dominate a partition-level diagnostic.
            record[metric] = mean([row[metric] for row in client_rows])
        output.append(record)
    return sorted(
        output,
        key=lambda row: (
            row["stage"], row["partition"], float(row["beta"]),
            row["communication_round"], row["dataset"],
            BRANCH_ORDER[row["branch"]], row["seed"],
        ),
    )


def condition_summary(seed_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in seed_rows:
        groups[group_key(row, GROUP_FIELDS)].append(row)
    output = []
    for key, members in groups.items():
        base = dict(zip(GROUP_FIELDS, key))
        for metric in METRICS:
            values = [row[metric] for row in members]
            std = statistics.stdev(values) if len(values) >= 2 else float("nan")
            ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) >= 2 else float("nan")
            output.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean(values),
                    "std_across_seeds": std,
                    "ci95_across_seeds": ci95,
                    "seeds": len(values),
                    "clients_per_seed_mean": mean([row["clients"] for row in members]),
                }
            )
    return sorted(
        output,
        key=lambda row: (
            row["metric"], row["stage"], row["partition"],
            float(row["beta"]), row["communication_round"],
            row["dataset"], BRANCH_ORDER[row["branch"]],
        ),
    )


def task_differences(summary: list[dict]) -> list[dict]:
    lookup = {}
    for row in summary:
        key = (
            row["partition"], row["beta"], row["stage"],
            row["communication_round"], row["branch"], row["metric"],
        )
        lookup[(key, row["dataset"])] = row
    output = []
    for (key, dataset), c10 in lookup.items():
        if dataset != "cifar10":
            continue
        c100 = lookup.get((key, "cifar100"))
        if c100 is None:
            continue
        partition, beta, stage, round_number, branch, metric = key
        output.append(
            {
                "partition": partition,
                "beta": beta,
                "stage": stage,
                "communication_round": round_number,
                "branch": branch,
                "metric": metric,
                "cifar10": c10["mean"],
                "cifar100": c100["mean"],
                "cifar100_minus_cifar10": c100["mean"] - c10["mean"],
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["metric"], row["stage"], row["partition"],
            float(row["beta"]), row["communication_round"],
            BRANCH_ORDER[row["branch"]],
        ),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_label(partition: str, beta: str) -> str:
    if partition == "iid":
        return "IID"
    return f"beta={float(beta):g}"


def report_markdown(summary: list[dict], differences: list[dict]) -> str:
    index = {
        (
            row["dataset"], row["partition"], row["beta"], row["stage"],
            row["communication_round"], row["branch"], row["metric"],
        ): row
        for row in summary
    }
    conditions = sorted(
        {
            (row["partition"], row["beta"], row["stage"], row["communication_round"])
            for row in summary
        },
        key=lambda item: (item[2], item[0], float(item[1]), item[3]),
    )
    lines = [
        "# Stage-1 KD-necessity diagnostic", "",
        "All values are client-macro means. If multiple seeds are present, each seed is first "
        "averaged over its selected clients and then averaged across seeds.", "",
        "- `JS`: normalized branch–teacher Jensen–Shannon divergence.",
        "- `Adv`: $[q_T(y)-p_b(y)]_+$, the teacher's positive true-label probability advantage.",
        "- `Need`: `JS × Adv`; it is large only when the predictions differ and the teacher is "
        "more informative about the true label.", "",
    ]
    for partition, beta, stage, round_number in conditions:
        lines.extend(
            [
                f"## {stage} · {condition_label(partition, beta)} · round {round_number}",
                "",
                "| Dataset | Branch | JS | Adv | Need | Teacher $p(y)$ | Branch $p(y)$ |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for dataset in ("cifar10", "cifar100"):
            for branch in ("B1", "B2", "B3"):
                def get(metric):
                    row = index.get(
                        (dataset, partition, beta, stage, round_number, branch, metric)
                    )
                    return float("nan") if row is None else row["mean"]
                lines.append(
                    f"| {dataset.upper()} | {branch} | {get('branch_teacher_js_normalized'):.4f} "
                    f"| {get('teacher_label_advantage_positive'):.4f} "
                    f"| {get('need_js_x_label_advantage'):.4f} "
                    f"| {get('teacher_true_label_probability'):.4f} "
                    f"| {get('branch_true_label_probability'):.4f} |"
                )
        lines.append("")

    primary_differences = [
        row for row in differences if row["metric"] in PRIMARY_METRICS
    ]
    lines.extend(
        [
            "## CIFAR-100 minus CIFAR-10", "",
            "A positive `Need` difference supports the proposed task-dependent KD-need signal; "
            "JS alone is insufficient unless teacher reliability/advantage is also favorable.", "",
            "| Stage | Partition | Round | Branch | Metric | C10 | C100 | Delta |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in primary_differences:
        lines.append(
            f"| {row['stage']} | {condition_label(row['partition'], row['beta'])} "
            f"| {row['communication_round']} | {row['branch']} | {row['metric']} "
            f"| {row['cifar10']:.4f} | {row['cifar100']:.4f} "
            f"| {row['cifar100_minus_cifar10']:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def make_plots(summary: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV and Markdown summaries were still written.")
        return

    lookup_groups = defaultdict(list)
    for row in summary:
        if row["metric"] not in PRIMARY_METRICS:
            continue
        lookup_groups[
            (
                row["metric"], row["dataset"], row["partition"],
                row["beta"], row["stage"],
            )
        ].append(row)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"B1": "#1f77b4", "B2": "#ff7f0e", "B3": "#2ca02c"}
    for key, rows in lookup_groups.items():
        metric, dataset, partition, beta, stage = key
        fig, axis = plt.subplots(figsize=(6.2, 4.2))
        for branch in ("B1", "B2", "B3"):
            branch_rows = sorted(
                [row for row in rows if row["branch"] == branch],
                key=lambda row: row["communication_round"],
            )
            if not branch_rows:
                continue
            axis.plot(
                [row["communication_round"] for row in branch_rows],
                [row["mean"] for row in branch_rows],
                marker="o", label=branch, color=colors[branch],
            )
        axis.set_xlabel("Communication round")
        axis.set_ylabel(metric)
        axis.set_title(
            f"{dataset.upper()} · {condition_label(partition, beta)} · {stage}"
        )
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        tag = f"{metric}_{dataset}_{partition}_b{float(beta):g}_{stage}"
        fig.savefig(plot_dir / f"{tag}.png", dpi=180)
        plt.close(fig)


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(input_root)
    seeds = seed_macros(rows)
    summary = condition_summary(seeds)
    differences = task_differences(summary)
    write_csv(output_dir / "seed_client_macro.csv", seeds)
    write_csv(output_dir / "condition_summary_long.csv", summary)
    write_csv(output_dir / "cifar100_minus_cifar10.csv", differences)
    (output_dir / "report.md").write_text(
        report_markdown(summary, differences), encoding="utf-8"
    )
    make_plots(summary, output_dir)
    print(f"Wrote stage-1 KD-necessity summary to {output_dir}")


if __name__ == "__main__":
    main()
