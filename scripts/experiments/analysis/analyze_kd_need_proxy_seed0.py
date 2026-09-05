#!/usr/bin/env python3
"""Summarize seed-0 KD-need rules and paired control runs."""

from __future__ import annotations

import argparse
import csv
import pickle
import statistics
from pathlib import Path


METHOD_ORDER = {
    "plain": 0, "current": 1, "constant": 2, "js_client": 3,
    "js": 4, "advantage": 5, "combined": 6,
}
PARTITION_ORDER = {"iid": 0, "beta_0.5": 1, "beta_0.1": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--current-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def partition_label(args: dict) -> str:
    if args.get("partition") == "iid":
        return "iid"
    return f"beta_{float(args.get('beta', 0.0)):g}"


def load_run(path: Path, forced_method: str | None = None) -> tuple[dict, dict]:
    with path.open("rb") as handle:
        result = pickle.load(handle)
    args = dict(result.get("args", {}))
    if forced_method is not None:
        method = forced_method
    elif str(args.get("alg")) in {"fedavg", "fedavgM"} and str(args.get("model")) == "resnet18":
        method = "plain"
    else:
        method = str(args.get("byot_branch_need_proxy", "none"))
        if method == "none":
            method = "current"
    accuracy = list(result.get("acc_global", []))
    if not accuracy:
        raise ValueError(f"Run has no global accuracy trajectory: {path}")
    effective_lambda = list(result.get("byot_effective_alpha_mean", []))
    gate_rounds = list(result.get("byot_branch_need_client_stats", []))
    gate_tail = gate_rounds[-50:]
    gate_values = {branch: [] for branch in ("B1", "B2", "B3")}
    raw_values = {branch: [] for branch in ("B1", "B2", "B3")}
    for round_clients in gate_tail:
        for stats in (round_clients or {}).values():
            for branch in gate_values:
                if f"gate_{branch}" in stats:
                    gate_values[branch].append(float(stats[f"gate_{branch}"]))
                if f"raw_{branch}" in stats:
                    raw_values[branch].append(float(stats[f"raw_{branch}"]))

    record = {
        "dataset": str(args.get("dataset")),
        "partition": partition_label(args),
        "method": method,
        "seed": int(args.get("seed", 0)),
        "rounds": len(accuracy),
        "final_accuracy": float(accuracy[-1]),
        "best_accuracy": float(max(accuracy)),
        "last10_accuracy": float(statistics.mean(accuracy[-10:])),
        "last30_accuracy": float(statistics.mean(accuracy[-30:])),
        "last50_accuracy": float(statistics.mean(accuracy[-50:])),
        "full_accuracy_mean": float(statistics.mean(accuracy)),
        "last50_effective_lambda": (
            float(statistics.mean(effective_lambda[-50:]))
            if effective_lambda and method != "plain" else float("nan")
        ),
        "need_gain": float(args.get("byot_branch_need_gain", 1.0)),
        "path": str(path),
    }
    for branch in ("B1", "B2", "B3"):
        record[f"last50_gate_{branch}"] = (
            float(statistics.mean(gate_values[branch]))
            if gate_values[branch] else float("nan")
        )
        record[f"last50_raw_{branch}"] = (
            float(statistics.mean(raw_values[branch]))
            if raw_values[branch] else float("nan")
        )
    return record, {"accuracy": accuracy}


def collect_runs(experiment_root: Path, current_root: Path):
    records, trajectories = [], {}
    for path in sorted((experiment_root / "runs").rglob("*.pkl")):
        record, trajectory = load_run(path)
        records.append(record)
        trajectories[(record["dataset"], record["partition"], record["method"])] = trajectory
    for path in sorted((current_root / "runs").rglob("*.pkl")):
        record, trajectory = load_run(path, forced_method="current")
        key = (record["dataset"], record["partition"], "current")
        if key not in trajectories:
            records.append(record)
            trajectories[key] = trajectory
    records.sort(
        key=lambda row: (
            row["dataset"], PARTITION_ORDER.get(row["partition"], 99),
            METHOD_ORDER.get(row["method"], 99),
        )
    )
    return records, trajectories


def add_current_deltas(records: list[dict]) -> None:
    current = {
        (row["dataset"], row["partition"]): row
        for row in records if row["method"] == "current"
    }
    for row in records:
        baseline = current.get((row["dataset"], row["partition"]))
        for metric in (
            "final_accuracy", "best_accuracy", "last10_accuracy",
            "last30_accuracy", "last50_accuracy",
        ):
            row[f"delta_current_{metric}"] = (
                row[metric] - baseline[metric] if baseline is not None else float("nan")
            )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report(records: list[dict]) -> str:
    index = {
        (row["dataset"], row["partition"], row["method"]): row
        for row in records
    }
    lines = [
        "# Seed-0 KD-need proxy and control comparison", "",
        "`Plain` is paired FedAvg without BYOT. `Current` is the matching stage-1 "
        "soft-b adaptive trajectory without a need gate. `Constant` multiplies every "
        "current lambda by the same mean-matched factor. `JS-client` averages B1/B2/B3 "
        "JS into one client-wise gate; `JS-branch` retains one gate per branch.", "",
    ]
    for dataset in ("cifar10", "cifar100"):
        lines.extend([f"## {dataset.upper()}", ""])
        for partition in ("iid", "beta_0.5", "beta_0.1"):
            lines.extend(
                [
                    f"### {partition}", "",
                    "| Method | Final | Last 30 | Best | Δ Last 30 vs Current | Last-50 λ |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for method in ("plain", "current", "constant", "js_client", "js", "advantage", "combined"):
                row = index.get((dataset, partition, method))
                if row is None:
                    continue
                lambda_text = (
                    "—" if method == "plain"
                    else f"{row['last50_effective_lambda']:.4f}"
                )
                lines.append(
                    f"| {method} | {row['final_accuracy']:.3f} | {row['last30_accuracy']:.3f} "
                    f"| {row['best_accuracy']:.3f} "
                    f"| {row['delta_current_last30_accuracy']:+.3f} "
                    f"| {lambda_text} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Last-50 branch gates", "",
            "Equal B1/B2/B3 values identify the client-wise/common gate; unequal values show "
            "the extra branch-specific routing used by the other proxies.", "",
            "| Dataset | Partition | Method | B1 | B2 | B3 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in records:
        if row["method"] in {"plain", "current"}:
            continue
        lines.append(
            f"| {row['dataset']} | {row['partition']} | {row['method']} "
            f"| {row['last50_gate_B1']:.4f} | {row['last50_gate_B2']:.4f} "
            f"| {row['last50_gate_B3']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def make_plots(trajectories: dict, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plots")
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "plain": "#111111", "current": "#555555", "constant": "#9467bd",
        "js_client": "#17becf", "js": "#1f77b4",
        "advantage": "#ff7f0e", "combined": "#2ca02c",
    }
    for dataset in ("cifar10", "cifar100"):
        for partition in ("iid", "beta_0.5", "beta_0.1"):
            fig, axis = plt.subplots(figsize=(6.5, 4.3))
            plotted = False
            for method in ("plain", "current", "constant", "js_client", "js", "advantage", "combined"):
                run = trajectories.get((dataset, partition, method))
                if run is None:
                    continue
                accuracy = run["accuracy"]
                axis.plot(
                    range(1, len(accuracy) + 1), accuracy,
                    label=method, color=colors[method], linewidth=1.4,
                )
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            axis.set_xlabel("Communication round")
            axis.set_ylabel("Global test accuracy (%)")
            axis.set_title(f"{dataset.upper()} · {partition}")
            axis.grid(alpha=0.25)
            axis.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"accuracy_{dataset}_{partition}.png", dpi=180)
            plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, trajectories = collect_runs(
        Path(args.experiment_root), Path(args.current_root)
    )
    add_current_deltas(records)
    write_csv(output_dir / "comparison.csv", records)
    (output_dir / "report.md").write_text(make_report(records), encoding="utf-8")
    make_plots(trajectories, output_dir)
    print(f"Wrote KD-need proxy comparison to {output_dir}")


if __name__ == "__main__":
    main()
