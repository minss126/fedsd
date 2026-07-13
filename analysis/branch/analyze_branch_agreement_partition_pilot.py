import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
for _analysis_path in (REPO_ROOT / "analysis" / "alpha", REPO_ROOT / "analysis" / "branch", REPO_ROOT / "analysis" / "drift", REPO_ROOT / "analysis" / "general"):
    sys.path.insert(0, str(_analysis_path))


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

import analyze_fedsd_alpha_ablation as base


RUNS = [
    ("beta_0.3", "Beta 0.3", "fedsd_fixed_alpha", "Fixed alpha"),
    ("beta_0.3", "Beta 0.3", "fedsd_proxy_branch_agreement", "Branch agreement"),
    ("beta_0.5", "Beta 0.5", "fedsd_fixed_alpha_partition_pilot", "Fixed alpha"),
    ("beta_0.5", "Beta 0.5", "fedsd_branch_agree_min0p2_partition_pilot", "Branch agreement"),
    ("noniid_grouping", "Grouping", "fedsd_fixed_alpha_partition_pilot", "Fixed alpha"),
    ("noniid_grouping", "Grouping", "fedsd_branch_agree_min0p2_partition_pilot", "Branch agreement"),
]


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_delta_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["partition_key"], {})[row["method_label"]] = row

    deltas = []
    for partition_key, methods in grouped.items():
        fixed = methods.get("Fixed alpha")
        branch = methods.get("Branch agreement")
        if not fixed or not branch:
            continue
        deltas.append(
            {
                "partition_key": partition_key,
                "partition_label": fixed["partition_label"],
                "delta_last_10_acc": branch["last_10_acc"] - fixed["last_10_acc"],
                "delta_best_acc": branch["best_acc"] - fixed["best_acc"],
                "delta_final_acc": branch["final_acc"] - fixed["final_acc"],
                "delta_last_10_loss": branch["last_10_loss"] - fixed["last_10_loss"],
                "delta_last_10_ece": branch["last_10_ece"] - fixed["last_10_ece"],
                "fixed_last_10_acc": fixed["last_10_acc"],
                "branch_last_10_acc": branch["last_10_acc"],
                "fixed_best_acc": fixed["best_acc"],
                "branch_best_acc": branch["best_acc"],
            }
        )
    return deltas


def plot_bars(rows, deltas, out_dir):
    df = pd.DataFrame(rows)
    delta_df = pd.DataFrame(deltas)

    plt.figure(figsize=(8, 4.8))
    pivot = df.pivot(index="partition_label", columns="method_label", values="last_10_acc")
    pivot = pivot.loc[[label for _, label, _, _ in RUNS if label in pivot.index]]
    pivot.plot(kind="bar", ax=plt.gca(), color=["#4c78a8", "#54a24b"])
    plt.ylabel("Last-10 Accuracy (%)")
    plt.xlabel("")
    plt.title("Branch Agreement Partition Pilot - Last-10 Accuracy")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "last10_accuracy_by_partition.png"), dpi=300)
    plt.close()

    if not delta_df.empty:
        plt.figure(figsize=(7, 4.5))
        plt.axhline(0, color="black", linewidth=1)
        plt.bar(delta_df["partition_label"], delta_df["delta_last_10_acc"], color="#f58518")
        plt.ylabel("Branch - Fixed Last-10 Acc (%)")
        plt.title("Branch Agreement Delta by Partition")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "last10_delta_by_partition.png"), dpi=300)
        plt.close()


def write_report(path, rows, deltas):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Branch Agreement Partition Pilot Report\n\n")
        f.write("## Runs\n\n")
        f.write("| Partition | Method | Last-10 Acc | Best Acc | Best Round | Final Acc | Last-10 Loss | Last-10 ECE |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['partition_label']} | {row['method_label']} | "
                f"{row['last_10_acc']:.3f} | {row['best_acc']:.3f} | "
                f"{row['best_round']} | {row['final_acc']:.3f} | "
                f"{row['last_10_loss']:.4f} | {row['last_10_ece']:.4f} |\n"
            )

        f.write("\n## Deltas\n\n")
        f.write("| Partition | Delta Last-10 Acc | Delta Best Acc | Delta Final Acc | Delta Last-10 Loss | Delta Last-10 ECE |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in deltas:
            f.write(
                f"| {row['partition_label']} | {row['delta_last_10_acc']:.3f} | "
                f"{row['delta_best_acc']:.3f} | {row['delta_final_acc']:.3f} | "
                f"{row['delta_last_10_loss']:.4f} | {row['delta_last_10_ece']:.4f} |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- Beta 0.3 reuses the existing seed-0 fixed and branch-agreement runs.\n")
        f.write("- Beta 0.5 and Grouping use the partition pilot runs.\n")
        f.write("- Use this as a trend check; multi-seed is still needed before making a strong claim.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="logs_prev/logs_tuning")
    parser.add_argument("--base-algo", default="fedprox")
    parser.add_argument("--out-dir", default="results_analysis/branch_agreement_partition_pilot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    curves = []
    missing = []
    for partition_key, partition_label, method, method_label in RUNS:
        log_dir = os.path.join(args.root_dir, partition_key, args.base_algo)
        summary, curve_rows = base.summarize_method(log_dir, method, method_label)
        if summary is None:
            missing.append(f"{partition_key}/{method}")
            continue
        summary["partition_key"] = partition_key
        summary["partition_label"] = partition_label
        summary["method_label"] = method_label
        rows.append(summary)
        for curve_row in curve_rows:
            curve_row["partition_key"] = partition_key
            curve_row["partition_label"] = partition_label
            curves.append(curve_row)

    if not rows:
        raise SystemExit("No partition pilot runs found")

    deltas = make_delta_rows(rows)
    write_csv(os.path.join(args.out_dir, "runs.csv"), rows, list(rows[0].keys()))
    if deltas:
        write_csv(os.path.join(args.out_dir, "deltas.csv"), deltas, list(deltas[0].keys()))
    if curves:
        write_csv(os.path.join(args.out_dir, "curves.csv"), curves, list(curves[0].keys()))

    plot_bars(rows, deltas, args.out_dir)
    write_report(os.path.join(args.out_dir, "report.md"), rows, deltas)

    print(f"Analyzed {len(rows)} runs")
    if missing:
        print(f"Missing runs: {', '.join(missing)}")
    print(f"Report: {os.path.join(args.out_dir, 'report.md')}")
    print(f"Deltas: {os.path.join(args.out_dir, 'deltas.csv')}")


if __name__ == "__main__":
    main()
