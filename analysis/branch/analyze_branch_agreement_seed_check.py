import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
for _analysis_path in (REPO_ROOT / "analysis" / "alpha", REPO_ROOT / "analysis" / "branch", REPO_ROOT / "analysis" / "drift", REPO_ROOT / "analysis" / "general"):
    sys.path.insert(0, str(_analysis_path))

import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

import analyze_fedsd_alpha_ablation as base


RUNS = [
    ("fixed", 0, "fedsd_fixed_alpha", "Fixed alpha"),
    ("branch_agreement", 0, "fedsd_proxy_branch_agreement", "Branch agreement"),
    ("fixed", 1, "fedsd_fixed_alpha_s1", "Fixed alpha"),
    ("branch_agreement", 1, "fedsd_branch_agree_min0p2_s1", "Branch agreement"),
    ("fixed", 2, "fedsd_fixed_alpha_s2", "Fixed alpha"),
    ("branch_agreement", 2, "fedsd_branch_agree_min0p2_s2", "Branch agreement"),
]


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values) if values else None


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_by_method(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method_group"], []).append(row)

    summary = []
    for method_group, items in grouped.items():
        label = items[0]["label"]
        last10 = [row["last_10_acc"] for row in items]
        best = [row["best_acc"] for row in items]
        final = [row["final_acc"] for row in items]
        loss = [row["last_10_loss"] for row in items]
        ece = [row["last_10_ece"] for row in items]
        summary.append(
            {
                "method_group": method_group,
                "label": label,
                "n": len(items),
                "last_10_acc_mean": mean(last10),
                "last_10_acc_std": stdev(last10),
                "best_acc_mean": mean(best),
                "best_acc_std": stdev(best),
                "final_acc_mean": mean(final),
                "final_acc_std": stdev(final),
                "last_10_loss_mean": mean(loss),
                "last_10_loss_std": stdev(loss),
                "last_10_ece_mean": mean(ece),
                "last_10_ece_std": stdev(ece),
            }
        )
    return summary


def seed_deltas(rows):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["method_group"]] = row

    deltas = []
    for seed, values in sorted(by_seed.items()):
        fixed = values.get("fixed")
        branch = values.get("branch_agreement")
        if not fixed or not branch:
            continue
        deltas.append(
            {
                "seed": seed,
                "delta_last_10_acc": branch["last_10_acc"] - fixed["last_10_acc"],
                "delta_best_acc": branch["best_acc"] - fixed["best_acc"],
                "delta_final_acc": branch["final_acc"] - fixed["final_acc"],
                "delta_last_10_loss": branch["last_10_loss"] - fixed["last_10_loss"],
                "delta_last_10_ece": branch["last_10_ece"] - fixed["last_10_ece"],
            }
        )
    return deltas


def plot_summary(summary_rows, delta_rows, out_dir):
    summary_df = pd.DataFrame(summary_rows)
    delta_df = pd.DataFrame(delta_rows)

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(
        summary_df["label"],
        summary_df["last_10_acc_mean"],
        yerr=summary_df["last_10_acc_std"],
        capsize=6,
        color=["#4c78a8", "#54a24b"],
    )
    plt.ylabel("Last-10 Accuracy (%)")
    plt.title("Branch Agreement Seed Check")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "last10_accuracy_mean_std.png"), dpi=300)
    plt.close()

    if not delta_df.empty:
        plt.figure(figsize=(7, 4.5))
        plt.axhline(0, color="black", linewidth=1)
        plt.bar(delta_df["seed"].astype(str), delta_df["delta_last_10_acc"], color="#f58518")
        plt.xlabel("Seed")
        plt.ylabel("Branch - Fixed Last-10 Acc (%)")
        plt.title("Per-Seed Last-10 Accuracy Delta")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "last10_accuracy_delta_by_seed.png"), dpi=300)
        plt.close()


def write_report(path, summary_rows, delta_rows):
    by_group = {row["method_group"]: row for row in summary_rows}
    fixed = by_group.get("fixed")
    branch = by_group.get("branch_agreement")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Branch Agreement Seed Check Report\n\n")
        f.write("## Method Summary\n\n")
        f.write("| Method | N | Last-10 Acc Mean | Last-10 Acc Std | Best Acc Mean | Final Acc Mean | Last-10 Loss Mean | Last-10 ECE Mean |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['label']} | {row['n']} | {row['last_10_acc_mean']:.3f} | "
                f"{row['last_10_acc_std']:.3f} | {row['best_acc_mean']:.3f} | "
                f"{row['final_acc_mean']:.3f} | {row['last_10_loss_mean']:.4f} | "
                f"{row['last_10_ece_mean']:.4f} |\n"
            )

        f.write("\n## Per-Seed Delta\n\n")
        f.write("| Seed | Delta Last-10 Acc | Delta Best Acc | Delta Final Acc | Delta Last-10 Loss | Delta Last-10 ECE |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for row in delta_rows:
            f.write(
                f"| {row['seed']} | {row['delta_last_10_acc']:.3f} | {row['delta_best_acc']:.3f} | "
                f"{row['delta_final_acc']:.3f} | {row['delta_last_10_loss']:.4f} | "
                f"{row['delta_last_10_ece']:.4f} |\n"
            )

        if fixed and branch:
            f.write("\n## Takeaway\n\n")
            delta_mean = branch["last_10_acc_mean"] - fixed["last_10_acc_mean"]
            f.write(f"- Mean last-10 accuracy delta: {delta_mean:.3f} points.\n")
            positive = sum(1 for row in delta_rows if row["delta_last_10_acc"] > 0)
            f.write(f"- Positive seeds: {positive}/{len(delta_rows)}.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs_tuning/beta_0.3/fedprox")
    parser.add_argument("--out-dir", default="results_analysis/branch_agreement_seed_check")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    run_rows = []
    curves = []
    missing = []
    for method_group, seed, method, label in RUNS:
        summary, curve_rows = base.summarize_method(args.log_dir, method, label)
        if summary is None:
            missing.append(method)
            continue
        summary["method_group"] = method_group
        summary["seed"] = seed
        run_rows.append(summary)
        for curve_row in curve_rows:
            curve_row["method_group"] = method_group
            curve_row["seed"] = seed
            curves.append(curve_row)

    if missing:
        print(f"Missing runs: {', '.join(missing)}")
    if not run_rows:
        raise SystemExit(f"No seed-check runs found in {args.log_dir}")

    method_summary = summarize_by_method(run_rows)
    deltas = seed_deltas(run_rows)

    write_csv(os.path.join(args.out_dir, "runs.csv"), run_rows, list(run_rows[0].keys()))
    write_csv(os.path.join(args.out_dir, "summary.csv"), method_summary, list(method_summary[0].keys()))
    if deltas:
        write_csv(os.path.join(args.out_dir, "seed_deltas.csv"), deltas, list(deltas[0].keys()))
    if curves:
        write_csv(os.path.join(args.out_dir, "curves.csv"), curves, list(curves[0].keys()))

    plot_summary(method_summary, deltas, args.out_dir)
    write_report(os.path.join(args.out_dir, "report.md"), method_summary, deltas)

    print(f"Analyzed {len(run_rows)} runs from {args.log_dir}")
    print(f"Report: {os.path.join(args.out_dir, 'report.md')}")
    print(f"Summary: {os.path.join(args.out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
