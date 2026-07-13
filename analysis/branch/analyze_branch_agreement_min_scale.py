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


METHODS = [
    ("fedsd_fixed_alpha", "Fixed alpha"),
    ("fedsd_branch_agree_min0p0", "Branch agree min=0.0"),
    ("fedsd_branch_agree_min0p1", "Branch agree min=0.1"),
    ("fedsd_proxy_branch_agreement", "Branch agree min=0.2"),
    ("fedsd_branch_agree_min0p4", "Branch agree min=0.4"),
]


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(curve_rows, out_dir):
    df = pd.DataFrame(curve_rows)

    plt.figure(figsize=(10, 5.5))
    for method, label in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        plt.plot(sub["round"], sub["acc"], label=label, linewidth=1.4)
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Branch Agreement Min-Scale Sweep - Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    for method, label in METHODS:
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue
        sub["acc_smooth"] = sub["acc"].rolling(window=10, min_periods=1).mean()
        plt.plot(sub["round"], sub["acc_smooth"], label=label, linewidth=1.7)
    plt.xlabel("Round")
    plt.ylabel("Accuracy, 10-round moving avg (%)")
    plt.title("Branch Agreement Min-Scale Sweep - Smoothed Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve_smoothed.png"), dpi=300)
    plt.close()


def plot_bars(summary_rows, out_dir):
    df = pd.DataFrame(summary_rows)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    metrics = [
        ("last_10_acc", "Last-10 Accuracy (%)", "last10_accuracy_bar.png"),
        ("best_acc", "Best Accuracy (%)", "best_accuracy_bar.png"),
        ("final_acc", "Final Accuracy (%)", "final_accuracy_bar.png"),
        ("last_10_loss", "Last-10 Test Loss", "last10_loss_bar.png"),
        ("last_10_ece", "Last-10 ECE", "last10_ece_bar.png"),
    ]

    for key, ylabel, filename in metrics:
        if key not in df or df[key].isna().all():
            continue
        plt.figure(figsize=(8.8, 4.8))
        plt.bar(df["label"], df[key], color=colors[: len(df)])
        plt.ylabel(ylabel)
        plt.title(f"Branch Agreement Min-Scale Sweep - {ylabel}")
        plt.xticks(rotation=15, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close()


def write_report(path, summaries):
    fixed = next((row for row in summaries if row["method"] == "fedsd_fixed_alpha"), None)
    best_last10 = max(summaries, key=lambda row: row.get("last_10_acc") or float("-inf"))
    best_best = max(summaries, key=lambda row: row.get("best_acc") or float("-inf"))

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Branch Agreement Min-Scale Sweep Report\n\n")
        f.write("## Summary\n\n")
        f.write("| Method | alpha_min_scale | Last-10 Acc | Best Acc | Best Round | Final Acc | Delta Last-10 vs Fixed | Last-10 Loss | Last-10 ECE |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            delta = ""
            if fixed and row.get("last_10_acc") is not None and fixed.get("last_10_acc") is not None:
                delta = f"{row['last_10_acc'] - fixed['last_10_acc']:.3f}"
            f.write(
                f"| {row['label']} | {row.get('alpha_min_scale', '')} | "
                f"{row.get('last_10_acc', 0):.3f} | {row.get('best_acc', 0):.3f} | "
                f"{row.get('best_round', 0)} | {row.get('final_acc', 0):.3f} | "
                f"{delta} | {row.get('last_10_loss', 0):.4f} | {row.get('last_10_ece', 0):.4f} |\n"
            )

        f.write("\n## Best Rows\n\n")
        f.write(f"- Best last-10 accuracy: {best_last10['label']} ({best_last10['last_10_acc']:.3f})\n")
        f.write(f"- Best peak accuracy: {best_best['label']} ({best_best['best_acc']:.3f})\n")

        f.write("\n## Notes\n\n")
        f.write("- `Fixed alpha` has no branch-agreement proxy and is the control run.\n")
        f.write("- `Branch agree min=0.2` reuses the existing `fedsd_proxy_branch_agreement` run.\n")
        f.write("- `last_10_acc` is the primary stability metric; `best_acc` is more optimistic.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs_prev/logs_tuning/beta_0.3/fedprox")
    parser.add_argument("--out-dir", default="results_analysis/branch_agreement_min_scale")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summaries = []
    curves = []
    missing = []
    for method, label in METHODS:
        summary, curve_rows = base.summarize_method(args.log_dir, method, label)
        if summary is None:
            missing.append(method)
            continue
        summaries.append(summary)
        curves.extend(curve_rows)

    if not summaries:
        raise SystemExit(f"No completed branch-agreement min-scale runs found in {args.log_dir}")

    fixed = next((row for row in summaries if row["method"] == "fedsd_fixed_alpha"), None)
    for row in summaries:
        if fixed and row.get("last_10_acc") is not None and fixed.get("last_10_acc") is not None:
            row["delta_last_10_acc_vs_fixed"] = row["last_10_acc"] - fixed["last_10_acc"]
        else:
            row["delta_last_10_acc_vs_fixed"] = None
        if fixed and row.get("best_acc") is not None and fixed.get("best_acc") is not None:
            row["delta_best_acc_vs_fixed"] = row["best_acc"] - fixed["best_acc"]
        else:
            row["delta_best_acc_vs_fixed"] = None

    write_csv(os.path.join(args.out_dir, "summary.csv"), summaries, list(summaries[0].keys()))
    if curves:
        write_csv(os.path.join(args.out_dir, "curves.csv"), curves, list(curves[0].keys()))
        plot_curves(curves, args.out_dir)

    plot_bars(summaries, args.out_dir)
    write_report(os.path.join(args.out_dir, "report.md"), summaries)

    print(f"Analyzed {len(summaries)} methods from {args.log_dir}")
    if missing:
        print(f"Missing methods: {', '.join(missing)}")
    print(f"Summary: {os.path.join(args.out_dir, 'summary.csv')}")
    print(f"Report: {os.path.join(args.out_dir, 'report.md')}")
    print(f"Plots: {args.out_dir}")


if __name__ == "__main__":
    main()
