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


PARTITIONS = [
    ("beta_0.3", "Beta 0.3", "fedsd_fixed_alpha"),
    ("beta_0.5", "Beta 0.5", "fedsd_fixed_alpha_partition_pilot"),
    ("noniid_grouping", "Grouping", "fedsd_fixed_alpha_partition_pilot"),
]

ALPHAS = [
    (0.00, "fedsd_alpha0p00_partition_sweep", "alpha=0.00"),
    (0.01, "fedsd_alpha0p01_partition_sweep", "alpha=0.01"),
    (0.03, "fedsd_alpha0p03_partition_sweep", "alpha=0.03"),
    (0.05, None, "alpha=0.05"),
    (0.10, "fedsd_alpha0p10_partition_sweep", "alpha=0.10"),
    (0.20, "fedsd_alpha0p20_partition_sweep", "alpha=0.20"),
    (0.30, "fedsd_alpha0p30_partition_sweep", "alpha=0.30"),
]


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows, out_dir, key, ylabel, filename):
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    for partition_key, partition_label, _ in PARTITIONS:
        sub = df[df["partition_key"] == partition_key].sort_values("alpha_value")
        if sub.empty:
            continue
        plt.plot(sub["alpha_value"], sub[key], marker="o", linewidth=1.8, label=partition_label)
    plt.xlabel("Fixed BYOT alpha")
    plt.ylabel(ylabel)
    plt.title(f"Fedsd Fixed Alpha Partition Sweep - {ylabel}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()


def plot_curves(curves, out_dir):
    df = pd.DataFrame(curves)
    if df.empty:
        return

    for partition_key, partition_label, _ in PARTITIONS:
        sub_df = df[df["partition_key"] == partition_key]
        if sub_df.empty:
            continue

        plt.figure(figsize=(10, 5.5))
        for alpha_value, _, label in ALPHAS:
            sub = sub_df[sub_df["alpha_value"] == alpha_value].copy()
            if sub.empty:
                continue
            sub["acc_smooth"] = sub["acc"].rolling(window=10, min_periods=1).mean()
            plt.plot(sub["round"], sub["acc_smooth"], label=label, linewidth=1.5)
        plt.xlabel("Round")
        plt.ylabel("Accuracy, 10-round moving avg (%)")
        plt.title(f"{partition_label} - Smoothed Accuracy by Fixed Alpha")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{partition_key}_accuracy_curve_smoothed.png"), dpi=300)
        plt.close()


def make_best_rows(rows):
    best_rows = []
    grouped = {}
    for row in rows:
        grouped.setdefault(row["partition_key"], []).append(row)

    for partition_key, partition_rows in grouped.items():
        best_last10 = max(partition_rows, key=lambda row: row["last_10_acc"])
        best_best = max(partition_rows, key=lambda row: row["best_acc"])
        control = next((row for row in partition_rows if abs(row["alpha_value"] - 0.05) < 1e-12), None)
        best_rows.append(
            {
                "partition_key": partition_key,
                "partition_label": best_last10["partition_label"],
                "best_last10_alpha": best_last10["alpha_value"],
                "best_last10_acc": best_last10["last_10_acc"],
                "control_last10_acc": control["last_10_acc"] if control else None,
                "delta_last10_vs_alpha0p05": best_last10["last_10_acc"] - control["last_10_acc"] if control else None,
                "best_acc_alpha": best_best["alpha_value"],
                "best_acc": best_best["best_acc"],
                "control_best_acc": control["best_acc"] if control else None,
                "delta_best_vs_alpha0p05": best_best["best_acc"] - control["best_acc"] if control else None,
            }
        )
    return best_rows


def write_report(path, rows, best_rows):
    rows = sorted(rows, key=lambda row: (row["partition_key"], row["alpha_value"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Fedsd Fixed Alpha Partition Sweep Report\n\n")
        f.write("## Best by Partition\n\n")
        f.write("| Partition | Best Last-10 Alpha | Best Last-10 Acc | Delta vs alpha=0.05 | Best Acc Alpha | Best Acc | Delta Best vs alpha=0.05 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in best_rows:
            f.write(
                f"| {row['partition_label']} | {row['best_last10_alpha']:.2f} | "
                f"{row['best_last10_acc']:.3f} | {row['delta_last10_vs_alpha0p05']:.3f} | "
                f"{row['best_acc_alpha']:.2f} | {row['best_acc']:.3f} | "
                f"{row['delta_best_vs_alpha0p05']:.3f} |\n"
            )

        f.write("\n## Runs\n\n")
        f.write("| Partition | Alpha | Method | Last-10 Acc | Best Acc | Best Round | Final Acc | Last-10 Loss | Last-10 ECE |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['partition_label']} | {row['alpha_value']:.2f} | {row['method']} | "
                f"{row['last_10_acc']:.3f} | {row['best_acc']:.3f} | "
                f"{row['best_round']} | {row['final_acc']:.3f} | "
                f"{row['last_10_loss']:.4f} | {row['last_10_ece']:.4f} |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- This sweep changes only `--byot_alpha`; FedProx mu, BYOT beta, temperature, seed, rounds, and model are fixed.\n")
        f.write("- `alpha=0.05` reuses existing fixed-alpha controls to avoid overwriting prior runs.\n")
        f.write("- If each partition prefers a different fixed alpha, adaptive alpha has a real motivation. If the curve is flat, the proxy design is probably not the bottleneck.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="logs_tuning")
    parser.add_argument("--base-algo", default="fedprox")
    parser.add_argument("--out-dir", default="results_analysis/fedsd_fixed_alpha_partition_sweep")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    curves = []
    missing = []
    for partition_key, partition_label, control_method in PARTITIONS:
        log_dir = os.path.join(args.root_dir, partition_key, args.base_algo)
        for alpha_value, method, label in ALPHAS:
            method_name = control_method if method is None else method
            summary, curve_rows = base.summarize_method(log_dir, method_name, label)
            if summary is None:
                missing.append(f"{partition_key}/{method_name}")
                continue
            summary["partition_key"] = partition_key
            summary["partition_label"] = partition_label
            summary["alpha_value"] = alpha_value
            rows.append(summary)
            for curve_row in curve_rows:
                curve_row["partition_key"] = partition_key
                curve_row["partition_label"] = partition_label
                curve_row["alpha_value"] = alpha_value
                curves.append(curve_row)

    if not rows:
        raise SystemExit("No fixed-alpha partition sweep results found")

    rows = sorted(rows, key=lambda row: (row["partition_key"], row["alpha_value"]))
    best_rows = make_best_rows(rows)

    write_csv(os.path.join(args.out_dir, "summary.csv"), rows, list(rows[0].keys()))
    write_csv(os.path.join(args.out_dir, "best_by_partition.csv"), best_rows, list(best_rows[0].keys()))
    if curves:
        write_csv(os.path.join(args.out_dir, "curves.csv"), curves, list(curves[0].keys()))

    plot_metric(rows, args.out_dir, "last_10_acc", "Last-10 Accuracy (%)", "last10_accuracy_vs_alpha.png")
    plot_metric(rows, args.out_dir, "best_acc", "Best Accuracy (%)", "best_accuracy_vs_alpha.png")
    plot_metric(rows, args.out_dir, "last_10_loss", "Last-10 Test Loss", "last10_loss_vs_alpha.png")
    plot_metric(rows, args.out_dir, "last_10_ece", "Last-10 ECE", "last10_ece_vs_alpha.png")
    plot_curves(curves, args.out_dir)
    write_report(os.path.join(args.out_dir, "report.md"), rows, best_rows)

    print(f"Analyzed {len(rows)} runs")
    if missing:
        print(f"Missing runs: {', '.join(missing)}")
    print(f"Report: {os.path.join(args.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
