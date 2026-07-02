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
    (0.00, "fedsd_alpha0p00", "alpha=0.00"),
    (0.01, "fedsd_alpha0p01", "alpha=0.01"),
    (0.03, "fedsd_alpha0p03", "alpha=0.03"),
    (0.05, "fedsd_fixed_alpha", "alpha=0.05"),
    (0.10, "fedsd_alpha0p10", "alpha=0.10"),
    (0.20, "fedsd_alpha0p20", "alpha=0.20"),
    (0.30, "fedsd_alpha0p30", "alpha=0.30"),
]


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows, out_dir):
    df = pd.DataFrame(rows).sort_values("alpha_value")

    metrics = [
        ("last_10_acc", "Last-10 Accuracy (%)", "last10_accuracy_vs_alpha.png"),
        ("best_acc", "Best Accuracy (%)", "best_accuracy_vs_alpha.png"),
        ("last_10_loss", "Last-10 Test Loss", "last10_loss_vs_alpha.png"),
        ("last_10_ece", "Last-10 ECE", "last10_ece_vs_alpha.png"),
    ]
    for key, ylabel, filename in metrics:
        if key not in df or df[key].isna().all():
            continue
        plt.figure(figsize=(7.5, 4.8))
        plt.plot(df["alpha_value"], df[key], marker="o", linewidth=1.8)
        plt.xlabel("Fixed BYOT alpha")
        plt.ylabel(ylabel)
        plt.title(f"Fedsd Fixed Alpha Sweep - {ylabel}")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close()


def plot_curves(curves, out_dir):
    df = pd.DataFrame(curves)
    if df.empty:
        return

    plt.figure(figsize=(10, 5.5))
    for alpha_value, _, label in RUNS:
        sub = df[df["alpha_value"] == alpha_value].copy()
        if sub.empty:
            continue
        sub["acc_smooth"] = sub["acc"].rolling(window=10, min_periods=1).mean()
        plt.plot(sub["round"], sub["acc_smooth"], label=label, linewidth=1.6)
    plt.xlabel("Round")
    plt.ylabel("Accuracy, 10-round moving avg (%)")
    plt.title("Fedsd Fixed Alpha Sweep - Smoothed Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve_smoothed.png"), dpi=300)
    plt.close()


def write_report(path, rows):
    rows = sorted(rows, key=lambda row: row["alpha_value"])
    best_last10 = max(rows, key=lambda row: row["last_10_acc"])
    best_best = max(rows, key=lambda row: row["best_acc"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Fedsd Fixed Alpha Sweep Report\n\n")
        f.write("## Summary\n\n")
        f.write("| Alpha | Method | Last-10 Acc | Best Acc | Best Round | Final Acc | Last-10 Loss | Last-10 ECE |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['alpha_value']:.2f} | {row['method']} | "
                f"{row['last_10_acc']:.3f} | {row['best_acc']:.3f} | "
                f"{row['best_round']} | {row['final_acc']:.3f} | "
                f"{row['last_10_loss']:.4f} | {row['last_10_ece']:.4f} |\n"
            )

        f.write("\n## Best Points\n\n")
        f.write(f"- Best Last-10 Acc: alpha={best_last10['alpha_value']:.2f}, {best_last10['last_10_acc']:.3f}\n")
        f.write(f"- Best Acc: alpha={best_best['alpha_value']:.2f}, {best_best['best_acc']:.3f}\n")

        f.write("\n## Notes\n\n")
        f.write("- This sweep changes only `--byot_alpha`; beta, temperature, FedProx mu, seed, rounds, and data split settings are fixed.\n")
        f.write("- `alpha=0.05` reuses the existing `fedsd_fixed_alpha` run to avoid overwriting prior results.\n")
        f.write("- If the curve is flat, adaptive alpha is unlikely to help much without changing the proxy or the loss design.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs_tuning/beta_0.3/fedprox")
    parser.add_argument("--out-dir", default="results_analysis/fedsd_fixed_alpha_sweep")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    curves = []
    missing = []
    for alpha_value, method, label in RUNS:
        summary, curve_rows = base.summarize_method(args.log_dir, method, label)
        if summary is None:
            missing.append(method)
            continue
        summary["alpha_value"] = alpha_value
        rows.append(summary)
        for curve_row in curve_rows:
            curve_row["alpha_value"] = alpha_value
            curves.append(curve_row)

    if not rows:
        raise SystemExit(f"No fixed-alpha sweep results found in {args.log_dir}")

    rows = sorted(rows, key=lambda row: row["alpha_value"])
    write_csv(os.path.join(args.out_dir, "summary.csv"), rows, list(rows[0].keys()))
    if curves:
        write_csv(os.path.join(args.out_dir, "curves.csv"), curves, list(curves[0].keys()))

    plot_summary(rows, args.out_dir)
    plot_curves(curves, args.out_dir)
    write_report(os.path.join(args.out_dir, "report.md"), rows)

    print(f"Analyzed {len(rows)} runs")
    if missing:
        print(f"Missing runs: {', '.join(missing)}")
    print(f"Report: {os.path.join(args.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
