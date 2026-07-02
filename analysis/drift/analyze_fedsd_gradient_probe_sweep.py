import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
for _analysis_path in (REPO_ROOT / "analysis" / "alpha", REPO_ROOT / "analysis" / "branch", REPO_ROOT / "analysis" / "drift", REPO_ROOT / "analysis" / "general"):
    sys.path.insert(0, str(_analysis_path))

import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


PARTITIONS = [
    ("iid", "IID"),
    ("beta_0.5", "Beta 0.5"),
    ("beta_0.3", "Beta 0.3"),
    ("beta_0.1", "Beta 0.1"),
]

ALPHA_TAGS = [
    (0.00, "0p00"),
    (0.05, "0p05"),
    (0.10, "0p10"),
    (0.30, "0p30"),
]

PROBE_KEYS = [
    "gradient_probe_clients",
    "gradient_ce_divergence",
    "gradient_ce_relative",
    "gradient_ce_norm",
    "gradient_ce_norm_sq",
    "gradient_ce_mean_norm",
    "gradient_ce_cosine",
    "gradient_kd_divergence",
    "gradient_kd_relative",
    "gradient_kd_norm",
    "gradient_kd_norm_sq",
    "gradient_kd_mean_norm",
    "gradient_kd_cosine",
    "gradient_combined_divergence",
    "gradient_combined_relative",
    "gradient_combined_norm",
    "gradient_combined_norm_sq",
    "gradient_combined_mean_norm",
    "gradient_combined_cosine",
    "gradient_ce_kd_cross",
    "gradient_ce_kd_corr",
]


def clean_values(values):
    return [float(v) for v in values if v is not None]


def mean(values):
    values = clean_values(values)
    if not values:
        return None
    return sum(values) / len(values)


def mean_tail(values, n):
    values = clean_values(values)
    if not values:
        return None
    return sum(values[-n:]) / min(len(values), n)


def load_run(root_dir, base_algo, partition_key, partition_label, alpha_value, method):
    prefix = os.path.join(root_dir, partition_key, base_algo, method)
    json_path = prefix + ".json"
    pkl_path = prefix + ".pkl"
    if not os.path.exists(json_path) or not os.path.exists(pkl_path):
        return None, []

    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    acc = list(data.get("acc_global") or data.get("acc") or [])
    probe_values = {key: list(data.get(key) or []) for key in PROBE_KEYS}
    rounds = max([len(acc)] + [len(values) for values in probe_values.values()])

    curves = []
    for idx in range(rounds):
        row = {
            "partition_key": partition_key,
            "partition_label": partition_label,
            "alpha_value": alpha_value,
            "method": method,
            "round": idx,
            "acc": acc[idx] if idx < len(acc) else None,
        }
        for key, values in probe_values.items():
            row[key] = values[idx] if idx < len(values) else None
        if row.get("gradient_combined_divergence") is not None:
            curves.append(row)

    ce = [row["gradient_ce_divergence"] for row in curves]
    kd = [row["gradient_kd_divergence"] for row in curves]
    combined = [row["gradient_combined_divergence"] for row in curves]
    ce_rel = [row["gradient_ce_relative"] for row in curves]
    kd_rel = [row["gradient_kd_relative"] for row in curves]
    combined_rel = [row["gradient_combined_relative"] for row in curves]
    combined_cos = [row["gradient_combined_cosine"] for row in curves]
    ce_kd_cross = [row.get("gradient_ce_kd_cross") for row in curves]
    ce_kd_corr = [row.get("gradient_ce_kd_corr") for row in curves]
    clients = [row["gradient_probe_clients"] for row in curves]

    summary = {
        "partition_key": partition_key,
        "partition_label": partition_label,
        "alpha_value": alpha_value,
        "method": method,
        "json_path": json_path,
        "pkl_path": pkl_path,
        "rounds": len(acc),
        "probe_points": len(curves),
        "dataset": cfg.get("dataset"),
        "partition": cfg.get("partition"),
        "beta": cfg.get("beta"),
        "base_algorithm": "fedprox" if cfg.get("use_fedprox") else cfg.get("alg"),
        "alg": cfg.get("alg"),
        "byot_alpha": cfg.get("byot_alpha"),
        "byot_beta": cfg.get("byot_beta"),
        "temperature": cfg.get("temperature"),
        "probe_clients_mean": mean(clients),
        "ce_divergence_mean": mean(ce),
        "kd_divergence_mean": mean(kd),
        "combined_divergence_mean": mean(combined),
        "ce_relative_mean": mean(ce_rel),
        "kd_relative_mean": mean(kd_rel),
        "combined_relative_mean": mean(combined_rel),
        "combined_cosine_mean": mean(combined_cos),
        "ce_kd_cross_mean": mean(ce_kd_cross),
        "ce_kd_corr_mean": mean(ce_kd_corr),
        "combined_divergence_last_3": mean_tail(combined, 3),
        "combined_relative_last_3": mean_tail(combined_rel, 3),
        "acc_mean": mean(acc),
        "acc_last_10": mean_tail(acc, 10),
        "acc_best": max(acc) if acc else None,
        "acc_final": acc[-1] if acc else None,
    }
    return summary, curves


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows, out_dir, metric, ylabel, filename):
    df = pd.DataFrame(rows)
    if df.empty or metric not in df:
        return
    plt.figure(figsize=(8, 5))
    for partition_key, partition_label in PARTITIONS:
        sub = df[df["partition_key"] == partition_key].sort_values("alpha_value")
        if sub.empty:
            continue
        plt.plot(sub["alpha_value"], sub[metric], marker="o", linewidth=1.8, label=partition_label)
    plt.xlabel("Fixed BYOT alpha")
    plt.ylabel(ylabel)
    plt.title(ylabel + " by Alpha and Partition")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()


def plot_probe_curves(curves, out_dir, alpha_tags):
    df = pd.DataFrame(curves)
    if df.empty:
        return
    for partition_key, partition_label in PARTITIONS:
        sub_df = df[df["partition_key"] == partition_key]
        if sub_df.empty:
            continue
        plt.figure(figsize=(10, 5.5))
        for alpha_value, _ in alpha_tags:
            sub = sub_df[sub_df["alpha_value"] == alpha_value].sort_values("round")
            if sub.empty:
                continue
            plt.plot(sub["round"], sub["gradient_combined_divergence"], marker="o", label=f"alpha={alpha_value:.2f}", linewidth=1.4)
        plt.xlabel("Round")
        plt.ylabel("Combined Gradient Divergence")
        plt.title(f"{partition_label} - Combined Gradient Divergence Probe")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{partition_key}_combined_gradient_curve.png"), dpi=300)
        plt.close()


def write_report(path, rows, missing):
    rows = sorted(rows, key=lambda row: (row["partition_key"], row["alpha_value"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("# FedSD Gradient Probe Sweep Report\n\n")
        f.write("This report probes gradient dissimilarity at the round-start global model, before local training. Combined gradient is CE + alpha * KD.\n\n")
        f.write("## Summary\n\n")
        f.write("| Partition | Alpha | Probe Points | CE Div | KD Div | CE-KD Cross | CE-KD Corr | Combined Div | Combined Rel | Acc Last-10 | Acc Best |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['partition_label']} | {row['alpha_value']:.2f} | {row['probe_points']} | "
                f"{row['ce_divergence_mean']:.6f} | {row['kd_divergence_mean']:.6f} | "
                f"{row.get('ce_kd_cross_mean') or 0.0:.6f} | {row.get('ce_kd_corr_mean') or 0.0:.6f} | "
                f"{row['combined_divergence_mean']:.6f} | {row['combined_relative_mean']:.6f} | "
                f"{row['acc_last_10']:.3f} | {row['acc_best']:.3f} |\n"
            )

        if rows:
            df = pd.DataFrame(rows)
            f.write("\n## Alpha Sensitivity\n\n")
            f.write("| Partition | Lowest Combined Alpha | Lowest Combined Div | Highest Combined Alpha | Highest Combined Div | Ratio |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for _, part in df.groupby("partition_key", sort=False):
                low = part.loc[part["combined_divergence_mean"].idxmin()]
                high = part.loc[part["combined_divergence_mean"].idxmax()]
                ratio = high["combined_divergence_mean"] / low["combined_divergence_mean"] if low["combined_divergence_mean"] else 0.0
                f.write(
                    f"| {low['partition_label']} | {low['alpha_value']:.2f} | {low['combined_divergence_mean']:.6f} | "
                    f"{high['alpha_value']:.2f} | {high['combined_divergence_mean']:.6f} | {ratio:.3f} |\n"
                )

        f.write("\n## Notes\n\n")
        f.write("- CE Div is the weighted dissimilarity of supervised gradients.\n")
        f.write("- KD Div is the weighted dissimilarity of self-distillation gradients.\n")
        f.write("- Combined Div is measured from CE + alpha * KD gradients and is the closest probe to the theorem-level objective.\n")
        if missing:
            f.write("\n## Missing Runs\n\n")
            for item in missing:
                f.write(f"- {item}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="logs_probe")
    parser.add_argument("--base-algo", default="fedprox")
    parser.add_argument("--out-dir", default="results_analysis/fedsd_gradient_probe_sweep")
    parser.add_argument("--method-suffix", default="gprobe",
                        help="Method suffix after fedsd_alpha<tag>_, e.g. gprobe or probe_drift.")
    parser.add_argument("--include-alpha0p01", action="store_true",
                        help="Also analyze alpha=0.01 runs.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    curves = []
    missing = []
    alpha_tags = list(ALPHA_TAGS)
    if args.include_alpha0p01:
        alpha_tags.insert(1, (0.01, "0p01"))
    for partition_key, partition_label in PARTITIONS:
        for alpha_value, alpha_tag in alpha_tags:
            method = f"fedsd_alpha{alpha_tag}_{args.method_suffix}"
            summary, curve_rows = load_run(args.root_dir, args.base_algo, partition_key, partition_label, alpha_value, method)
            if summary is None:
                missing.append(f"{partition_key}/{args.base_algo}/{method}")
                continue
            summaries.append(summary)
            curves.extend(curve_rows)

    if not summaries:
        raise SystemExit(f"No gradient probe runs found under {args.root_dir}")

    write_csv(os.path.join(args.out_dir, "summary.csv"), summaries)
    write_csv(os.path.join(args.out_dir, "curves.csv"), curves)
    write_report(os.path.join(args.out_dir, "report.md"), summaries, missing)
    plot_metric(summaries, args.out_dir, "ce_divergence_mean", "CE Gradient Divergence", "ce_divergence_by_alpha.png")
    plot_metric(summaries, args.out_dir, "kd_divergence_mean", "KD Gradient Divergence", "kd_divergence_by_alpha.png")
    plot_metric(summaries, args.out_dir, "combined_divergence_mean", "Combined Gradient Divergence", "combined_divergence_by_alpha.png")
    plot_metric(summaries, args.out_dir, "combined_relative_mean", "Combined Relative Gradient Drift", "combined_relative_by_alpha.png")
    plot_metric(summaries, args.out_dir, "ce_kd_cross_mean", "CE-KD Drift Cross Term", "ce_kd_cross_by_alpha.png")
    plot_metric(summaries, args.out_dir, "ce_kd_corr_mean", "CE-KD Drift Correlation", "ce_kd_corr_by_alpha.png")
    plot_probe_curves(curves, args.out_dir, alpha_tags)

    print(f"Analyzed {len(summaries)} runs")
    if missing:
        print(f"Missing runs: {', '.join(missing)}")
    print(f"Report: {os.path.join(args.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
