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
    (0.01, "0p01"),
    (0.05, "0p05"),
    (0.10, "0p10"),
    (0.30, "0p30"),
]

DRIFT_KEYS = [
    "client_update_norm",
    "client_update_norm_sq",
    "client_mean_update_norm",
    "client_update_divergence",
    "client_relative_drift",
    "client_update_cosine",
]


def mean_tail(values, n):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values[-n:]) / min(len(values), n)


def mean_initial(values, n):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values[:n]) / min(len(values), n)


def mean_head(values, n):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values[:n]) / min(len(values), n)


def mean_all(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


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
    drift_values = {key: list(data.get(key) or []) for key in DRIFT_KEYS}
    rounds = max([len(acc)] + [len(values) for values in drift_values.values()])

    curve_rows = []
    for idx in range(rounds):
        row = {
            "partition_key": partition_key,
            "partition_label": partition_label,
            "alpha_value": alpha_value,
            "method": method,
            "round": idx,
            "acc": acc[idx] if idx < len(acc) else None,
        }
        for key, values in drift_values.items():
            row[key] = values[idx] if idx < len(values) else None
        curve_rows.append(row)

    divergence = drift_values["client_update_divergence"]
    relative = drift_values["client_relative_drift"]
    cosine = drift_values["client_update_cosine"]
    norm = drift_values["client_update_norm"]
    mean_norm = drift_values["client_mean_update_norm"]

    summary = {
        "partition_key": partition_key,
        "partition_label": partition_label,
        "alpha_value": alpha_value,
        "method": method,
        "json_path": json_path,
        "pkl_path": pkl_path,
        "rounds": len(acc),
        "dataset": cfg.get("dataset"),
        "partition": cfg.get("partition"),
        "beta": cfg.get("beta"),
        "base_algorithm": "fedprox" if cfg.get("use_fedprox") else cfg.get("alg"),
        "alg": cfg.get("alg"),
        "byot_alpha": cfg.get("byot_alpha"),
        "byot_beta": cfg.get("byot_beta"),
        "temperature": cfg.get("temperature"),
        "drift_mean": mean_all(divergence),
        "drift_initial_30": mean_initial(divergence, 30),
        "drift_first_100": mean_head(divergence, 100),
        "drift_first_300": mean_head(divergence, 300),
        "drift_last_30": mean_tail(divergence, 30),
        "relative_drift_mean": mean_all(relative),
        "relative_drift_initial_30": mean_initial(relative, 30),
        "relative_drift_first_300": mean_head(relative, 300),
        "relative_drift_last_30": mean_tail(relative, 30),
        "update_cosine_mean": mean_all(cosine),
        "update_cosine_last_30": mean_tail(cosine, 30),
        "update_norm_mean": mean_all(norm),
        "mean_update_norm_mean": mean_all(mean_norm),
        "acc_mean": mean_all(acc),
        "acc_last_10": mean_tail(acc, 10),
        "acc_best": max(acc) if acc else None,
        "acc_final": acc[-1] if acc else None,
    }
    return summary, curve_rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(summary_rows, out_dir, metric, ylabel, filename):
    df = pd.DataFrame(summary_rows)
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


def plot_curves(curve_rows, out_dir):
    df = pd.DataFrame(curve_rows)
    if df.empty:
        return
    for partition_key, partition_label in PARTITIONS:
        sub_df = df[df["partition_key"] == partition_key]
        if sub_df.empty:
            continue
        for metric, ylabel, filename in [
            ("client_update_divergence", "Client Update Divergence", "drift_curve"),
            ("client_relative_drift", "Relative Drift", "relative_drift_curve"),
        ]:
            plt.figure(figsize=(10, 5.5))
            for alpha_value, _ in ALPHA_TAGS:
                sub = sub_df[sub_df["alpha_value"] == alpha_value].copy()
                if sub.empty:
                    continue
                sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
                sub[f"{metric}_smooth"] = sub[metric].rolling(window=10, min_periods=1).mean()
                plt.plot(sub["round"], sub[f"{metric}_smooth"], label=f"alpha={alpha_value:.2f}", linewidth=1.4)
            plt.xlabel("Round")
            plt.ylabel(ylabel + " (10-round moving avg)")
            plt.title(f"{partition_label} - {ylabel}")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{partition_key}_{filename}.png"), dpi=300)
            plt.close()


def write_report(path, summary_rows, missing):
    rows = sorted(summary_rows, key=lambda row: (row["partition_key"], row["alpha_value"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("# FedSD Drift Sweep Report\n\n")
        f.write("This report measures update-space client drift after local training and before aggregation.\n\n")
        f.write("## Summary\n\n")
        f.write("| Partition | Alpha | Drift Mean | Drift Initial-30 | Drift First-300 | Drift Last-30 | Relative Drift Mean | Relative Initial-30 | Cosine Mean | Acc Last-10 | Acc Best |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['partition_label']} | {row['alpha_value']:.2f} | "
                f"{row['drift_mean']:.6f} | {row['drift_initial_30']:.6f} | "
                f"{row['drift_first_300']:.6f} | {row['drift_last_30']:.6f} | "
                f"{row['relative_drift_mean']:.6f} | {row['relative_drift_initial_30']:.6f} | "
                f"{row['update_cosine_mean']:.6f} | {row['acc_last_10']:.3f} | {row['acc_best']:.3f} |\n"
            )

        if rows:
            df = pd.DataFrame(rows)
            f.write("\n## Monotonicity Checks\n\n")
            f.write("| Partition | Lowest Drift Alpha | Lowest Drift | Highest Drift Alpha | Highest Drift | Drift Ratio |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for _, part in df.groupby("partition_key", sort=False):
                low = part.loc[part["drift_mean"].idxmin()]
                high = part.loc[part["drift_mean"].idxmax()]
                ratio = high["drift_mean"] / low["drift_mean"] if low["drift_mean"] else 0.0
                f.write(
                    f"| {low['partition_label']} | {low['alpha_value']:.2f} | {low['drift_mean']:.6f} | "
                    f"{high['alpha_value']:.2f} | {high['drift_mean']:.6f} | {ratio:.3f} |\n"
                )

        f.write("\n## Notes\n\n")
        f.write("- `client_update_divergence` is `sum_k p_k ||Delta_k - mean(Delta)||^2`.\n")
        f.write("- `client_relative_drift` divides divergence by the squared mean-update norm.\n")
        f.write("- This is an update-space proxy for the theorem's gradient drift term.\n")
        if missing:
            f.write("\n## Missing Runs\n\n")
            for item in missing:
                f.write(f"- {item}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="logs/drift/logs_drift")
    parser.add_argument("--base-algo", default="fedprox")
    parser.add_argument("--out-dir", default="results_analysis/fedsd_drift_sweep")
    parser.add_argument("--method-suffix", default="drift",
                        help="Method suffix after fedsd_alpha<tag>_, e.g. drift or probe_drift.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    curves = []
    missing = []
    for partition_key, partition_label in PARTITIONS:
        for alpha_value, alpha_tag in ALPHA_TAGS:
            method = f"fedsd_alpha{alpha_tag}_{args.method_suffix}"
            summary, curve_rows = load_run(args.root_dir, args.base_algo, partition_key, partition_label, alpha_value, method)
            if summary is None:
                missing.append(f"{partition_key}/{args.base_algo}/{method}")
                continue
            summaries.append(summary)
            curves.extend(curve_rows)

    if not summaries:
        raise SystemExit(f"No drift sweep runs found under {args.root_dir}")

    write_csv(os.path.join(args.out_dir, "summary.csv"), summaries)
    write_csv(os.path.join(args.out_dir, "curves.csv"), curves)
    write_report(os.path.join(args.out_dir, "report.md"), summaries, missing)
    plot_metric(summaries, args.out_dir, "drift_mean", "Client Update Divergence", "drift_mean_by_alpha.png")
    plot_metric(summaries, args.out_dir, "relative_drift_mean", "Relative Drift", "relative_drift_by_alpha.png")
    plot_metric(summaries, args.out_dir, "update_cosine_mean", "Cosine to Mean Update", "cosine_by_alpha.png")
    plot_metric(summaries, args.out_dir, "acc_last_10", "Last-10 Accuracy (%)", "accuracy_by_alpha.png")
    plot_curves(curves, args.out_dir)

    print(f"Analyzed {len(summaries)} runs")
    if missing:
        print(f"Missing runs: {', '.join(missing)}")
    print(f"Report: {os.path.join(args.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
