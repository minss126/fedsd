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


STUDIES = [
    ("fixed_alpha_partition_sweep", "results_analysis/fedsd_fixed_alpha_partition_sweep/curves.csv"),
    ("alpha_adaptation", "results_analysis/fedsd_alpha_ablation/curves.csv"),
    ("proxy_ablation", "results_analysis/fedsd_proxy_ablation/curves.csv"),
    ("branch_min_scale", "results_analysis/branch_agreement_min_scale/curves.csv"),
    ("branch_seed_check", "results_analysis/branch_agreement_seed_check/curves.csv"),
    ("soft_branch_proxy", "results_analysis/soft_branch_proxy_pilot/curves.csv"),
]

TARGETS = [60.0, 65.0, 67.0, 68.0, 69.0, 70.0]
SNAPSHOT_ROUNDS = [100, 200, 300, 400]


def read_study(study_name, path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "acc" not in df:
        return pd.DataFrame()
    df["study"] = study_name
    return df


def normalize_columns(df):
    defaults = {
        "partition_key": "beta_0.3",
        "partition_label": "Beta 0.3",
        "method": "",
        "label": "",
        "alpha_value": "",
        "seed": "",
        "kind": "",
        "method_label": "",
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df["acc"] = pd.to_numeric(df["acc"], errors="coerce")
    return df.dropna(subset=["round", "acc"])


def first_round_at_or_above(group, target):
    hit = group[group["acc"] >= target]
    if hit.empty:
        return None
    return int(hit["round"].min())


def acc_at_round(group, round_idx):
    hit = group[group["round"] == round_idx]
    if hit.empty:
        before = group[group["round"] <= round_idx]
        if before.empty:
            return None
        return float(before.sort_values("round").iloc[-1]["acc"])
    return float(hit.iloc[0]["acc"])


def summarize_group(keys, group):
    group = group.sort_values("round")
    acc = group["acc"].astype(float)
    row = dict(zip(keys[0], keys[1]))
    row.update(
        {
            "rounds": int(group["round"].nunique()),
            "auc_mean_acc": float(acc.mean()),
            "auc_first_100": float(group[group["round"] < 100]["acc"].mean()),
            "auc_first_300": float(group[group["round"] < 300]["acc"].mean()),
            "last_10_acc": float(acc.tail(10).mean()),
            "last_30_acc": float(acc.tail(30).mean()),
            "late_std_10": float(acc.tail(10).std(ddof=0)),
            "late_std_30": float(acc.tail(30).std(ddof=0)),
            "late_range_30": float(acc.tail(30).max() - acc.tail(30).min()),
            "best_acc": float(acc.max()),
            "best_round": int(group.loc[group["acc"].idxmax(), "round"]),
            "final_acc": float(acc.iloc[-1]),
        }
    )
    for round_idx in SNAPSHOT_ROUNDS:
        row[f"acc_r{round_idx}"] = acc_at_round(group, round_idx)
    for target in TARGETS:
        tag = str(target).replace(".", "p")
        row[f"round_to_{tag}"] = first_round_at_or_above(group, target)
    return row


def make_summaries(df):
    group_cols = [
        "study",
        "partition_key",
        "partition_label",
        "method",
        "label",
        "alpha_value",
        "seed",
        "kind",
        "method_label",
    ]
    rows = []
    for values, group in df.groupby(group_cols, dropna=False, sort=False):
        rows.append(summarize_group((group_cols, values), group))
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value, digits=3):
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_fixed_alpha_section(f, rows):
    df = pd.DataFrame(rows)
    sub = df[df["study"] == "fixed_alpha_partition_sweep"].copy()
    if sub.empty:
        return
    sub["alpha_value"] = pd.to_numeric(sub["alpha_value"], errors="coerce")

    f.write("## Fixed Alpha Partition Sweep\n\n")
    f.write("| Partition | Alpha | AUC | AUC First 300 | R65 | R67 | R68 | Last-10 | Late Std30 | Best |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, row in sub.sort_values(["partition_key", "alpha_value"]).iterrows():
        f.write(
            f"| {row['partition_label']} | {row['alpha_value']:.2f} | "
            f"{row['auc_mean_acc']:.3f} | {row['auc_first_300']:.3f} | "
            f"{format_cell(row.get('round_to_65p0'), 0)} | {format_cell(row.get('round_to_67p0'), 0)} | "
            f"{format_cell(row.get('round_to_68p0'), 0)} | {row['last_10_acc']:.3f} | "
            f"{row['late_std_30']:.3f} | {row['best_acc']:.3f} |\n"
        )

    f.write("\n### Best By Convergence Metric\n\n")
    f.write("| Partition | Best AUC Alpha | Best AUC | Best First-300 Alpha | Best First-300 AUC | Lowest Late Std30 Alpha | Late Std30 |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for _, part in sub.groupby("partition_key", sort=False):
        best_auc = part.loc[part["auc_mean_acc"].idxmax()]
        best_early = part.loc[part["auc_first_300"].idxmax()]
        best_stable = part.loc[part["late_std_30"].idxmin()]
        f.write(
            f"| {best_auc['partition_label']} | {best_auc['alpha_value']:.2f} | {best_auc['auc_mean_acc']:.3f} | "
            f"{best_early['alpha_value']:.2f} | {best_early['auc_first_300']:.3f} | "
            f"{best_stable['alpha_value']:.2f} | {best_stable['late_std_30']:.3f} |\n"
        )
    f.write("\n")


def write_study_section(f, rows, study, title):
    df = pd.DataFrame(rows)
    sub = df[df["study"] == study].copy()
    if sub.empty:
        return
    label_col = "method_label" if sub["method_label"].astype(str).str.len().sum() else "label"
    f.write(f"## {title}\n\n")
    f.write("| Method | AUC | AUC First 300 | R65 | R67 | Last-10 | Late Std30 | Best | Final |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, row in sub.iterrows():
        f.write(
            f"| {row[label_col] or row['label']} | {row['auc_mean_acc']:.3f} | "
            f"{row['auc_first_300']:.3f} | {format_cell(row.get('round_to_65p0'), 0)} | "
            f"{format_cell(row.get('round_to_67p0'), 0)} | {row['last_10_acc']:.3f} | "
            f"{row['late_std_30']:.3f} | {row['best_acc']:.3f} | {row['final_acc']:.3f} |\n"
        )
    f.write("\n")


def write_report(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Convergence Metrics Report\n\n")
        f.write("Metrics here are computed from existing `curves.csv` files. `AUC` is the mean accuracy over communication rounds, so larger is better. `R65` means the first round reaching 65% accuracy.\n\n")
        write_fixed_alpha_section(f, rows)
        write_study_section(f, rows, "alpha_adaptation", "Alpha Adaptation")
        write_study_section(f, rows, "proxy_ablation", "Proxy Ablation")
        write_study_section(f, rows, "branch_min_scale", "Branch Agreement Min Scale")
        write_study_section(f, rows, "soft_branch_proxy", "Soft Branch Proxy Pilot")
        write_study_section(f, rows, "branch_seed_check", "Branch Agreement Seed Check")
        f.write("## Notes\n\n")
        f.write("- Use AUC and round-to-target as convergence-speed indicators, not replacement metrics for final accuracy.\n")
        f.write("- Late Std30 measures stability over the last 30 rounds; lower is more stable.\n")
        f.write("- If adaptive methods do not improve AUC, target rounds, or stability, drift-only improvements would be hard to claim as a practical FL gain.\n")


def plot_fixed_alpha(rows, out_dir):
    df = pd.DataFrame(rows)
    sub = df[df["study"] == "fixed_alpha_partition_sweep"].copy()
    if sub.empty:
        return
    sub["alpha_value"] = pd.to_numeric(sub["alpha_value"], errors="coerce")

    for metric, ylabel, filename in [
        ("auc_mean_acc", "AUC Mean Accuracy (%)", "fixed_alpha_auc.png"),
        ("auc_first_300", "First-300 Mean Accuracy (%)", "fixed_alpha_auc_first300.png"),
        ("late_std_30", "Late Std30", "fixed_alpha_late_std30.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for _, part in sub.groupby("partition_key", sort=False):
            part = part.sort_values("alpha_value")
            plt.plot(part["alpha_value"], part[metric], marker="o", linewidth=1.8, label=part["partition_label"].iloc[0])
        plt.xlabel("Fixed BYOT alpha")
        plt.ylabel(ylabel)
        plt.title(ylabel + " by Alpha")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results_analysis/convergence_metrics")
    args = parser.parse_args()

    frames = []
    missing = []
    for study, path in STUDIES:
        df = read_study(study, path)
        if df.empty:
            missing.append(path)
            continue
        frames.append(normalize_columns(df))

    if not frames:
        raise SystemExit("No curve files found")

    all_curves = pd.concat(frames, ignore_index=True)
    rows = make_summaries(all_curves)

    os.makedirs(args.out_dir, exist_ok=True)
    all_curves.to_csv(os.path.join(args.out_dir, "curves_combined.csv"), index=False)
    write_csv(os.path.join(args.out_dir, "summary.csv"), rows)
    write_report(os.path.join(args.out_dir, "report.md"), rows)
    plot_fixed_alpha(rows, args.out_dir)

    print(f"Analyzed {len(rows)} runs from {len(frames)} studies")
    if missing:
        print("Missing curve files:")
        for path in missing:
            print(f"  - {path}")
    print(f"Report: {os.path.join(args.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
