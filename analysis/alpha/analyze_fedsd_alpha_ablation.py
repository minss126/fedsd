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


METHODS = [
    ("fedsd_fixed_alpha", "Fixed alpha"),
    ("fedsd_beta_alpha", "Beta-aware alpha"),
    ("fedsd_entropy_alpha", "Entropy-aware alpha"),
    ("fedsd_beta_entropy_alpha", "Beta+Entropy alpha"),
    ("fedsd_proxy_teacher_conf", "Teacher confidence"),
    ("fedsd_proxy_teacher_entropy", "Teacher entropy"),
    ("fedsd_proxy_branch_agreement", "Branch agreement"),
    ("fedsd_proxy_teacher_correctness", "Teacher correctness"),
]


def mean_last(values, n):
    if not values:
        return None
    tail = values[-n:]
    return sum(tail) / len(tail)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def effective_beta_alpha(cfg):
    base_alpha = float(cfg.get("byot_alpha", 0.0))
    if not cfg.get("beta_aware_byot_alpha", False):
        return base_alpha
    if cfg.get("partition") == "iid":
        scale = 1.0
    elif cfg.get("partition") == "noniid":
        beta = max(float(cfg.get("beta", 0.0)), 0.0)
        beta_ref = max(float(cfg.get("alpha_beta_ref", 0.5)), 1e-8)
        min_scale = min(max(float(cfg.get("alpha_min_scale", 0.2)), 0.0), 1.0)
        scale = max(min_scale, min(1.0, beta / beta_ref))
    else:
        scale = 1.0
    return base_alpha * scale


def summarize_method(log_dir, method, label):
    json_path = os.path.join(log_dir, f"{method}.json")
    pkl_path = os.path.join(log_dir, f"{method}.pkl")
    log_path = os.path.join(log_dir, f"{method}.log")

    if not os.path.exists(json_path) or not os.path.exists(pkl_path):
        return None, []

    cfg = load_json(json_path)
    data = load_pickle(pkl_path)

    acc = list(data.get("acc_global") or data.get("acc") or [])
    test_loss = list(data.get("test_loss") or [])
    train_loss = list(data.get("avg_train_loss") or [])
    efficiency = list(data.get("efficiency") or [])
    round_time = list(data.get("round_time") or [])
    ece = list(data.get("ece") or [])
    branch_acc = list(data.get("branch_acc") or [])

    rounds = list(range(len(acc)))
    curve_rows = []
    for idx, round_idx in enumerate(rounds):
        branch = branch_acc[idx] if idx < len(branch_acc) else []
        curve_rows.append(
            {
                "method": method,
                "label": label,
                "round": round_idx,
                "acc": acc[idx],
                "test_loss": test_loss[idx] if idx < len(test_loss) else None,
                "train_loss": train_loss[idx] if idx < len(train_loss) else None,
                "efficiency": efficiency[idx] if idx < len(efficiency) else None,
                "round_time": round_time[idx] if idx < len(round_time) else None,
                "ece": ece[idx] if idx < len(ece) else None,
                "branch_1": branch[0] if len(branch) > 0 else None,
                "branch_2": branch[1] if len(branch) > 1 else None,
                "branch_3": branch[2] if len(branch) > 2 else None,
                "teacher": branch[3] if len(branch) > 3 else None,
            }
        )

    best_acc = max(acc) if acc else None
    best_round = acc.index(best_acc) if acc else None
    summary = {
        "method": method,
        "label": label,
        "json_path": json_path,
        "pkl_path": pkl_path,
        "log_path": log_path,
        "rounds": len(acc),
        "dataset": cfg.get("dataset"),
        "partition": cfg.get("partition"),
        "beta": cfg.get("beta"),
        "base_algorithm": "fedprox" if cfg.get("use_fedprox") else cfg.get("alg"),
        "alg": cfg.get("alg"),
        "byot_alpha": cfg.get("byot_alpha"),
        "beta_aware_byot_alpha": cfg.get("beta_aware_byot_alpha", False),
        "adaptive_byot_alpha": cfg.get("adaptive_byot_alpha", False),
        "byot_alpha_proxy": cfg.get("byot_alpha_proxy", "none"),
        "alpha_beta_ref": cfg.get("alpha_beta_ref"),
        "alpha_min_scale": cfg.get("alpha_min_scale"),
        "alpha_entropy_power": cfg.get("alpha_entropy_power"),
        "effective_global_alpha": effective_beta_alpha(cfg),
        "final_acc": acc[-1] if acc else None,
        "best_acc": best_acc,
        "best_round": best_round,
        "last_10_acc": mean_last(acc, 10),
        "last_30_acc": mean_last(acc, 30),
        "final_loss": test_loss[-1] if test_loss else None,
        "last_10_loss": mean_last(test_loss, 10),
        "avg_round_time": mean_last(round_time, len(round_time)) if round_time else None,
        "avg_efficiency": mean_last(efficiency, len(efficiency)) if efficiency else None,
        "final_ece": ece[-1] if ece else None,
        "last_10_ece": mean_last(ece, 10),
    }
    return summary, curve_rows


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy_curves(curves, out_dir):
    df = pd.DataFrame(curves)
    plt.figure(figsize=(10, 5.5))
    for method, label in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        plt.plot(sub["round"], sub["acc"], label=label, linewidth=1.6)
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Fedsd Alpha Ablation - Accuracy")
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
        plt.plot(sub["round"], sub["acc_smooth"], label=label, linewidth=1.8)
    plt.xlabel("Round")
    plt.ylabel("Accuracy, 10-round moving avg (%)")
    plt.title("Fedsd Alpha Ablation - Smoothed Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve_smoothed.png"), dpi=300)
    plt.close()


def plot_summary_bars(summary_rows, out_dir):
    df = pd.DataFrame(summary_rows)
    metrics = [
        ("last_10_acc", "Last-10 Accuracy (%)", "last10_accuracy_bar.png"),
        ("best_acc", "Best Accuracy (%)", "best_accuracy_bar.png"),
        ("final_acc", "Final Accuracy (%)", "final_accuracy_bar.png"),
        ("last_10_loss", "Last-10 Test Loss", "last10_loss_bar.png"),
    ]
    for key, ylabel, filename in metrics:
        if key not in df or df[key].isna().all():
            continue
        plt.figure(figsize=(8, 4.8))
        plt.bar(df["label"], df[key], color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"])
        plt.ylabel(ylabel)
        plt.title(f"Fedsd Alpha Ablation - {ylabel}")
        plt.xticks(rotation=15, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close()


def write_report(path, summaries):
    fixed = next((row for row in summaries if row["method"] == "fedsd_fixed_alpha"), None)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Fedsd Alpha Ablation Report\n\n")
        f.write("## Summary\n\n")
        f.write("| Method | Last-10 Acc | Best Acc | Best Round | Final Acc | Delta Last-10 vs Fixed | Effective Global Alpha |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            delta = ""
            if fixed and row.get("last_10_acc") is not None and fixed.get("last_10_acc") is not None:
                delta = f"{row['last_10_acc'] - fixed['last_10_acc']:.3f}"
            f.write(
                f"| {row['label']} | {row.get('last_10_acc', 0):.3f} | "
                f"{row.get('best_acc', 0):.3f} | {row.get('best_round', 0)} | "
                f"{row.get('final_acc', 0):.3f} | {delta} | "
                f"{row.get('effective_global_alpha', 0):.4f} |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- `Fixed alpha` is the same-code control run.\n")
        f.write("- `Beta-aware alpha` changes the global alpha according to Dirichlet beta.\n")
        f.write("- `Entropy-aware alpha` additionally varies alpha per client by label entropy.\n")
        f.write("- Use `last_10_acc` as the primary stability metric unless you intentionally report best accuracy.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs_tuning/beta_0.3/fedprox")
    parser.add_argument("--out-dir", default="results_analysis/fedsd_alpha_ablation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summaries = []
    curves = []
    missing = []
    for method, label in METHODS:
        summary, curve_rows = summarize_method(args.log_dir, method, label)
        if summary is None:
            missing.append(method)
            continue
        summaries.append(summary)
        curves.extend(curve_rows)

    if not summaries:
        raise SystemExit(f"No completed ablation pkl/json files found in {args.log_dir}")

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

    summary_fields = list(summaries[0].keys())
    write_csv(os.path.join(args.out_dir, "summary.csv"), summaries, summary_fields)
    if curves:
        curve_fields = list(curves[0].keys())
        write_csv(os.path.join(args.out_dir, "curves.csv"), curves, curve_fields)

    plot_accuracy_curves(curves, args.out_dir)
    plot_summary_bars(summaries, args.out_dir)
    write_report(os.path.join(args.out_dir, "report.md"), summaries)

    print(f"Analyzed {len(summaries)} methods from {args.log_dir}")
    if missing:
        print(f"Missing methods: {', '.join(missing)}")
    print(f"Summary: {os.path.join(args.out_dir, 'summary.csv')}")
    print(f"Report: {os.path.join(args.out_dir, 'report.md')}")
    print(f"Plots: {args.out_dir}")


if __name__ == "__main__":
    main()
