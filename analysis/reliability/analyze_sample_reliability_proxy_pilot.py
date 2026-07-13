import argparse
import csv
import os
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


METHOD_LABELS = {
    "fedsd_fixed": "Fixed FedSD",
    "fixed_alpha0p00": "Fixed alpha 0.00",
    "fixed_alpha0p01": "Fixed alpha 0.01",
    "fixed_alpha0p05": "Fixed alpha 0.05",
    "fixed_alpha0p10": "Fixed alpha 0.10",
    "fixed_alpha0p30": "Fixed alpha 0.30",
    "sample_teacher_conf": "Teacher confidence",
    "sample_teacher_entropy": "Teacher entropy",
    "sample_teacher_margin": "Teacher margin",
    "sample_teacher_label_prob": "Teacher label prob",
    "sample_teacher_correctness": "Teacher correctness",
    "sample_branch_agreement": "Branch agreement",
    "sample_branch_soft_kl": "Branch soft KL",
    "sample_branch_js": "Branch JS",
    "client_label_prob_0p01_0p30": "Client label prob 0.01-0.30",
    "client_correctness_0p01_0p30": "Client correctness 0.01-0.30",
    "client_branch_js_0p01_0p30": "Client branch JS 0.01-0.30",
    "client_entropy_0p01_0p30": "Client entropy 0.01-0.30",
}


def finite(values):
    return [float(v) for v in values if v is not None]


def mean_tail(values, n):
    values = finite(values)
    if not values:
        return None
    return sum(values[-n:]) / min(n, len(values))


def mean_head(values, n):
    values = finite(values)
    if not values:
        return None
    return sum(values[:n]) / min(n, len(values))


def mean_all(values):
    values = finite(values)
    if not values:
        return None
    return sum(values) / len(values)


def std_tail(values, n):
    values = finite(values)[-n:]
    if not values:
        return None
    mu = sum(values) / len(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


def first_round_at(values, target):
    for idx, value in enumerate(finite(values), start=1):
        if value >= target:
            return idx
    return None


def load_run(path):
    with path.open("rb") as f:
        data = pickle.load(f)
    acc = data.get("acc_global") or data.get("acc") or []
    loss = data.get("test_loss") or []
    ece = data.get("ece") or []
    args = data.get("args", {})
    method = path.stem
    return {
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "seed": args.get("seed", ""),
        "rounds": len(finite(acc)),
        "effective_alpha_mean": mean_all(data.get("byot_effective_alpha_mean") or []),
        "effective_alpha_last_10": mean_tail(data.get("byot_effective_alpha_mean") or [], 10),
        "effective_alpha_min": mean_all(data.get("byot_effective_alpha_min") or []),
        "effective_alpha_max": mean_all(data.get("byot_effective_alpha_max") or []),
        "auc_acc": mean_all(acc),
        "auc_first_100": mean_head(acc, 100),
        "auc_first_300": mean_head(acc, 300),
        "last_10_acc": mean_tail(acc, 10),
        "last_30_acc": mean_tail(acc, 30),
        "late_std_30": std_tail(acc, 30),
        "best_acc": max(finite(acc)) if finite(acc) else None,
        "best_round": finite(acc).index(max(finite(acc))) + 1 if finite(acc) else None,
        "final_acc": finite(acc)[-1] if finite(acc) else None,
        "last_10_loss": mean_tail(loss, 10),
        "last_10_ece": mean_tail(ece, 10),
        "round_to_60": first_round_at(acc, 60.0),
        "round_to_65": first_round_at(acc, 65.0),
        "round_to_67": first_round_at(acc, 67.0),
    }


def fmt(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows, root_dir):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (-(r["last_10_acc"] or -1), -(r["auc_acc"] or -1)))
    fixed = next((r for r in rows if r["method"] == "fedsd_fixed"), None)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Sample-wise Reliability Proxy Pilot\n\n")
        f.write(f"- Source: `{root_dir}`\n")
        f.write("- Setting: beta_0.3 / FedAvg / seed 0 unless overridden in the run script.\n\n")

        f.write("## Summary\n\n")
        f.write("| Rank | Method | Eff Alpha Mean | Eff Alpha Last-10 | Last-10 Acc | Δ Last-10 vs Fixed | Best Acc | Best Round | AUC | AUC First-300 | R65 | R67 | Last-10 Loss | Last-10 ECE | Late Std30 |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for idx, row in enumerate(rows, start=1):
            delta = None
            if fixed and row["last_10_acc"] is not None and fixed["last_10_acc"] is not None:
                delta = row["last_10_acc"] - fixed["last_10_acc"]
            f.write(
                f"| {idx} | {row['label']} | {fmt(row['effective_alpha_mean'], 4)} | {fmt(row['effective_alpha_last_10'], 4)} | "
                f"{fmt(row['last_10_acc'])} | {fmt(delta)} | "
                f"{fmt(row['best_acc'])} | {fmt(row['best_round'], 0)} | {fmt(row['auc_acc'])} | "
                f"{fmt(row['auc_first_300'])} | {fmt(row['round_to_65'], 0)} | {fmt(row['round_to_67'], 0)} | "
                f"{fmt(row['last_10_loss'], 4)} | {fmt(row['last_10_ece'], 4)} | {fmt(row['late_std_30'])} |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- `AUC` is mean accuracy over all communication rounds; larger means better overall convergence.\n")
        f.write("- `R65`/`R67` are the first rounds reaching 65%/67% accuracy. Empty means the target was not reached.\n")
        f.write("- This pilot has one seed, so small differences should be treated as ranking hints rather than final evidence.\n")


def plot_metric(rows, out_dir, metric, ylabel, filename):
    rows = sorted(rows, key=lambda r: r["label"])
    labels = [r["label"] for r in rows]
    values = [r[metric] if r[metric] is not None else 0.0 for r in rows]
    plt.figure(figsize=(11, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="logs/reliability/logs_reliability/beta_0.3/fedavg")
    parser.add_argument("--out-dir", default="results_analysis/sample_reliability_proxy_pilot")
    args = parser.parse_args()

    root = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    rows = [load_run(path) for path in sorted(root.glob("*.pkl"))]
    rows = [row for row in rows if row["rounds"] > 0]
    if not rows:
        raise SystemExit(f"No pkl runs found in {root}")

    write_csv(out_dir / "summary.csv", rows)
    write_report(out_dir / "report.md", rows, root)
    plot_metric(rows, out_dir, "last_10_acc", "Last-10 Accuracy (%)", "last10_acc.png")
    plot_metric(rows, out_dir, "auc_acc", "AUC Mean Accuracy (%)", "auc_acc.png")
    print(f"Analyzed {len(rows)} runs")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
