import argparse
import csv
import os
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


PARTITION_LABELS = {
    "beta_0.1": "Beta 0.1",
    "beta_0.3": "Beta 0.3",
    "beta_0.5": "Beta 0.5",
}

METHOD_LABELS = {
    "fixed_alpha0p00": "Fixed alpha 0.00",
    "fixed_alpha0p01": "Fixed alpha 0.01",
    "fixed_alpha0p05": "Fixed alpha 0.05",
    "fixed_alpha0p10": "Fixed alpha 0.10",
    "fixed_alpha0p30": "Fixed alpha 0.30",
    "client_label_prob_0p01_0p30": "Client label prob",
    "client_correctness_0p01_0p30": "Client correctness",
    "client_branch_js_0p01_0p30": "Client branch JS",
    "client_entropy_0p01_0p30": "Client entropy",
}

METHOD_ORDER = {
    "fixed_alpha0p00": 0,
    "fixed_alpha0p01": 1,
    "fixed_alpha0p05": 2,
    "fixed_alpha0p10": 3,
    "fixed_alpha0p30": 4,
    "client_label_prob_0p01_0p30": 10,
    "client_correctness_0p01_0p30": 11,
    "client_branch_js_0p01_0p30": 12,
    "client_entropy_0p01_0p30": 13,
}


def finite(values):
    return [float(v) for v in values if v is not None]


def mean(values):
    values = finite(values)
    return sum(values) / len(values) if values else None


def mean_tail(values, n):
    values = finite(values)
    return sum(values[-n:]) / min(n, len(values)) if values else None


def mean_head(values, n):
    values = finite(values)
    return sum(values[:n]) / min(n, len(values)) if values else None


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


def fmt(value, digits=3):
    if value is None or value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def infer_partition(path, root_dir):
    rel = path.relative_to(root_dir)
    return rel.parts[0] if rel.parts else ""


def load_run(path, root_dir):
    with path.open("rb") as f:
        data = pickle.load(f)

    acc = data.get("acc_global") or data.get("acc") or []
    loss = data.get("test_loss") or []
    ece = data.get("ece") or []
    args = data.get("args", {})
    method = path.stem
    partition = infer_partition(path, root_dir)

    acc_values = finite(acc)
    best_acc = max(acc_values) if acc_values else None
    best_round = acc_values.index(best_acc) + 1 if acc_values and best_acc is not None else None

    return {
        "partition_key": partition,
        "partition": PARTITION_LABELS.get(partition, partition),
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "seed": args.get("seed", ""),
        "rounds": len(acc_values),
        "effective_alpha_mean": mean(data.get("byot_effective_alpha_mean") or []),
        "effective_alpha_last_10": mean_tail(data.get("byot_effective_alpha_mean") or [], 10),
        "effective_alpha_min": mean(data.get("byot_effective_alpha_min") or []),
        "effective_alpha_max": mean(data.get("byot_effective_alpha_max") or []),
        "auc_acc": mean(acc),
        "auc_first_100": mean_head(acc, 100),
        "auc_first_300": mean_head(acc, 300),
        "last_10_acc": mean_tail(acc, 10),
        "last_30_acc": mean_tail(acc, 30),
        "late_std_30": std_tail(acc, 30),
        "best_acc": best_acc,
        "best_round": best_round,
        "final_acc": acc_values[-1] if acc_values else None,
        "last_10_loss": mean_tail(loss, 10),
        "last_10_ece": mean_tail(ece, 10),
        "round_to_60": first_round_at(acc, 60.0),
        "round_to_65": first_round_at(acc, 65.0),
        "round_to_67": first_round_at(acc, 67.0),
    }


def add_baseline_deltas(rows):
    by_partition = {}
    for row in rows:
        by_partition.setdefault(row["partition_key"], {})[row["method"]] = row

    for row in rows:
        baselines = by_partition.get(row["partition_key"], {})
        fixed05 = baselines.get("fixed_alpha0p05")
        fixed30 = baselines.get("fixed_alpha0p30")
        row["delta_last10_vs_fixed05"] = (
            row["last_10_acc"] - fixed05["last_10_acc"]
            if fixed05 and row["last_10_acc"] is not None and fixed05["last_10_acc"] is not None
            else None
        )
        row["delta_last10_vs_fixed30"] = (
            row["last_10_acc"] - fixed30["last_10_acc"]
            if fixed30 and row["last_10_acc"] is not None and fixed30["last_10_acc"] is not None
            else None
        )
        row["delta_auc_vs_fixed05"] = (
            row["auc_acc"] - fixed05["auc_acc"]
            if fixed05 and row["auc_acc"] is not None and fixed05["auc_acc"] is not None
            else None
        )
        row["delta_ece_vs_fixed05"] = (
            row["last_10_ece"] - fixed05["last_10_ece"]
            if fixed05 and row["last_10_ece"] is not None and fixed05["last_10_ece"] is not None
            else None
        )
    return rows


def sort_rows(rows):
    partition_order = {"beta_0.1": 0, "beta_0.3": 1, "beta_0.5": 2}
    return sorted(
        rows,
        key=lambda r: (
            partition_order.get(r["partition_key"], 99),
            METHOD_ORDER.get(r["method"], 99),
        ),
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(f, rows):
    f.write("| Partition | Method | Eff Alpha Mean | Eff Alpha Last-10 | Last-10 Acc | Δ vs Fixed 0.05 | Δ vs Fixed 0.30 | Best Acc | Best Round | AUC | R65 | R67 | Last-10 Loss | Last-10 ECE | Late Std30 |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        f.write(
            f"| {row['partition']} | {row['method_label']} | "
            f"{fmt(row['effective_alpha_mean'], 4)} | {fmt(row['effective_alpha_last_10'], 4)} | "
            f"{fmt(row['last_10_acc'])} | {fmt(row['delta_last10_vs_fixed05'])} | {fmt(row['delta_last10_vs_fixed30'])} | "
            f"{fmt(row['best_acc'])} | {fmt(row['best_round'], 0)} | {fmt(row['auc_acc'])} | "
            f"{fmt(row['round_to_65'], 0)} | {fmt(row['round_to_67'], 0)} | "
            f"{fmt(row['last_10_loss'], 4)} | {fmt(row['last_10_ece'], 4)} | {fmt(row['late_std_30'])} |\n"
        )


def best_rows(rows, metric, reverse=True):
    selected = []
    for partition in sorted({r["partition_key"] for r in rows}):
        part = [r for r in rows if r["partition_key"] == partition and r.get(metric) is not None]
        if not part:
            continue
        selected.append(sorted(part, key=lambda r: r[metric], reverse=reverse)[0])
    return selected


def write_report(path, rows, root_dir):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sort_rows(rows)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Client-wise Reliability Alpha Extended Report\n\n")
        f.write(f"- Source: `{root_dir}`\n")
        f.write("- Client-wise adaptive alpha range: `0.01 ~ 0.30`\n")
        f.write("- Fixed baselines are included for direct comparison.\n\n")

        f.write("## All Runs\n\n")
        write_markdown_table(f, rows)

        f.write("\n## Best By Last-10 Accuracy\n\n")
        write_markdown_table(f, best_rows(rows, "last_10_acc", reverse=True))

        f.write("\n## Best By AUC\n\n")
        write_markdown_table(f, best_rows(rows, "auc_acc", reverse=True))

        f.write("\n## Lowest Last-10 ECE\n\n")
        write_markdown_table(f, best_rows(rows, "last_10_ece", reverse=False))

        f.write("\n## Adaptive Methods Only\n\n")
        adaptive = [r for r in rows if r["method"].startswith("client_")]
        write_markdown_table(f, adaptive)

        f.write("\n## Notes\n\n")
        f.write("- `Δ vs Fixed 0.05` and `Δ vs Fixed 0.30` use the same partition.\n")
        f.write("- `Eff Alpha Mean` is the round-level mean of client-wise effective alpha, averaged over all rounds.\n")
        f.write("- `Eff Alpha Last-10` shows whether the client-wise alpha rises or falls near convergence.\n")
        f.write("- This report is single-seed unless the source logs contain multiple seeds.\n")


def plot_partition_metric(rows, out_dir, metric, ylabel, filename):
    out_dir.mkdir(parents=True, exist_ok=True)
    partitions = ["beta_0.1", "beta_0.3", "beta_0.5"]
    fig, axes = plt.subplots(len(partitions), 1, figsize=(11, 11), sharex=False)
    for ax, partition in zip(axes, partitions):
        part = [r for r in sort_rows(rows) if r["partition_key"] == partition]
        labels = [r["method_label"] for r in part]
        values = [r[metric] if r[metric] is not None else 0.0 for r in part]
        ax.bar(labels, values)
        ax.set_title(PARTITION_LABELS.get(partition, partition))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def plot_adaptive_delta(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    adaptive = [r for r in sort_rows(rows) if r["method"].startswith("client_")]
    labels = [f"{r['partition']} / {r['method_label']}" for r in adaptive]
    values = [r["delta_last10_vs_fixed05"] or 0.0 for r in adaptive]
    plt.figure(figsize=(12, 6))
    colors = ["#2b8cbe" if v >= 0 else "#d95f0e" for v in values]
    plt.bar(labels, values, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Last-10 Acc Delta vs Fixed 0.05")
    plt.tight_layout()
    plt.savefig(out_dir / "adaptive_delta_vs_fixed05.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="logs_client_reliability_extended")
    parser.add_argument("--base-algo", default="fedavg")
    parser.add_argument("--out-dir", default="results_analysis/client_reliability_alpha_extended")
    args = parser.parse_args()

    root = Path(args.root_dir)
    rows = []
    for path in sorted(root.glob(f"*/{args.base_algo}/*.pkl")):
        row = load_run(path, root)
        if row["rounds"] > 0:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No pkl runs found under {root}/*/{args.base_algo}")

    rows = add_baseline_deltas(rows)
    rows = sort_rows(rows)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "summary.csv", rows)
    write_report(out_dir / "report.md", rows, root)
    plot_partition_metric(rows, out_dir, "last_10_acc", "Last-10 Accuracy (%)", "last10_accuracy_by_partition.png")
    plot_partition_metric(rows, out_dir, "effective_alpha_mean", "Effective Alpha Mean", "effective_alpha_by_partition.png")
    plot_adaptive_delta(rows, out_dir)

    print(f"Analyzed {len(rows)} runs")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
