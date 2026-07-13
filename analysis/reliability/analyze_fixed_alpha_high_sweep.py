import argparse
import csv
import os
import pickle
import re
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


def finite(values):
    return [float(v) for v in values if v is not None]


def mean(values):
    values = finite(values)
    return sum(values) / len(values) if values else None


def mean_tail(values, n):
    values = finite(values)
    return sum(values[-n:]) / min(n, len(values)) if values else None


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


def parse_alpha(method):
    match = re.search(r"fixed_alpha([0-9]+p[0-9]+)", method)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def load_run(path, root):
    with path.open("rb") as f:
        data = pickle.load(f)
    partition = path.relative_to(root).parts[0]
    method = path.stem
    alpha = parse_alpha(method)
    acc = data.get("acc_global") or data.get("acc") or []
    loss = data.get("test_loss") or []
    ece = data.get("ece") or []
    acc_values = finite(acc)
    best_acc = max(acc_values) if acc_values else None
    best_round = acc_values.index(best_acc) + 1 if best_acc is not None else None
    return {
        "partition_key": partition,
        "partition": PARTITION_LABELS.get(partition, partition),
        "method": method,
        "alpha": alpha,
        "rounds": len(acc_values),
        "last_10_acc": mean_tail(acc, 10),
        "last_30_acc": mean_tail(acc, 30),
        "best_acc": best_acc,
        "best_round": best_round,
        "final_acc": acc_values[-1] if acc_values else None,
        "auc_acc": mean(acc),
        "last_10_loss": mean_tail(loss, 10),
        "last_10_ece": mean_tail(ece, 10),
        "late_std_30": std_tail(acc, 30),
        "round_to_65": first_round_at(acc, 65.0),
        "round_to_67": first_round_at(acc, 67.0),
        "round_to_68": first_round_at(acc, 68.0),
        "round_to_69": first_round_at(acc, 69.0),
    }


def fmt(value, digits=3):
    if value is None or value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def collect_rows(roots, base_algo):
    rows_by_key = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.glob(f"*/{base_algo}/fixed_alpha*.pkl")):
            row = load_run(path, root)
            if row["rounds"] <= 0 or row["alpha"] is None:
                continue
            key = (row["partition_key"], row["alpha"])
            rows_by_key[key] = row
    rows = list(rows_by_key.values())
    order = {"beta_0.1": 0, "beta_0.3": 1, "beta_0.5": 2}
    return sorted(rows, key=lambda r: (order.get(r["partition_key"], 99), r["alpha"]))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_table(f, rows):
    f.write("| Partition | Alpha | Last-10 Acc | Last-30 Acc | Best Acc | Best Round | Final Acc | AUC | R67 | R68 | R69 | Last-10 Loss | Last-10 ECE | Late Std30 |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        f.write(
            f"| {row['partition']} | {row['alpha']:.2f} | {fmt(row['last_10_acc'])} | {fmt(row['last_30_acc'])} | "
            f"{fmt(row['best_acc'])} | {fmt(row['best_round'], 0)} | {fmt(row['final_acc'])} | {fmt(row['auc_acc'])} | "
            f"{fmt(row['round_to_67'], 0)} | {fmt(row['round_to_68'], 0)} | {fmt(row['round_to_69'], 0)} | "
            f"{fmt(row['last_10_loss'], 4)} | {fmt(row['last_10_ece'], 4)} | {fmt(row['late_std_30'])} |\n"
        )


def best_by_partition(rows, metric, reverse=True):
    selected = []
    for partition in sorted({r["partition_key"] for r in rows}):
        part = [r for r in rows if r["partition_key"] == partition and r[metric] is not None]
        if part:
            selected.append(sorted(part, key=lambda r: r[metric], reverse=reverse)[0])
    return selected


def write_report(path, rows, roots):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Fixed Alpha High Sweep Report\n\n")
        f.write("- Sources:\n")
        for root in roots:
            f.write(f"  - `{root}`\n")
        f.write("\n## All Fixed Alpha Runs\n\n")
        write_table(f, rows)
        f.write("\n## Best By Last-10 Accuracy\n\n")
        write_table(f, best_by_partition(rows, "last_10_acc", True))
        f.write("\n## Best By Best Accuracy\n\n")
        write_table(f, best_by_partition(rows, "best_acc", True))
        f.write("\n## Lowest Last-10 ECE\n\n")
        write_table(f, best_by_partition(rows, "last_10_ece", False))
        f.write("\n## Notes\n\n")
        f.write("- Main FL comparison should prioritize Last-10/Last-30/Best accuracy over AUC.\n")
        f.write("- Alpha 1.00 removes the branch CE term in the current BYOT loss, so treat it as a stress test.\n")


def plot_metric(rows, out_dir, metric, ylabel, filename):
    out_dir.mkdir(parents=True, exist_ok=True)
    partitions = ["beta_0.1", "beta_0.3", "beta_0.5"]
    plt.figure(figsize=(8, 5))
    for partition in partitions:
        part = [r for r in rows if r["partition_key"] == partition and r[metric] is not None]
        if not part:
            continue
        plt.plot([r["alpha"] for r in part], [r[metric] for r in part], marker="o", label=PARTITION_LABELS[partition])
    plt.xlabel("Fixed alpha")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["logs/reliability/logs_client_reliability_extended", "logs/reliability/logs_fixed_alpha_high"],
        help="Log roots to merge. Later roots override duplicate partition/alpha pairs.",
    )
    parser.add_argument("--base-algo", default="fedavg")
    parser.add_argument("--out-dir", default="results_analysis/fixed_alpha_high_sweep")
    args = parser.parse_args()

    rows = collect_rows(args.roots, args.base_algo)
    if not rows:
        raise SystemExit("No fixed alpha pkl files found.")

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "summary.csv", rows)
    write_report(out_dir / "report.md", rows, args.roots)
    plot_metric(rows, out_dir, "last_10_acc", "Last-10 Accuracy (%)", "last10_acc_vs_alpha.png")
    plot_metric(rows, out_dir, "best_acc", "Best Accuracy (%)", "best_acc_vs_alpha.png")
    plot_metric(rows, out_dir, "last_10_ece", "Last-10 ECE", "last10_ece_vs_alpha.png")

    print(f"Analyzed {len(rows)} fixed-alpha runs")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
