#!/usr/bin/env python3
import csv
import math
import pickle
import re
from pathlib import Path


OUT_DIR = Path("results_analysis/high_alpha_reliability_and_base")


PARTITION_LABELS = {
    "beta_0.1": "Beta 0.1",
    "beta_0.3": "Beta 0.3",
    "beta_0.5": "Beta 0.5",
}


def mean_tail(values, n):
    if not values:
        return math.nan
    tail = values[-min(n, len(values)) :]
    return sum(tail) / len(tail)


def std_tail(values, n):
    if not values:
        return math.nan
    tail = values[-min(n, len(values)) :]
    m = sum(tail) / len(tail)
    return math.sqrt(sum((x - m) ** 2 for x in tail) / len(tail))


def parse_alpha_tag(tag):
    return float(tag.replace("p", "."))


def fixed_row(path, source):
    with path.open("rb") as f:
        data = pickle.load(f)

    partition = path.parts[-3]
    base = path.parts[-2]
    method = path.stem
    match = re.match(r"fixed_alpha(.+)", method)
    alpha = parse_alpha_tag(match.group(1)) if match else math.nan
    return summarize(
        data=data,
        source=source,
        partition=partition,
        base=base,
        method_type="fixed",
        method=method,
        proxy="",
        alpha_min=alpha,
        alpha_max=alpha,
    )


def adaptive_row(path):
    with path.open("rb") as f:
        data = pickle.load(f)

    partition = path.parts[-3]
    base = path.parts[-2]
    method = path.stem
    match = re.match(r"client_(.+)_(0p\d+)_(1p00)", method)
    proxy = ""
    alpha_min = math.nan
    alpha_max = math.nan
    if match:
        proxy = match.group(1)
        alpha_min = parse_alpha_tag(match.group(2))
        alpha_max = parse_alpha_tag(match.group(3))

    return summarize(
        data=data,
        source="logs/reliability/logs_client_reliability_high_alpha",
        partition=partition,
        base=base,
        method_type="adaptive",
        method=method,
        proxy=proxy,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )


def summarize(data, source, partition, base, method_type, method, proxy, alpha_min, alpha_max):
    acc = data.get("acc_global") or data.get("acc") or []
    loss = data.get("test_loss") or []
    ece = data.get("ece") or []
    eff_alpha = data.get("byot_effective_alpha_mean") or []
    eff_alpha_min = data.get("byot_effective_alpha_min") or []
    eff_alpha_max = data.get("byot_effective_alpha_max") or []
    round_time = data.get("round_time") or []

    best_acc = max(acc) if acc else math.nan
    best_round = acc.index(best_acc) + 1 if acc else ""

    return {
        "source": source,
        "partition_key": partition,
        "partition": PARTITION_LABELS.get(partition, partition),
        "base": base,
        "method_type": method_type,
        "method": method,
        "proxy": proxy,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "eff_alpha_mean_all": mean_tail(eff_alpha, len(eff_alpha)) if eff_alpha else math.nan,
        "eff_alpha_last10": mean_tail(eff_alpha, 10),
        "eff_alpha_min_last10": mean_tail(eff_alpha_min, 10),
        "eff_alpha_max_last10": mean_tail(eff_alpha_max, 10),
        "last_10_acc": mean_tail(acc, 10),
        "last_30_acc": mean_tail(acc, 30),
        "best_acc": best_acc,
        "best_round": best_round,
        "final_acc": acc[-1] if acc else math.nan,
        "last_10_loss": mean_tail(loss, 10),
        "last_10_ece": mean_tail(ece, 10),
        "late_std_30": std_tail(acc, 30),
        "avg_round_time": mean_tail(round_time, len(round_time)) if round_time else math.nan,
    }


def fmt(value, digits=3):
    if value == "" or value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def table(rows, columns, digits=None):
    digits = digits or {}
    lines = []
    lines.append("| " + " | ".join(label for label, _ in columns) + " |")
    lines.append("|" + "|".join("---" if label in {"Partition", "Base", "Method", "Proxy", "Type"} else "---:" for label, _ in columns) + "|")
    for row in rows:
        cells = []
        for label, key in columns:
            cells.append(fmt(row.get(key), digits.get(key, 3)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_by(rows, group_keys, metric="last_10_acc"):
    best = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        if key not in best or row[metric] > best[key][metric]:
            best[key] = row
    return [best[key] for key in sorted(best)]


def main():
    rows = []

    for path in sorted(Path("logs/reliability/logs_fixed_alpha_high").glob("beta_*/fedavg/fixed_alpha*.pkl")):
        rows.append(fixed_row(path, "logs/reliability/logs_fixed_alpha_high"))

    for path in sorted(Path("logs/reliability/logs_fixed_alpha_high_base").glob("beta_*/*/fixed_alpha*.pkl")):
        rows.append(fixed_row(path, "logs/reliability/logs_fixed_alpha_high_base"))

    for path in sorted(Path("logs/reliability/logs_client_reliability_high_alpha").glob("beta_*/fedavg/client_*.pkl")):
        rows.append(adaptive_row(path))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.csv"
    if rows:
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    fixed_rows = [r for r in rows if r["method_type"] == "fixed"]
    adaptive_rows = [r for r in rows if r["method_type"] == "adaptive"]
    adaptive_best = best_by(adaptive_rows, ["partition_key"], "last_10_acc")
    fixed_best = best_by(fixed_rows, ["partition_key", "base"], "last_10_acc")

    base_best = best_by(fixed_rows, ["partition_key", "base"], "last_10_acc")
    adaptive_by_range = best_by(adaptive_rows, ["partition_key", "alpha_min"], "last_10_acc")

    cols_main = [
        ("Partition", "partition"),
        ("Base", "base"),
        ("Type", "method_type"),
        ("Method", "method"),
        ("Alpha Min", "alpha_min"),
        ("Alpha Max", "alpha_max"),
        ("EffAlpha L10", "eff_alpha_last10"),
        ("Last-10 Acc", "last_10_acc"),
        ("Best Acc", "best_acc"),
        ("Final Acc", "final_acc"),
        ("Last-10 Loss", "last_10_loss"),
        ("Last-10 ECE", "last_10_ece"),
        ("Late Std30", "late_std_30"),
    ]

    report = []
    report.append("# High-Alpha Reliability and Base Sweep Report\n")
    report.append("## Best Fixed Runs By Partition/Base\n")
    report.append(table(base_best, cols_main, {"last_10_loss": 4, "last_10_ece": 4, "late_std_30": 3, "eff_alpha_last10": 3}))
    report.append("\n## Best Adaptive Runs By Partition\n")
    report.append(table(adaptive_best, cols_main, {"last_10_loss": 4, "last_10_ece": 4, "late_std_30": 3, "eff_alpha_last10": 3}))
    report.append("\n## Best Adaptive Runs By Partition/Range\n")
    report.append(table(adaptive_by_range, cols_main, {"last_10_loss": 4, "last_10_ece": 4, "late_std30": 3, "eff_alpha_last10": 3}))
    report.append("\n## All Adaptive Runs\n")
    report.append(table(sorted(adaptive_rows, key=lambda r: (r["partition_key"], r["alpha_min"], r["method"])), cols_main, {"last_10_loss": 4, "last_10_ece": 4, "late_std_30": 3, "eff_alpha_last10": 3}))
    report.append("\n## All Fixed Runs\n")
    report.append(table(sorted(fixed_rows, key=lambda r: (r["partition_key"], r["base"], r["alpha_max"])), cols_main, {"last_10_loss": 4, "last_10_ece": 4, "late_std_30": 3, "eff_alpha_last10": 3}))

    (OUT_DIR / "report.md").write_text("\n".join(report))
    print(f"Analyzed {len(rows)} runs")
    print(f"Report: {OUT_DIR / 'report.md'}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
