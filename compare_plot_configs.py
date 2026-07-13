import argparse
import csv
import json
import os
from collections import defaultdict


PARTITION_MAP = [
    ("IID", "iid", "iid"),
    ("Beta 0.3", "beta_0.3", "beta_0.3"),
    ("Beta 0.5", "beta_0.5", "beta_0.5"),
    ("Grouping", "noniid_grouping", "noniid_grouping"),
]

ALGS = ["fedavg", "fedprox", "moon", "fedrcl", "feddecorr"]
METHODS = ["baseline", "fedsd", "selective", "warmup"]

# These keys should generally match inside a fair comparison group.
# Method-defining keys such as alg/model/kd_conf_threshold are reported
# separately so intentional differences do not hide accidental ones.
FAIRNESS_KEYS = [
    "seed",
    "dataset",
    "datadir",
    "partition",
    "beta",
    "partition_groups",
    "imbalance_factor",
    "n_clients",
    "sample_fraction",
    "round",
    "epochs",
    "batch_size",
    "test_batch_size",
    "optimizer",
    "lr",
    "scheduler",
    "schedule_round",
    "lr_gamma",
    "eta_min",
    "momentum",
    "reg",
    "num_workers",
    "min_require_size",
    "unavailability",
    "in_channels",
    "group_norm",
    "num_groups",
    "last_fc",
    "init",
    "fan",
    "linit",
    "no_init",
    "train_file",
]

METHOD_KEYS = [
    "alg",
    "model",
    "use_fedprox",
    "use_moon",
    "use_fedrcl",
    "feddecorr",
    "mu",
    "temperature",
    "byot_alpha",
    "byot_beta",
    "kd_conf_threshold",
    "kd_min_keep_ratio",
    "min_threshold",
    "warmup_epochs",
    "feddecorr_coef",
    "fedrs_alpha",
    "calibration_temp",
    "use_sd",
    "use_adaptive",
    "use_orthogonal",
    "use_ensemble",
    "use_cosine",
    "use_norm_agg",
    "tau",
    "alpha_t",
    "gamma",
]

IGNORED_KEYS = {
    "device",
    "logdir",
    "log_file_name",
    "time",
    "ckpt_dir",
    "save_best_ckpt",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_existing(*paths):
    for label, path in paths:
        if os.path.exists(path):
            return label, path
    return None, None


def discover_entries():
    entries = []
    missing = []
    for partition_label, tuned_folder, orig_folder in PARTITION_MAP:
        for alg in ALGS:
            for method in METHODS:
                tuned_json = os.path.join("logs_prev/logs_tuning", tuned_folder, alg, f"{method}.json")
                orig_json = os.path.join("logs", orig_folder, alg, f"{method}.json")
                source, path = first_existing(("logs_prev/logs_tuning", tuned_json), ("logs", orig_json))
                if path is None:
                    missing.append(
                        {
                            "partition": partition_label,
                            "algorithm": alg,
                            "method": method,
                            "expected_tuned_json": tuned_json,
                            "expected_orig_json": orig_json,
                        }
                    )
                    continue

                cfg = load_json(path)
                entries.append(
                    {
                        "partition": partition_label,
                        "algorithm": alg,
                        "method": method,
                        "source": source,
                        "path": path,
                        "config": cfg,
                    }
                )
    return entries, missing


def normalize(value):
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value


def diff_group(entries, keys):
    rows = []
    for key in keys:
        values = defaultdict(list)
        for entry in entries:
            value = normalize(entry["config"].get(key, "<MISSING>"))
            values[value].append(f'{entry["algorithm"]}/{entry["method"]}')
        if len(values) > 1:
            rows.append(
                {
                    "key": key,
                    "values": "; ".join(
                        f"{repr(value)}: {', '.join(sorted(labels))}"
                        for value, labels in sorted(values.items(), key=lambda x: repr(x[0]))
                    ),
                }
            )
    return rows


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, entries, missing, fairness_diffs, method_diffs, source_mix):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Plot Config Comparison\n\n")
        f.write("This report follows the same path priority as plot.py: logs_prev/logs_tuning first, logs second.\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Loaded JSON files: {len(entries)}\n")
        f.write(f"- Missing JSON files: {len(missing)}\n")
        f.write(f"- Fairness diff groups: {sum(len(v) for v in fairness_diffs.values())}\n")
        f.write(f"- Method diff groups: {sum(len(v) for v in method_diffs.values())}\n")
        f.write(f"- Groups mixing logs_prev/logs_tuning/logs sources: {len(source_mix)}\n\n")

        if source_mix:
            f.write("## Source Mixing\n\n")
            for group, sources in source_mix.items():
                f.write(f"- {group}: {', '.join(sorted(sources))}\n")
            f.write("\n")

        if missing:
            f.write("## Missing Files\n\n")
            for row in missing:
                f.write(
                    f"- {row['partition']} / {row['algorithm']} / {row['method']}: "
                    f"{row['expected_tuned_json']} or {row['expected_orig_json']}\n"
                )
            f.write("\n")

        f.write("## Fairness Differences\n\n")
        if not fairness_diffs:
            f.write("No differences found in fairness keys.\n\n")
        else:
            for group, rows in fairness_diffs.items():
                f.write(f"### {group}\n\n")
                for row in rows:
                    f.write(f"- `{row['key']}`: {row['values']}\n")
                f.write("\n")

        f.write("## Method Differences\n\n")
        for group, rows in method_diffs.items():
            if not rows:
                continue
            f.write(f"### {group}\n\n")
            for row in rows:
                f.write(f"- `{row['key']}`: {row['values']}\n")
            f.write("\n")


def expected_partition(entry):
    if entry["partition"] == "IID":
        return {"partition": "iid"}
    if entry["partition"] == "Beta 0.3":
        return {"partition": "noniid", "beta": 0.3}
    if entry["partition"] == "Beta 0.5":
        return {"partition": "noniid", "beta": 0.5}
    if entry["partition"] == "Grouping":
        return {"partition": "noniid_grouping"}
    return {}


def expected_method(entry):
    alg = entry["algorithm"]
    method = entry["method"]

    if method == "baseline":
        expected = {"alg": alg, "model": "resnet18"}
    elif method == "fedsd":
        expected = {"alg": "fedbyot", "model": "resnet18_byot", "kd_conf_threshold": 0.0}
    elif method == "selective":
        expected = {"alg": "fedbyot_selective", "model": "resnet18_byot", "kd_conf_threshold": 0.8}
    elif method == "warmup":
        expected = {
            "alg": "fedbyot_selective_greedy",
            "model": "resnet18_byot",
            "kd_conf_threshold": 0.8,
            "warmup_epochs": 2,
        }
    else:
        expected = {}

    expected.update(
        {
            "use_fedprox": alg == "fedprox",
            "use_moon": alg == "moon",
            "use_fedrcl": alg == "fedrcl",
            "feddecorr": alg == "feddecorr",
        }
    )
    if alg == "fedavg":
        expected.update(
            {
                "use_fedprox": False,
                "use_moon": False,
                "use_fedrcl": False,
                "feddecorr": False,
            }
        )
    return expected


def find_consistency_issues(entries):
    issues = []
    for entry in entries:
        expected = {}
        expected.update(expected_partition(entry))
        expected.update(expected_method(entry))

        for key, expected_value in expected.items():
            actual = entry["config"].get(key, "<MISSING>")
            if normalize(actual) != normalize(expected_value):
                issues.append(
                    {
                        "partition": entry["partition"],
                        "algorithm": entry["algorithm"],
                        "method": entry["method"],
                        "source": entry["source"],
                        "path": entry["path"],
                        "key": key,
                        "expected": repr(expected_value),
                        "actual": repr(actual),
                    }
                )
    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results_analysis/config_compare")
    args = parser.parse_args()

    entries, missing = discover_entries()

    loaded_rows = [
        {
            "partition": e["partition"],
            "algorithm": e["algorithm"],
            "method": e["method"],
            "source": e["source"],
            "path": e["path"],
            "dataset": e["config"].get("dataset"),
            "config_partition": e["config"].get("partition"),
            "beta": e["config"].get("beta"),
            "round": e["config"].get("round"),
            "epochs": e["config"].get("epochs"),
            "seed": e["config"].get("seed"),
            "model": e["config"].get("model"),
            "alg_arg": e["config"].get("alg"),
        }
        for e in entries
    ]
    write_csv(
        os.path.join(args.out_dir, "loaded_files.csv"),
        loaded_rows,
        [
            "partition",
            "algorithm",
            "method",
            "source",
            "path",
            "dataset",
            "config_partition",
            "beta",
            "round",
            "epochs",
            "seed",
            "model",
            "alg_arg",
        ],
    )

    write_csv(
        os.path.join(args.out_dir, "missing_files.csv"),
        missing,
        ["partition", "algorithm", "method", "expected_tuned_json", "expected_orig_json"],
    )

    consistency_issues = find_consistency_issues(entries)
    write_csv(
        os.path.join(args.out_dir, "consistency_issues.csv"),
        consistency_issues,
        ["partition", "algorithm", "method", "source", "path", "key", "expected", "actual"],
    )

    fairness_diffs = {}
    method_diffs = {}
    source_mix = {}

    by_partition = defaultdict(list)
    by_partition_alg = defaultdict(list)
    for entry in entries:
        by_partition[entry["partition"]].append(entry)
        by_partition_alg[(entry["partition"], entry["algorithm"])].append(entry)

    # Check objective comparability within each plot panel.
    for partition, group_entries in sorted(by_partition.items()):
        rows = diff_group(group_entries, FAIRNESS_KEYS)
        if rows:
            fairness_diffs[f"Partition panel: {partition}"] = rows

        sources = {entry["source"] for entry in group_entries}
        if len(sources) > 1:
            source_mix[f"Partition panel: {partition}"] = sources

    # Check intended method-level differences inside each algorithm block.
    for (partition, alg), group_entries in sorted(by_partition_alg.items()):
        rows = diff_group(group_entries, METHOD_KEYS)
        if rows:
            method_diffs[f"{partition} / {alg}"] = rows

    flat_fairness_rows = []
    for group, rows in fairness_diffs.items():
        for row in rows:
            flat_fairness_rows.append({"group": group, **row})
    write_csv(
        os.path.join(args.out_dir, "fairness_differences.csv"),
        flat_fairness_rows,
        ["group", "key", "values"],
    )

    flat_method_rows = []
    for group, rows in method_diffs.items():
        for row in rows:
            flat_method_rows.append({"group": group, **row})
    write_csv(
        os.path.join(args.out_dir, "method_differences.csv"),
        flat_method_rows,
        ["group", "key", "values"],
    )

    write_markdown(
        os.path.join(args.out_dir, "report.md"),
        entries,
        missing,
        fairness_diffs,
        method_diffs,
        source_mix,
    )

    if consistency_issues:
        with open(os.path.join(args.out_dir, "report.md"), "a", encoding="utf-8") as f:
            f.write("## Consistency Issues\n\n")
            for row in consistency_issues:
                f.write(
                    f"- {row['partition']} / {row['algorithm']} / {row['method']} "
                    f"(`{row['key']}`): expected {row['expected']}, actual {row['actual']} "
                    f"[{row['source']}]\n"
                )
            f.write("\n")

    print(f"Loaded JSON files: {len(entries)}")
    print(f"Missing JSON files: {len(missing)}")
    print(f"Fairness diff groups: {sum(len(v) for v in fairness_diffs.values())}")
    print(f"Method diff groups: {sum(len(v) for v in method_diffs.values())}")
    print(f"Consistency issues: {len(consistency_issues)}")
    print(f"Report: {os.path.join(args.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
