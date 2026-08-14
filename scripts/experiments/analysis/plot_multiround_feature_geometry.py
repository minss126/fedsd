#!/usr/bin/env python3
"""Plot multi-round within-client feature and frozen-probe logit geometry."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dxfl-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEPTHS = ("b1", "b2", "b3", "final")
DEPTH_PAIRS = (
    "b1-b2",
    "b1-b3",
    "b2-b3",
    "b1-final",
    "b2-final",
    "b3-final",
)
METRICS = (
    ("within_class_variance", "Within-class variance"),
    ("between_class_variance", "Between-class variance"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--title_prefix", default="CIFAR-100 IID, teacher-only FedAvg",
    )
    return parser.parse_args()


def load_runs(input_root):
    root = Path(input_root)
    paths = sorted(set(
        root.rglob("*_postlocal_feature_geometry.json")
    ) | set(
        root.rglob("*_postlocal_internal_geometry.json")
    ))
    if not paths:
        raise FileNotFoundError(
            f"No post-local feature/internal geometry JSON files found under {input_root}"
        )
    runs = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("records", [])
        if not records:
            continue
        config = payload.get("args", {})
        n_clients = int(config["n_clients"])
        sizes = payload.get("client_train_sizes", [])
        samples_per_client = float(np.mean(sizes)) if sizes else (
            float(payload.get("global_train_samples", 50000)) / n_clients
        )
        runs.append({
            "path": str(path),
            "seed": int(config.get("seed", 0)),
            "n_clients": n_clients,
            "samples_per_client": samples_per_client,
            "records": records,
        })
    if not runs:
        raise ValueError("Geometry JSON files were found, but none contained records.")
    return runs


def condition_label(samples_per_client, n_clients):
    rounded = int(round(samples_per_client))
    return f"{rounded:,} samples/client (K={n_clients})"


def main():
    args = parse_args()
    runs = load_runs(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # condition -> round -> seed-run macro values
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run in runs:
        condition = (run["samples_per_client"], run["n_clients"])
        for record in run["records"]:
            round_number = int(record["round"]) + 1
            macro = record["postlocal_client_macro"]
            for depth in DEPTHS:
                for metric, _ in METRICS:
                    values[(depth, metric)][condition][round_number].append(
                        float(macro[depth][f"{metric}_mean"])
                    )

    csv_rows = []
    colors = plt.get_cmap("viridis")(
        np.linspace(0.08, 0.92, max(2, len({(r['samples_per_client'], r['n_clients']) for r in runs})))
    )
    for depth in DEPTHS:
        for metric, metric_title in METRICS:
            fig, axis = plt.subplots(figsize=(7.4, 4.8))
            conditions = sorted(values[(depth, metric)], reverse=True)
            for color, condition in zip(colors, conditions):
                samples_per_client, n_clients = condition
                rounds = sorted(values[(depth, metric)][condition])
                means, lower, upper = [], [], []
                for round_number in rounds:
                    observations = np.asarray(
                        values[(depth, metric)][condition][round_number], dtype=np.float64
                    )
                    mean = float(observations.mean())
                    if observations.size >= 2:
                        half_width = float(
                            1.96 * observations.std(ddof=1) / math.sqrt(observations.size)
                        )
                    else:
                        half_width = 0.0
                    means.append(mean)
                    lower.append(mean - half_width)
                    upper.append(mean + half_width)
                    csv_rows.append({
                        "depth": depth,
                        "metric": metric,
                        "samples_per_client": samples_per_client,
                        "n_clients": n_clients,
                        "communication_round": round_number,
                        "seed_runs": int(observations.size),
                        "mean": mean,
                        "std_across_seeds": float(observations.std(ddof=1))
                        if observations.size >= 2 else 0.0,
                        "ci95_half_width": half_width,
                    })
                label = condition_label(samples_per_client, n_clients)
                axis.plot(rounds, means, marker="o", markersize=3.5, linewidth=1.8,
                          color=color, label=label)
                if any(high > low for low, high in zip(lower, upper)):
                    axis.fill_between(rounds, lower, upper, color=color, alpha=0.14)

            axis.set_xlabel("Communication round")
            axis.set_ylabel(f"{metric_title} (L2-normalized features)")
            axis.set_title(f"{args.title_prefix}: {depth.upper()}")
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            stem = f"{depth}_{metric}"
            fig.savefig(output_dir / f"{stem}.png", dpi=args.dpi, bbox_inches="tight")
            fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
            plt.close(fig)

    csv_path = output_dir / "multiround_feature_geometry_plot_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    logit_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run in runs:
        condition = (run["samples_per_client"], run["n_clients"])
        for record in run["records"]:
            if "postlocal_logit_client_macro" not in record:
                continue
            round_number = int(record["round"]) + 1
            sources = {
                "absolute": record["postlocal_logit_client_macro"],
                "delta": record["delta_from_round_start_global_logit_macro"],
            }
            for mode, source in sources.items():
                for metric in (
                    "off_diagonal_centered_cosine_mean",
                    "directional_variance_mean",
                ):
                    logit_values[(mode, "aggregate", metric)][condition][round_number].append(
                        float(source["depth_summary"][metric]["mean"])
                    )
                for pair in DEPTH_PAIRS:
                    logit_values[(mode, "pair", pair)][condition][round_number].append(
                        float(source["pairwise"][pair]["centered_logit_cosine_mean"]["mean"])
                    )
                for depth in DEPTHS:
                    logit_values[(mode, "sanity_accuracy", depth)][condition][round_number].append(
                        float(source["sanity"][depth]["accuracy_pct"]["mean"])
                    )

    logit_rows = []
    logit_plot_count = 0

    def plot_logit_series(series_key, ylabel, title_suffix, stem):
        nonlocal logit_plot_count
        if series_key not in logit_values:
            return
        fig, axis = plt.subplots(figsize=(7.4, 4.8))
        conditions = sorted(logit_values[series_key], reverse=True)
        for color, condition in zip(colors, conditions):
            samples_per_client, n_clients = condition
            rounds = sorted(logit_values[series_key][condition])
            means, lower, upper = [], [], []
            for round_number in rounds:
                observations = np.asarray(
                    logit_values[series_key][condition][round_number], dtype=np.float64
                )
                mean = float(observations.mean())
                half_width = (
                    float(1.96 * observations.std(ddof=1) / math.sqrt(observations.size))
                    if observations.size >= 2 else 0.0
                )
                means.append(mean)
                lower.append(mean - half_width)
                upper.append(mean + half_width)
                logit_rows.append({
                    "mode": series_key[0],
                    "category": series_key[1],
                    "metric": series_key[2],
                    "samples_per_client": samples_per_client,
                    "n_clients": n_clients,
                    "communication_round": round_number,
                    "seed_runs": int(observations.size),
                    "mean": mean,
                    "std_across_seeds": float(observations.std(ddof=1))
                    if observations.size >= 2 else 0.0,
                    "ci95_half_width": half_width,
                })
            axis.plot(
                rounds, means, marker="o", markersize=3.5, linewidth=1.8,
                color=color, label=condition_label(samples_per_client, n_clients),
            )
            if any(high > low for low, high in zip(lower, upper)):
                axis.fill_between(rounds, lower, upper, color=color, alpha=0.14)
        if series_key[0] == "delta":
            axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2)
        axis.set_xlabel("Communication round")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{args.title_prefix}: {title_suffix}")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{stem}.png", dpi=args.dpi, bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        logit_plot_count += 1

    for mode, mode_title in (("absolute", "post-local"), ("delta", "change from global")):
        plot_logit_series(
            (mode, "aggregate", "off_diagonal_centered_cosine_mean"),
            "Mean pairwise centered-logit cosine",
            f"{mode_title} aggregate cosine",
            f"logit_aggregate_cosine_{mode}",
        )
        plot_logit_series(
            (mode, "aggregate", "directional_variance_mean"),
            "Logit directional variance across depths",
            f"{mode_title} directional variance",
            f"logit_directional_variance_{mode}",
        )
        for pair in DEPTH_PAIRS:
            plot_logit_series(
                (mode, "pair", pair),
                "Centered-logit cosine",
                f"{mode_title} {pair.upper()}",
                f"logit_pair_{pair}_cosine_{mode}",
            )

    if logit_rows:
        logit_csv_path = output_dir / "multiround_logit_geometry_plot_data.csv"
        with logit_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(logit_rows[0]))
            writer.writeheader()
            writer.writerows(logit_rows)
    else:
        logit_csv_path = None
    print(
        f"Loaded {len(runs)} runs; wrote 8 feature and {logit_plot_count} logit "
        f"PNG/PDF plots; feature_csv={csv_path}, logit_csv={logit_csv_path}"
    )


if __name__ == "__main__":
    main()
