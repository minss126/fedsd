#!/usr/bin/env python3
"""Plot FedPart-style local update-step trajectories for plain vs adaptive FL."""

import argparse
import csv
import glob
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {"Plain": "#4C78A8", "Adaptive": "#E45756"}


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    unique = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def load_run(path, group, metric, round_start, round_end):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    rounds = payload.get("update_step_size_rounds", [])
    if not rounds:
        raise ValueError(f"No update-step records in {path}; was --log_update_step_size enabled?")

    selected = [
        record for record in rounds
        if int(record["round"]) >= round_start
        and (round_end is None or int(record["round"]) <= round_end)
    ]
    if not selected:
        raise ValueError(f"No rounds in requested range for {path}")

    x_values = []
    y_values = []
    boundaries = []
    rows = []
    cursor = 0
    ratios = []
    ratio_rounds = []
    previous_round_last = None
    for round_record in selected:
        boundaries.append(cursor)
        current_round_values = []
        for step in round_record["step_summaries"]:
            try:
                summary = step["groups"][group][metric]
            except KeyError as error:
                raise KeyError(
                    f"Missing group/metric {group}/{metric} in {path}"
                ) from error
            x_values.append(cursor)
            y_values.append(float(summary["mean"]))
            current_round_values.append(float(summary["mean"]))
            rows.append({
                "source": path,
                "round": int(round_record["round"]),
                "local_step": int(step["local_step"]),
                "iteration": int(cursor),
                "mean": float(summary["mean"]),
                "client_std": float(summary["std"]),
                "client_count": int(summary["count"]),
                "lr": float(step["lr_mean"]),
            })
            cursor += 1
        if previous_round_last is not None and current_round_values:
            ratio_rounds.append(int(round_record["round"]))
            ratios.append(float(current_round_values[0] / max(previous_round_last, 1e-12)))
        if current_round_values:
            previous_round_last = current_round_values[-1]

    return {
        "path": path,
        "x": np.asarray(x_values, dtype=np.int64),
        "y": np.asarray(y_values, dtype=np.float64),
        "boundaries": boundaries,
        "ratio_rounds": np.asarray(ratio_rounds, dtype=np.int64),
        "ratios": np.asarray(ratios, dtype=np.float64),
        "rows": rows,
    }


def common_curve(runs, field="y"):
    length = min(len(run[field]) for run in runs)
    matrix = np.stack([run[field][:length] for run in runs], axis=0)
    mean = matrix.mean(axis=0)
    if matrix.shape[0] > 1:
        ci = 1.96 * matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
    else:
        ci = np.zeros_like(mean)
    return mean, ci, length


def plot_trajectory(method_runs, output_path, ylabel, max_boundaries=40):
    fig, axis = plt.subplots(figsize=(13.2, 4.8))
    for method, runs in method_runs.items():
        mean, ci, length = common_curve(runs, "y")
        x = np.arange(length)
        color = METHOD_COLORS[method]
        axis.plot(x, mean, color=color, linewidth=1.15, label=f"{method} (seeds={len(runs)})")
        if len(runs) > 1:
            axis.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.16, linewidth=0)

    reference_boundaries = next(iter(method_runs.values()))[0]["boundaries"]
    stride = max(1, int(np.ceil(len(reference_boundaries) / max_boundaries)))
    for boundary in reference_boundaries[::stride]:
        axis.axvline(boundary, color="#777777", alpha=0.16, linewidth=0.65)
    axis.set_xlabel("Local optimizer iteration (rounds concatenated)")
    axis.set_ylabel(ylabel)
    axis.set_title("Post-aggregation local update-step trajectory")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_zoom(method_runs, output_path, ylabel, zoom_rounds):
    fig, axis = plt.subplots(figsize=(13.2, 4.8))
    for method, runs in method_runs.items():
        mean, ci, length = common_curve(runs, "y")
        boundaries = [value for value in runs[0]["boundaries"] if value < length]
        start = boundaries[max(0, len(boundaries) - zoom_rounds)] if boundaries else 0
        x = np.arange(start, length)
        color = METHOD_COLORS[method]
        axis.plot(x, mean[start:length], color=color, linewidth=1.35, label=method)
        if len(runs) > 1:
            axis.fill_between(
                x, mean[start:length] - ci[start:length], mean[start:length] + ci[start:length],
                color=color, alpha=0.17, linewidth=0,
            )
        for boundary in boundaries:
            if boundary >= start:
                axis.axvline(boundary, color="#666666", alpha=0.3, linewidth=0.8)
    axis.set_xlabel("Local optimizer iteration")
    axis.set_ylabel(ylabel)
    axis.set_title(f"Last {zoom_rounds} rounds (vertical lines: aggregation/broadcast boundaries)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_spike_ratio(method_runs, output_path):
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    for method, runs in method_runs.items():
        length = min(len(run["ratios"]) for run in runs)
        matrix = np.stack([run["ratios"][:length] for run in runs], axis=0)
        rounds = runs[0]["ratio_rounds"][:length]
        mean = matrix.mean(axis=0)
        ci = (
            1.96 * matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
            if matrix.shape[0] > 1 else np.zeros_like(mean)
        )
        color = METHOD_COLORS[method]
        axis.plot(rounds, mean, color=color, linewidth=1.5, label=method)
        if len(runs) > 1:
            axis.fill_between(rounds, mean - ci, mean + ci, color=color, alpha=0.17, linewidth=0)
    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Communication round")
    axis.set_ylabel("First step after aggregation / last step before aggregation")
    axis.set_title("Aggregation-boundary update rebound")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain", action="append", required=True, help="Plain result pickle or glob; repeatable")
    parser.add_argument("--adaptive", action="append", required=True, help="Adaptive result pickle or glob; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group", default="shared_teacher")
    parser.add_argument(
        "--metric", default="fedpart_l2sum",
        choices=(
            "fedpart_l2sum", "l2", "relative_l2", "gradient_l2", "parameter_l2",
            "common_ce_gradient_l2", "common_ce_gradient_l2sum",
        ),
    )
    parser.add_argument("--round-start", type=int, default=0)
    parser.add_argument("--round-end", type=int)
    parser.add_argument("--zoom-rounds", type=int, default=10)
    args = parser.parse_args()

    paths = {
        "Plain": expand_paths(args.plain),
        "Adaptive": expand_paths(args.adaptive),
    }
    for method, method_paths in paths.items():
        if not method_paths or any(not os.path.isfile(path) for path in method_paths):
            raise FileNotFoundError(f"Missing {method} result(s): {method_paths}")

    method_runs = {
        method: [
            load_run(path, args.group, args.metric, args.round_start, args.round_end)
            for path in method_paths
        ]
        for method, method_paths in paths.items()
    }
    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"{args.group}_{args.metric}"
    ylabel = {
        "fedpart_l2sum": r"Update step size  $\sum_p\|\Delta\theta_p\|_2$",
        "l2": r"Update step size  $\|\Delta\theta\|_2$",
        "relative_l2": r"Relative update  $\|\Delta\theta\|_2/\|\theta\|_2$",
        "gradient_l2": r"Gradient norm  $\|\nabla L\|_2$",
        "parameter_l2": r"Parameter norm  $\|\theta\|_2$",
        "common_ce_gradient_l2": r"Common Final-CE gradient  $\|\nabla L_{CE}\|_2$",
        "common_ce_gradient_l2sum": r"Common Final-CE gradient  $\sum_p\|\nabla_p L_{CE}\|_2$",
    }[args.metric]
    plot_trajectory(method_runs, os.path.join(args.output_dir, f"{stem}_trajectory.png"), ylabel)
    plot_zoom(method_runs, os.path.join(args.output_dir, f"{stem}_zoom.png"), ylabel, args.zoom_rounds)
    plot_spike_ratio(method_runs, os.path.join(args.output_dir, f"{stem}_spike_ratio.png"))

    csv_path = os.path.join(args.output_dir, f"{stem}_values.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "method", "source", "round", "local_step", "iteration",
            "mean", "client_std", "client_count", "lr",
        ))
        writer.writeheader()
        for method, runs in method_runs.items():
            for run in runs:
                for row in run["rows"]:
                    writer.writerow({"method": method, **row})

    print(f"Saved update-step plots and values to {args.output_dir}")


if __name__ == "__main__":
    main()
