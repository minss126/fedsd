#!/usr/bin/env python3
"""Plot matched FedMLB/adaptive accuracy-cost curves from DXFL result pickles."""

import argparse
import csv
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("FedMLB", "Adaptive")
COLORS = {"FedMLB": "#D55E00", "Adaptive": "#0072B2"}
T_CRITICAL_95 = {
    1: 0.0,
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fedmlb", action="append", required=True)
    parser.add_argument("--adaptive", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target-accuracies",
        default="",
        help="Optional comma-separated target accuracies for first-crossing cost table.",
    )
    return parser.parse_args()


def load_result(path):
    with open(path, "rb") as handle:
        result = pickle.load(handle)
    required = ("acc_global", "round_time", "communication_bytes_per_round")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"{path} is missing cost fields: {', '.join(missing)}")
    accuracy = np.asarray(result["acc_global"], dtype=np.float64)
    round_time = np.asarray(result["round_time"], dtype=np.float64)
    if len(accuracy) != len(round_time):
        raise ValueError(
            f"{path}: acc_global has {len(accuracy)} rounds but round_time has {len(round_time)}."
        )
    peak_memory = np.asarray(
        result.get("peak_gpu_memory_bytes", np.zeros_like(round_time)), dtype=np.float64
    )
    if len(peak_memory) != len(round_time):
        peak_memory = np.zeros_like(round_time)
    return {
        "path": path,
        "accuracy": accuracy,
        "cumulative_gpu_hours": np.cumsum(round_time) / 3600.0,
        "round_time": round_time,
        "peak_memory_gib": peak_memory / float(1024**3),
        "communication_bytes_per_round": float(result["communication_bytes_per_round"]),
        "model_state_bytes": float(result.get("model_state_bytes", 0)),
        "model_parameter_count": int(result.get("model_parameter_count", 0)),
        "device_name": str(result.get("device_name", "unknown")),
        "args": result.get("args", {}),
    }


def ci95(values, axis=0):
    values = np.asarray(values, dtype=np.float64)
    count = values.shape[axis]
    if count <= 1:
        shape = list(values.shape)
        del shape[axis]
        return np.zeros(shape, dtype=np.float64)
    critical = T_CRITICAL_95.get(count, 1.96)
    return critical * np.std(values, axis=axis, ddof=1) / math.sqrt(count)


def summarize(runs):
    rounds = min(len(run["accuracy"]) for run in runs)
    accuracy = np.stack([run["accuracy"][:rounds] for run in runs])
    gpu_hours = np.stack([run["cumulative_gpu_hours"][:rounds] for run in runs])
    round_time = np.stack([run["round_time"][:rounds] for run in runs])
    peak_memory = np.stack([run["peak_memory_gib"][:rounds] for run in runs])
    bytes_per_round = np.asarray(
        [run["communication_bytes_per_round"] for run in runs], dtype=np.float64
    )
    completed_rounds = np.arange(1, rounds + 1, dtype=np.float64)
    cumulative_communication = (
        bytes_per_round[:, None] * completed_rounds[None, :] / float(1024**3)
    )
    return {
        "round": completed_rounds,
        "accuracy_runs": accuracy,
        "accuracy_mean": accuracy.mean(axis=0),
        "accuracy_ci95": ci95(accuracy),
        "gpu_hours_runs": gpu_hours,
        "gpu_hours_mean": gpu_hours.mean(axis=0),
        "gpu_hours_ci95": ci95(gpu_hours),
        "round_time_mean": round_time.mean(axis=0),
        "peak_memory_mean": peak_memory.mean(axis=0),
        "peak_memory_max": peak_memory.max(axis=0),
        "communication_gib_runs": cumulative_communication,
        "communication_gib_mean": cumulative_communication.mean(axis=0),
        "parameter_count": int(round(np.mean([run["model_parameter_count"] for run in runs]))),
        "model_state_mib": float(np.mean([run["model_state_bytes"] for run in runs])) / 1024**2,
        "seed_count": len(runs),
    }


def plot_curve(summaries, x_key, xlabel, output_path):
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for method in METHODS:
        summary = summaries[method]
        x_values = summary[x_key]
        accuracy = summary["accuracy_mean"]
        accuracy_ci = summary["accuracy_ci95"]
        axis.plot(x_values, accuracy, color=COLORS[method], linewidth=2.2, label=method)
        axis.fill_between(
            x_values,
            accuracy - accuracy_ci,
            accuracy + accuracy_ci,
            color=COLORS[method],
            alpha=0.18,
            linewidth=0,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Global test accuracy (%)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path + ".png", dpi=220)
    fig.savefig(output_path + ".pdf")
    plt.close(fig)


def write_round_summary(summaries, output_path):
    fields = [
        "method",
        "completed_round",
        "accuracy_mean_pct",
        "accuracy_ci95_pct",
        "cumulative_gpu_hours_mean",
        "cumulative_gpu_hours_ci95",
        "cumulative_communication_gib",
        "round_time_mean_sec",
        "peak_gpu_memory_mean_gib",
        "peak_gpu_memory_max_gib",
        "model_parameters",
        "model_state_mib",
        "seed_count",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            summary = summaries[method]
            for index, completed_round in enumerate(summary["round"]):
                writer.writerow(
                    {
                        "method": method,
                        "completed_round": int(completed_round),
                        "accuracy_mean_pct": summary["accuracy_mean"][index],
                        "accuracy_ci95_pct": summary["accuracy_ci95"][index],
                        "cumulative_gpu_hours_mean": summary["gpu_hours_mean"][index],
                        "cumulative_gpu_hours_ci95": summary["gpu_hours_ci95"][index],
                        "cumulative_communication_gib": summary["communication_gib_mean"][index],
                        "round_time_mean_sec": summary["round_time_mean"][index],
                        "peak_gpu_memory_mean_gib": summary["peak_memory_mean"][index],
                        "peak_gpu_memory_max_gib": summary["peak_memory_max"][index],
                        "model_parameters": summary["parameter_count"],
                        "model_state_mib": summary["model_state_mib"],
                        "seed_count": summary["seed_count"],
                    }
                )


def write_target_summary(summaries, targets, output_path):
    fields = [
        "method",
        "target_accuracy_pct",
        "successful_seeds",
        "total_seeds",
        "first_round_mean",
        "cumulative_gpu_hours_mean",
        "cumulative_communication_gib_mean",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target in targets:
            for method in METHODS:
                summary = summaries[method]
                crossings = []
                for accuracy, hours, communication in zip(
                    summary["accuracy_runs"],
                    summary["gpu_hours_runs"],
                    summary["communication_gib_runs"],
                ):
                    indices = np.flatnonzero(accuracy >= target)
                    if indices.size:
                        index = int(indices[0])
                        crossings.append((index + 1, hours[index], communication[index]))
                if crossings:
                    values = np.asarray(crossings, dtype=np.float64)
                    round_mean, hours_mean, communication_mean = values.mean(axis=0)
                else:
                    round_mean = hours_mean = communication_mean = float("nan")
                writer.writerow(
                    {
                        "method": method,
                        "target_accuracy_pct": target,
                        "successful_seeds": len(crossings),
                        "total_seeds": summary["seed_count"],
                        "first_round_mean": round_mean,
                        "cumulative_gpu_hours_mean": hours_mean,
                        "cumulative_communication_gib_mean": communication_mean,
                    }
                )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    runs = {
        "FedMLB": [load_result(path) for path in args.fedmlb],
        "Adaptive": [load_result(path) for path in args.adaptive],
    }
    device_names = sorted(
        {run["device_name"] for method_runs in runs.values() for run in method_runs}
    )
    if len(device_names) > 1:
        print(
            "WARNING: wall-clock curves combine different GPU models: "
            + ", ".join(device_names)
        )
    summaries = {method: summarize(method_runs) for method, method_runs in runs.items()}

    plot_curve(
        summaries,
        "round",
        "Communication round",
        os.path.join(args.output_dir, "accuracy_vs_round"),
    )
    plot_curve(
        summaries,
        "gpu_hours_mean",
        "Cumulative local-training GPU-hours",
        os.path.join(args.output_dir, "accuracy_vs_cumulative_gpu_hours"),
    )
    plot_curve(
        summaries,
        "communication_gib_mean",
        "Cumulative bidirectional communication (GiB)",
        os.path.join(args.output_dir, "accuracy_vs_cumulative_communication"),
    )
    write_round_summary(
        summaries, os.path.join(args.output_dir, "roundwise_accuracy_cost.csv")
    )

    targets = [
        float(token.strip())
        for token in args.target_accuracies.split(",")
        if token.strip()
    ]
    if targets:
        write_target_summary(
            summaries, targets, os.path.join(args.output_dir, "target_accuracy_cost.csv")
        )

    print(f"Saved matched accuracy-cost results to {args.output_dir}")


if __name__ == "__main__":
    main()
