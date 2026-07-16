#!/usr/bin/env python3
"""Visualize tail stability for the signal/power lambda ablations."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = [
    ("label_prob_p1", "Label prob (p=1)"),
    ("label_prob_p2", "Label prob (p=2)"),
    ("teacher_certainty_p1", "Certainty (p=1)"),
    ("teacher_certainty_p2", "Certainty (p=2)"),
    ("label_prob_x_certainty_p1", "Label prob × certainty (p=1)"),
    ("label_prob_x_certainty_p2", "Label prob × certainty (p=2)"),
    ("label_prob_x_client_pred_entropy_p1", "Label prob × client entropy (p=1)"),
    ("label_prob_x_client_pred_entropy_p2", "Label prob × client entropy (p=2)"),
]
PARTITIONS = ["iid", "beta_0.5", "beta_0.3", "beta_0.1"]
PARTITION_LABELS = {"iid": "IID", "beta_0.5": "Beta = 0.5", "beta_0.3": "Beta = 0.3", "beta_0.1": "Beta = 0.1"}
RESULT_RE = re.compile(r"Round (\d+) result: Acc=([0-9.]+)")


def load_accuracy(path: Path):
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RESULT_RE.search(line)
        if match:
            values[int(match.group(1))] = float(match.group(2))
    return [values[round_idx] for round_idx in sorted(values)]


def moving_average(values, window):
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(sum(values[start : index + 1]) / (index - start + 1))
    return result


def load_root(root: Path):
    curves = defaultdict(dict)
    for partition in PARTITIONS:
        for method, _ in METHODS:
            path = root / partition / "fedavg" / f"{method}.log"
            if not path.exists():
                raise FileNotFoundError(path)
            acc = load_accuracy(path)
            if len(acc) != 500:
                raise ValueError(f"Expected 500 accuracy points in {path}, got {len(acc)}")
            curves[partition][method] = acc
    return curves


def plot_trajectories(curves, title, output_path: Path, start_round):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for axis, partition in zip(axes, PARTITIONS):
        for index, (method, label) in enumerate(METHODS):
            values = curves[partition][method]
            rounds = list(range(start_round, len(values)))
            tail = values[start_round:]
            color = colors[index]
            axis.plot(rounds, tail, color=color, alpha=0.18, linewidth=0.8)
            axis.plot(rounds, moving_average(values, 10)[start_round:], color=color, linewidth=1.7, label=label)
        axis.set_title(PARTITION_LABELS[partition])
        axis.set_xlabel("Round")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy (%)\n(thin: raw, thick: 10-round moving avg)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_window_sensitivity(curves, title, output_path: Path):
    windows = list(range(5, 101, 5))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for axis, partition in zip(axes, PARTITIONS):
        for index, (method, label) in enumerate(METHODS):
            values = curves[partition][method]
            tail_means = [sum(values[-window:]) / window for window in windows]
            axis.plot(windows, tail_means, color=colors[index], linewidth=1.7, label=label)
        axis.axvline(10, color="gray", linestyle="--", linewidth=1)
        axis.axvline(30, color="gray", linestyle="--", linewidth=1)
        axis.axvline(50, color="gray", linestyle="--", linewidth=1)
        axis.set_title(PARTITION_LABELS[partition])
        axis.set_xlabel("Tail window length (rounds)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean tail accuracy (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = load_root(args.root)
    plot_trajectories(curves, f"Signal/Power Ablation Tail Trajectories ({args.label})", args.output_dir / "tail_trajectory.png", 400)
    plot_window_sensitivity(curves, f"Signal/Power Ablation Tail-Window Sensitivity ({args.label})", args.output_dir / "tail_window_sensitivity.png")


if __name__ == "__main__":
    main()
