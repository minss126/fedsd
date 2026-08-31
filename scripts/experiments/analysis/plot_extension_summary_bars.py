#!/usr/bin/env python3
"""Create a paper-friendly summary of model and FL extensions.

Layout
------
Columns: CIFAR-100, TinyImageNet, ImageNet100-64
Top row: ResNet18 and MobileNetV2 on IID
Bottom row: FedAvg, FedProx, and MOON, averaged over beta={0.5, 0.3}

Every bar is the last-window mean test accuracy of Plain, Fixed lambda=0.3,
or the final soft-b Adaptive method.  The script uses an explicit manifest
because the completed runs are split across several historical log roots.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dxfl-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = (
    ("cifar100", "CIFAR-100"),
    ("tinyimagenet", "TinyImageNet"),
    ("imagenet100_64", "ImageNet100-64"),
)
METHODS = ("Plain", r"Fixed $\lambda=0.3$", "Adaptive")
METHOD_KEYS = ("plain", "fixed", "adaptive")
# Paper-style gray/blue/red palette.  The proposed method receives the
# muted-red accent, while the fixed baseline uses muted blue.
METHOD_COLORS = ("#B8B8B8", "#4C72B0", "#C44E52")
METHOD_COLORS = ("#B0B0B0", "#4C72B0", "#DD8452")
MODELS = ("ResNet18", "MobileNetV2")
MECHANISMS = ("FedAvg", "FedProx", "MOON")
FL_PARTITIONS = ("beta_0.5", "beta_0.3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root containing logs/ (default: inferred from script path).",
    )
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/adaptive_lambda/presentation"),
    )
    parser.add_argument(
        "--stem", default="extension_summary_no_fedavgm_grouped_bars",
        help="Output filename stem.",
    )
    return parser.parse_args()


def first_existing(candidates: list[Path], description: str) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    formatted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing {description}; checked:\n  {formatted}")


def load_last_mean(path: Path, window: int) -> float:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    values = payload.get("acc_global") or payload.get("acc")
    if not isinstance(values, (list, tuple)) or len(values) < window:
        raise ValueError(
            f"{path} has only {len(values) if values is not None else 0} accuracy values; "
            f"at least {window} are required."
        )
    return float(np.mean(np.asarray(values[-window:], dtype=float)))


class RunManifest:
    """Resolve the exact completed run used for each plotted bar."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.logs = repo_root / "logs"
        self.adaptive = self.logs / "lambda" / "adaptive"

    def model(self, dataset: str, model: str, method: str) -> Path:
        a = self.adaptive
        l = self.logs
        if dataset == "cifar100" and model == "ResNet18":
            paths = {
                "plain": l / "reliability/logs_iid_fedavg_compare/iid/fedavg/plain_baseline.pkl",
                "fixed": l / "lambda/analysis/logs_cifar100_fixed_lambda_t1_compact/iid/fedavg/fixed_lambda0p30_tkd1p00.pkl",
                "adaptive": a / "logs_soft_adaptive_tuning_stage1/iid/fedavg/soft_b_tkd1p00_lmax1p00_warm250_tau0p85.pkl",
            }
            return paths[method]

        if dataset == "cifar100" and model == "MobileNetV2":
            paths = {
                "plain": a / "logs_extension_completion/mobilenet/iid/fedavg/plain_r500.pkl",
                "fixed": a / "logs_extension_fixed_lambda_competitors/mobilenet/iid/fedavg/fixed_lambda0p30_tkd1p00_r500.pkl",
                "adaptive": a / "logs_extension_completion/mobilenet/iid/fedavg/soft_b_tkd1p00_lmax1p00_warm250_tau0p85_r500.pkl",
            }
            return paths[method]

        if model == "ResNet18":
            filename = {
                "plain": "plain_r100.pkl",
                "fixed": "fixed_lambda0p30_tkd1p00_r100.pkl",
                "adaptive": "soft_b_tkd1p00_lmax1p00_warm50_tau0p85_r100.pkl",
            }[method]
            root = {
                "plain": "logs_extension_completion",
                "fixed": "logs_extension_fixed_lambda_competitors",
                "adaptive": "logs_extension_short_horizon_warmup50",
            }[method]
            return a / root / dataset / "iid" / "fedavg" / filename

        if model == "MobileNetV2":
            filename = {
                "plain": "plain_r100.pkl",
                "fixed": "fixed_lambda0p30_tkd1p00_r100.pkl",
                "adaptive": "soft_b_tkd1p00_lmax1p00_warm50_tau0p85_r100.pkl",
            }[method]
            return (
                a / "logs_final_extension_matrix/model/mobilenet" / dataset
                / "iid/mobilenet" / filename
            )
        raise KeyError((dataset, model, method))

    def fedavg(self, dataset: str, partition: str, method: str) -> Path:
        a = self.adaptive
        l = self.logs
        if dataset == "cifar100":
            paths = {
                "plain": l / "baseline/logs_plain_baseline_compare" / partition
                / "fedavg/baseline.pkl",
                "fixed": l / "lambda/analysis/logs_cifar100_fixed_lambda_t1_compact"
                / partition / "fedavg/fixed_lambda0p30_tkd1p00.pkl",
                "adaptive": a / "logs_soft_adaptive_tuning_stage1" / partition
                / "fedavg/soft_b_tkd1p00_lmax1p00_warm250_tau0p85.pkl",
            }
            return paths[method]

        filename = {
            "plain": "plain_r100.pkl",
            "fixed": "fixed_lambda0p30_tkd1p00_r100.pkl",
            "adaptive": "soft_b_tkd1p00_lmax1p00_warm50_tau0p85_r100.pkl",
        }[method]
        root = {
            "plain": "logs_extension_completion",
            "fixed": "logs_extension_fixed_lambda_competitors",
            "adaptive": "logs_extension_short_horizon_warmup50",
        }[method]
        return a / root / dataset / partition / "fedavg" / filename

    def cifar100_mechanism(
        self, mechanism: str, partition: str, method: str
    ) -> Path:
        if mechanism == "FedAvg":
            return self.fedavg("cifar100", partition, method)

        tag = mechanism.lower()
        if method == "plain":
            return (
                self.adaptive / "logs_extension_completion" / tag / partition
                / "fedavg/plain_r500.pkl"
            )
        if method == "fixed":
            return (
                self.adaptive / "logs_extension_fixed_lambda_competitors" / tag
                / partition / "fedavg/fixed_lambda0p30_tkd1p00_r500.pkl"
            )
        adaptive_root = (
            "logs_extension_completion"
            if partition == "beta_0.3"
            else "logs_soft_adaptive_extensions_no_warmup_fixed"
        )
        return (
            self.adaptive / adaptive_root / tag / partition
            / "fedavg/soft_b_tkd1p00_lmax1p00_warm250_tau0p85_r500.pkl"
        )

    def image_mechanism(
        self, dataset: str, mechanism: str, partition: str, method: str
    ) -> Path:
        if mechanism == "FedAvg":
            return self.fedavg(dataset, partition, method)

        tag = mechanism.lower()
        filename = {
            "plain": "plain_r100.pkl",
            "fixed": "fixed_lambda0p30_tkd1p00_r100.pkl",
            "adaptive": "soft_b_tkd1p00_lmax1p00_warm50_tau0p85_r100.pkl",
        }[method]
        regular = (
            self.adaptive / "logs_ofat_matrix_no_resnet50" / dataset
            / f"mechanism_{tag}/default" / partition / f"mechanism_{tag}" / filename
        )
        fallbacks = [regular]
        if (
            dataset == "tinyimagenet"
            and mechanism == "FedProx"
            and partition == "beta_0.5"
            and method == "adaptive"
        ):
            fallbacks.append(
                self.adaptive / "logs_final_extension_matrix/mechanism/fedprox"
                / "tinyimagenet/beta_0.5/fedprox" / filename
            )
        if (
            dataset == "imagenet100_64"
            and mechanism == "FedProx"
            and partition == "beta_0.5"
            and method == "plain"
        ):
            fallbacks.append(
                self.adaptive / "logs_final_extension_matrix/mechanism/fedprox"
                / "imagenet100_64/beta_0.5/fedprox" / filename
            )
        return first_existing(
            fallbacks, f"{dataset}/{mechanism}/{partition}/{method}"
        )

    def mechanism(
        self, dataset: str, mechanism: str, partition: str, method: str
    ) -> Path:
        if dataset == "cifar100":
            return self.cifar100_mechanism(mechanism, partition, method)
        return self.image_mechanism(dataset, mechanism, partition, method)

def collect_results(manifest: RunManifest, window: int) -> tuple[dict, list[dict]]:
    results: dict = {"model": {}, "fl": {}}
    rows: list[dict] = []

    for dataset, dataset_label in DATASETS:
        results["model"][dataset] = {}
        for model in MODELS:
            values = []
            for method, method_label in zip(METHOD_KEYS, METHODS):
                path = manifest.model(dataset, model, method)
                if not path.is_file():
                    raise FileNotFoundError(path)
                value = load_last_mean(path, window)
                values.append(value)
                rows.append(
                    {
                        "section": "Model (IID)",
                        "dataset": dataset_label,
                        "setting": model,
                        "partition": "IID",
                        "method": method_label.replace("$", ""),
                        "last_window_accuracy": value,
                        "source": str(path.relative_to(manifest.repo_root)),
                    }
                )
            results["model"][dataset][model] = values

        results["fl"][dataset] = {}
        for mechanism in MECHANISMS:
            method_values = []
            for method, method_label in zip(METHOD_KEYS, METHODS):
                partition_values = []
                sources = []
                for partition in FL_PARTITIONS:
                    path = manifest.mechanism(dataset, mechanism, partition, method)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    partition_values.append(load_last_mean(path, window))
                    sources.append(str(path.relative_to(manifest.repo_root)))
                value = float(np.mean(partition_values))
                method_values.append(value)
                rows.append(
                    {
                        "section": "FL algorithm",
                        "dataset": dataset_label,
                        "setting": mechanism,
                        "partition": "mean(beta=0.5,beta=0.3)",
                        "method": method_label.replace("$", ""),
                        "last_window_accuracy": value,
                        "source": " | ".join(sources),
                    }
                )
            results["fl"][dataset][mechanism] = method_values

    return results, rows


def add_grouped_bars(
    ax: plt.Axes,
    labels: tuple[str, ...],
    values_by_label: dict[str, list[float]],
    decimals: int,
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.24
    offsets = (-width, 0.0, width)
    values = np.asarray([values_by_label[label] for label in labels], dtype=float)

    for index, (method, color, offset) in enumerate(
        zip(METHODS, METHOD_COLORS, offsets)
    ):
        bars = ax.bar(
            x + offset,
            values[:, index],
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=method,
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.{decimals}f}" for value in values[:, index]],
            padding=2,
            fontsize=7.5,
            rotation=0,
            color="#374151",
        )

    ax.set_xticks(x, labels)
    ax.grid(axis="y", color="#D6DAE1", linewidth=0.8, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)


def make_figure(results: dict, window: int, decimals: int) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17.2, 9.2),
        sharey="col",
        constrained_layout=True,
    )

    dataset_limits = {
        "cifar100": (0, 80),
        "tinyimagenet": (0, 52),
        "imagenet100_64": (0, 68),
    }
    for column, (dataset, dataset_label) in enumerate(DATASETS):
        model_ax = axes[0, column]
        fl_ax = axes[1, column]
        add_grouped_bars(model_ax, MODELS, results["model"][dataset], decimals)
        add_grouped_bars(fl_ax, MECHANISMS, results["fl"][dataset], decimals)
        model_ax.set_title(dataset_label, fontsize=14, pad=12)
        model_ax.set_ylim(*dataset_limits[dataset])
        fl_ax.set_ylim(*dataset_limits[dataset])
        model_ax.set_xlabel("Model (IID)", labelpad=8)
        fl_ax.set_xlabel(
            r"FL algorithm (mean over $\beta\in\{0.5,0.3\}$)", labelpad=8
        )

    axes[0, 0].set_ylabel(f"Last-{window} accuracy (%)")
    axes[1, 0].set_ylabel(f"Last-{window} accuracy (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
        fontsize=11,
    )
    fig.suptitle(
        "Generalization across Datasets, Models, and FL Algorithms",
        fontsize=18,
        fontweight="bold",
        y=1.075,
    )
    fig.text(
        0.5,
        -0.018,
        "Bars report mean test accuracy over the final "
        f"{window} communication rounds. Fixed uses a constant $\\lambda=0.3$.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4B5563",
    )
    return fig


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(repo_root)
    results, rows = collect_results(manifest, args.window)
    figure = make_figure(results, args.window, args.decimals)

    stem = output_dir / args.stem
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    csv_path = stem.with_suffix(".csv")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"SVG: {svg_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
