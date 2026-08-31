#!/usr/bin/env python3
"""Recolor the legacy ablation plot without redrawing text or geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


SOURCE_STANDARD = np.array([127, 127, 127], dtype=float)
SOURCE_PROPOSED = np.array([44, 160, 44], dtype=float)
TARGET_STANDARD = np.array([36, 74, 115], dtype=float)   # muted dark blue
TARGET_PROPOSED = np.array([158, 61, 70], dtype=float)   # muted dark red


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results_plots/ablation_severity_trend.png"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_plots/ablation_severity_trend_muted.png"),
    )
    return parser.parse_args()


def recolor_antialiased_gray(
    rgb: np.ndarray, source: np.ndarray, target: np.ndarray
) -> None:
    exact = np.all(rgb == source.astype(np.uint8), axis=2)
    nearby = binary_dilation(exact, iterations=2)
    grayscale = np.ptp(rgb.astype(np.int16), axis=2) <= 2
    candidates = nearby & grayscale

    # Exact source pixels form the line/marker core.  Neighboring grayscale
    # pixels are anti-aliased mixtures with a light background; estimate the
    # mixture amount from their distance to the local light background.
    values = rgb[candidates, 0].astype(float)
    alpha = np.clip((245.0 - values) / (245.0 - source[0]), 0.0, 1.0)
    background = np.full((len(values), 3), 245.0)
    recolored = alpha[:, None] * target + (1.0 - alpha[:, None]) * background
    rgb[candidates] = np.clip(np.rint(recolored), 0, 255).astype(np.uint8)
    rgb[exact] = target.astype(np.uint8)


def recolor_green(rgb: np.ndarray, source: np.ndarray, target: np.ndarray) -> None:
    pixels = rgb.astype(float)
    green_dominant = (
        (pixels[:, :, 1] - pixels[:, :, 0] > 8)
        & (pixels[:, :, 1] - pixels[:, :, 2] > 8)
    )
    exact = np.all(rgb == source.astype(np.uint8), axis=2)
    candidates = green_dominant | binary_dilation(exact, iterations=1)

    selected = pixels[candidates]
    # The original green differs by 116 between G and R.  That difference
    # estimates foreground coverage even for anti-aliased edge pixels.
    alpha = np.clip((selected[:, 1] - selected[:, 0]) / 116.0, 0.0, 1.0)
    safe = np.maximum(1.0 - alpha, 1e-6)
    background_level = np.clip(
        (selected[:, 0] - alpha * source[0]) / safe, 0.0, 255.0
    )
    background_level[alpha > 0.999] = 255.0
    background = np.repeat(background_level[:, None], 3, axis=1)
    recolored = alpha[:, None] * target + (1.0 - alpha[:, None]) * background
    rgb[candidates] = np.clip(np.rint(recolored), 0, 255).astype(np.uint8)
    rgb[exact] = target.astype(np.uint8)


def main() -> None:
    args = parse_args()
    image = Image.open(args.input).convert("RGB")
    rgb = np.asarray(image).copy()

    recolor_antialiased_gray(rgb, SOURCE_STANDARD, TARGET_STANDARD)
    recolor_green(rgb, SOURCE_PROPOSED, TARGET_PROPOSED)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.output, optimize=True)
    print(f"PNG: {args.output.resolve()}")


if __name__ == "__main__":
    main()
