#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Sequential 4-GPU pipeline:
#   1) CIFAR-100 standard fixed lambda=1 baseline (no warm-up), all four
#      partitions;
#   2) extension screen, with the same no-warm-up fixed baseline versus the
#      selected soft-b adaptive method.
#
# The two underlying scripts preserve separate log roots, so their outputs do
# not overwrite the earlier fixed-with-warm-up runs.

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1 2 3\" bash $0"
    echo
    echo "Default extensions: tinyimagenet mobilenet fedprox moon"
    echo "Add ImageNet100-64 with: EXTENSIONS_OVERRIDE=\"tinyimagenet imagenet100_64 mobilenet fedprox moon\""
    echo "Expected time: 22-29h by default, or 41-50h with ImageNet100-64."
    exit 0
fi

GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}"
export GPUS_OVERRIDE

if [ "${#GPUS_OVERRIDE}" -eq 0 ]; then
    echo "Set GPUS_OVERRIDE to four GPU ids." >&2
    exit 1
fi

# ImageNet100-64 is intentionally opt-in for the default 3-day schedule.
EXTENSIONS_OVERRIDE="${EXTENSIONS_OVERRIDE:-tinyimagenet mobilenet fedprox moon}"
export EXTENSIONS_OVERRIDE

echo "========== Phase 1/2: CIFAR-100 Fixed Lambda=1 (No Warm-up) =========="
bash scripts/experiments/lambda/run_cifar100_fixed_lambda1_no_warmup_4gpu.sh

echo "========== Phase 2/2: Soft-b Extension Screen =========="
bash scripts/experiments/lambda/run_soft_adaptive_extensions_4gpu.sh

echo "Fixed baseline and extension pipeline complete."
