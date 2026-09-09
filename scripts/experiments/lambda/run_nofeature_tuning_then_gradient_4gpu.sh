#!/usr/bin/env bash

# Run the adaptive no-feature parameter screen first, then the independent
# no-feature CE/KD gradient analysis on the same four GPUs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}"

echo "[1/2] No-feature JS-branch adaptive parameter revalidation"
bash scripts/experiments/lambda/run_js_branch_nofeature_tuning_4gpu.sh

echo "[2/2] No-feature CE/KD gradient-route analysis (CIFAR-100)"
DATASETS_OVERRIDE=cifar100 \
    bash scripts/experiments/analysis/run_gradient_route_probe_nofeature_4gpu.sh

echo "No-feature tuning and gradient analysis complete."
