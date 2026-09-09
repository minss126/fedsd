#!/usr/bin/env bash

# Run the beta=.1 adaptive no-feature parameter screen first, then the
# CIFAR-10 no-feature CE/KD gradient analysis on the same two GPUs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"

echo "[1/2] No-feature JS-branch adaptive parameter revalidation (beta=0.1)"
bash scripts/experiments/lambda/run_js_branch_nofeature_tuning_2gpu.sh

echo "[2/2] No-feature CE/KD gradient-route analysis (CIFAR-10)"
bash scripts/experiments/analysis/run_gradient_route_probe_nofeature_2gpu.sh

echo "No-feature tuning and gradient analysis complete."
