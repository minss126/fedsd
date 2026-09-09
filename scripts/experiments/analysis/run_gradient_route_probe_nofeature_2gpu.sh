#!/usr/bin/env bash

# Two-GPU half of the no-feature CE-vs-KD gradient-route analysis.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"
export DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-cifar10}"

exec bash scripts/experiments/analysis/run_gradient_route_probe_nofeature_4gpu.sh
