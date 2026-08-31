#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export EXPECTED_GPU_COUNT=2
export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"

exec "${REPO_ROOT}/scripts/experiments/analysis/run_fedmlb_adaptive_pareto_4gpu.sh" "$@"
