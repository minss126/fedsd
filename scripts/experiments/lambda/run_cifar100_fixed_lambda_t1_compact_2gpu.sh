#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Two-GPU wrapper for the compact, no-warm-up, fixed-lambda T_KD=1 sweep.
# The shared launcher runs lambda={0.1,0.3,3,5} on
# IID/beta={0.5,0.3,0.1}; lambda=0 and lambda=1 are reused separately.

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"
export EXPECTED_GPU_COUNT=2

exec bash scripts/experiments/lambda/run_cifar100_fixed_lambda_t1_compact_4gpu.sh
