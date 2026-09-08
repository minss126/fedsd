#!/usr/bin/env bash

# Four-GPU half of the final JS-branch rerun matrix.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export RUN_SET=cifar_tiny_4gpu
export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}"

exec bash scripts/experiments/lambda/run_js_branch_final_reruns.sh
