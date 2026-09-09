#!/usr/bin/env bash

# No-feature JS-branch parameter revalidation: CIFAR-100 beta=0.1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export RUN_SET=tune_2gpu
export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"

exec bash scripts/experiments/lambda/run_js_branch_final_reruns.sh
