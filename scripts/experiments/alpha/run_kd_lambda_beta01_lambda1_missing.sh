#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Fill the missing fixed KD-only lambda=1 result for CIFAR-100 beta=0.1.
#
# Output:
#   logs/alpha/logs_kd_lambda_sweep/beta_0.1/fedavg/kd_lambda1p000.*

GPUS_OVERRIDE="${GPUS_OVERRIDE:-0}" \
ENVS_OVERRIDE="beta_0.1" \
LAMBDAS="1.00:1p000" \
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_kd_lambda_sweep}" \
USE_WANDB="${USE_WANDB:-1}" \
bash scripts/experiments/alpha/run_kd_lambda_sweep.sh
