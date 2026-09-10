#!/usr/bin/env bash

# Four-GPU half of the no-feature fixed-lambda/temperature grid.
# Runs IID, beta=.5, and beta=.3. The beta=.1 half belongs on the 2-GPU server.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export ENVS_OVERRIDE="${ENVS_OVERRIDE:-iid beta_0.5 beta_0.3}"
export FIXED_LAMBDAS_OVERRIDE="${FIXED_LAMBDAS_OVERRIDE:-0.1 0.3}"
export KD_TEMPERATURES_OVERRIDE="${KD_TEMPERATURES_OVERRIDE:-0.5 1.0}"
export LOG_ROOT="${LOG_ROOT:-logs/lambda/analysis/logs_fixed_lambda_temperature_no_feature}"

exec bash scripts/experiments/lambda/run_fixed_lambda_temperature_screen_4gpu.sh
