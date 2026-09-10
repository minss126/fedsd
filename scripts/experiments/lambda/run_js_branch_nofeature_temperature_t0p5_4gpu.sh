#!/usr/bin/env bash

# Final JS-branch/no-feature temperature screen.
# Runs CIFAR-100 IID, beta=0.5, beta=0.3, and beta=0.1 at KD T=0.5;
# all other final-method hyperparameters remain fixed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export RUN_SET=temperature_t0p5
export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}"
export KD_TEMPERATURE=0.5
export PROXY_TEMPERATURE=1.0
export MIN_REQUIRE_SIZE=64
export FEATURE_BETA_OVERRIDE=0.0
export LAMBDA_MAX=1.0
export SKEW_POWER=2.0
export SOFT_TAU=0.85
export SOFT_TEMPERATURE=0.05
export JS_GAIN=1.0
export LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_js_branch_temperature_t0p5_no_feature}"

exec bash scripts/experiments/lambda/run_js_branch_final_reruns.sh
