#!/bin/bash

set -euo pipefail

# Re-runs only the selected soft-b adaptive method for the two 100-round
# dataset extensions.  The former 250-round warm-up was designed for a
# 500-round horizon; this launcher uses 50 rounds to preserve the same 10%
# warm-up ratio.  Plain and all fixed-lambda baselines are unchanged.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1\" bash $0"
    echo
    echo "Runs soft-b adaptive only for TinyImageNet and ImageNet100-64:"
    echo "  rounds=100, warm-up=50, partitions=IID/beta={0.5,0.3,0.1}."
    echo "Plain and fixed-lambda runs are intentionally not repeated."
    echo "Expected wall time on two GPUs: about 14-16 hours."
    exit 0
fi

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"
export EXTENSIONS_OVERRIDE="tinyimagenet imagenet100_64"
export ENVS_OVERRIDE="iid beta_0.5 beta_0.3 beta_0.1"
export METHODS_OVERRIDE="soft_b_adaptive"
export ROUNDS="${ROUNDS:-500}"
export TINYIMAGENET_ROUNDS="${TINYIMAGENET_ROUNDS:-100}"
export IMAGENET100_ROUNDS="${IMAGENET100_ROUNDS:-100}"
export LAMBDA_WARMUP="${LAMBDA_WARMUP:-50}"
export LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_extension_short_horizon_warmup50}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "========== Short-Horizon Adaptive Re-validation =========="
echo "extensions=${EXTENSIONS_OVERRIDE}; gpus=${GPUS_OVERRIDE}"
echo "rounds: TinyImageNet=${TINYIMAGENET_ROUNDS}, ImageNet100-64=${IMAGENET100_ROUNDS}; warmup=${LAMBDA_WARMUP}"
echo "methods=${METHODS_OVERRIDE}; envs=${ENVS_OVERRIDE}; log_root=${LOG_ROOT}"

bash scripts/experiments/lambda/run_soft_adaptive_extensions_4gpu.sh
