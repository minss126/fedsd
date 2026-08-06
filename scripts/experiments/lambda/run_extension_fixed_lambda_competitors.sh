#!/bin/bash

set -euo pipefail

# Adds only the two strong fixed-lambda competitors to completed extension
# results.  Plain, fixed lambda=1, and soft-b adaptive are not rerun.
# Pass a disjoint EXTENSIONS_OVERRIDE on each server to distribute the work.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage examples:"
    echo "  # 4-GPU server"
    echo "  GPUS_OVERRIDE=\"0 1 2 3\" EXTENSIONS_OVERRIDE=\"tinyimagenet imagenet100_64 fedprox\" bash $0"
    echo "  # 2-GPU server"
    echo "  GPUS_OVERRIDE=\"0 1\" EXTENSIONS_OVERRIDE=\"mobilenet moon\" bash $0"
    echo
    echo "Adds fixed lambda={0.1,0.3} on IID/beta={0.5,0.3,0.1}."
    echo "Expected combined wall time with the recommended 4+2 GPU split: about 20-23 hours."
    exit 0
fi

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}"
export EXTENSIONS_OVERRIDE="${EXTENSIONS_OVERRIDE:-tinyimagenet imagenet100_64 mobilenet fedprox moon}"
export ENVS_OVERRIDE="${ENVS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1}"
export METHODS_OVERRIDE="fixed_lambda0p10 fixed_lambda0p30"
export ROUNDS="${ROUNDS:-500}"
export TINYIMAGENET_ROUNDS="${TINYIMAGENET_ROUNDS:-100}"
export IMAGENET100_ROUNDS="${IMAGENET100_ROUNDS:-100}"
export LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_extension_fixed_lambda_competitors}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "========== Extension Fixed-Lambda Competitors =========="
echo "extensions=${EXTENSIONS_OVERRIDE}; gpus=${GPUS_OVERRIDE}; fixed={0.1,0.3}"

bash scripts/experiments/lambda/run_soft_adaptive_extensions_4gpu.sh
