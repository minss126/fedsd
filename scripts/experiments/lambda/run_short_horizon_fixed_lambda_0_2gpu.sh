#!/usr/bin/env bash

# Strict lambda=0 control for the 100-round datasets.
# Keeps the ResNet18-BYOT architecture and every training setting unchanged;
# only the branch-KD coefficient is zero.  This is therefore distinct from
# the plain baseline, which has no BYOT branches.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  GPUS_OVERRIDE="0 1" bash scripts/experiments/lambda/run_short_horizon_fixed_lambda_0_2gpu.sh

Runs fixed lambda=0 on TinyImageNet and ImageNet100-64 for
IID / beta={0.5,0.3,0.1}, with ResNet18-BYOT, FedAvg, E=5, C=0.1,
100 rounds, and seed 0.  There is no fixed-lambda warm-up.
EOF
    exit 0
fi

export FIXED_LAMBDAS="0.0"
export LOG_ROOT="${LOG_ROOT:-logs/lambda/analysis/logs_short_horizon_fixed_lambda_0}"

exec bash scripts/experiments/lambda/run_short_horizon_fixed_lambda_3_5_2gpu.sh
