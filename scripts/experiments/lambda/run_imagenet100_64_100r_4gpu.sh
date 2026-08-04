#!/bin/bash

set -euo pipefail

# ImageNet100-64 extension screen (100 rounds, 4 GPUs).
# It compares the standard fixed lambda=1 baseline (no warm-up) with the
# selected client-wise soft-b adaptive lambda method.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1 2 3\" bash $0"
    echo
    echo "Runs ImageNet100-64 for 100 rounds:"
    echo "  partitions: beta=0.5, beta=0.1"
    echo "  methods: fixed lambda=1 (no warm-up), soft-b adaptive"
    echo "Expected wall time on four GPUs: about 3.5-4.5 hours."
    exit 0
fi

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}"
export EXTENSIONS_OVERRIDE="imagenet100_64"
export ROUNDS="${ROUNDS:-100}"
export LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_soft_adaptive_imagenet100_64_100r_no_warmup_fixed}"

echo "========== ImageNet100-64 Soft-b Screen (100 rounds) =========="
echo "gpus=${GPUS_OVERRIDE}; log_root=${LOG_ROOT}"

bash scripts/experiments/lambda/run_soft_adaptive_extensions_4gpu.sh
