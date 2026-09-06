#!/usr/bin/env bash

# 2-GPU share of the first-stage protocol decision:
# min-10, beta={0.3,0.5}, seed=0, four methods (8 runs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"
export PROTOCOL_OVERRIDE="min10"
export PARTITION_SEED_SPECS_OVERRIDE="${PARTITION_SEED_SPECS_OVERRIDE:-0.3:0 0.5:0}"
export LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_cifar10_min10_partition_stage1}"

exec bash "$SCRIPT_DIR/run_cifar10_partition_protocol_stage1_matrix.sh"

