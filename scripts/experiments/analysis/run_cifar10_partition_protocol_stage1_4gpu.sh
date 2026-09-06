#!/usr/bin/env bash

# 4-GPU share of the first-stage protocol decision:
# balanced-500 beta=0.1 seeds={1,2}, plus beta={0.3,0.5} seed=0,
# with four methods (16 runs).  Existing beta=0.1 seed=0 is not repeated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}"
export PROTOCOL_OVERRIDE="balanced500"
export PARTITION_SEED_SPECS_OVERRIDE="${PARTITION_SEED_SPECS_OVERRIDE:-0.1:1 0.1:2 0.3:0 0.5:0}"
export LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_cifar10_balanced500_partition_stage1}"

exec bash "$SCRIPT_DIR/run_cifar10_partition_protocol_stage1_matrix.sh"

