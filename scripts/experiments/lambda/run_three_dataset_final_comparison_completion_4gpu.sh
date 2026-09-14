#!/usr/bin/env bash

# 4-GPU side: TinyImageNet seed-0 fixed completion only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SET=server4 \
INCLUDE_VALIDATION="${INCLUDE_VALIDATION:-0}" \
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}" \
"${SCRIPT_DIR}/run_three_dataset_final_comparison_completion.sh"
