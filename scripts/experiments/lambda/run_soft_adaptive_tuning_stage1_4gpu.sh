#!/bin/bash

# Run this on the 4-GPU server.  It owns T_KD={0.25, 1.00}, i.e. 12 of the
# 18 stage-1 jobs.  Each GPU receives three sequential jobs.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TUNE_KD_TEMPERATURES="${TUNE_KD_TEMPERATURES:-0.25 1.00}"
export EXPECTED_GPU_COUNT=4
if [ -z "${GPUS_OVERRIDE:-}" ]; then
    export GPUS_OVERRIDE="0 1 2 3"
fi

exec bash "${SCRIPT_DIR}/run_soft_adaptive_tuning_stage1.sh" "$@"
