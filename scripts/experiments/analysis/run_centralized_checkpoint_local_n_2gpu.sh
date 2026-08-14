#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"
export EXPECTED_GPU_COUNT=2

exec "${SCRIPT_DIR}/run_centralized_checkpoint_local_n_4gpu.sh" "$@"
