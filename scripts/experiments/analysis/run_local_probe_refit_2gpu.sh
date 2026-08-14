#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"

exec "${SCRIPT_DIR}/run_local_probe_refit_4gpu.sh" "$@"
