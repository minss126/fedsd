#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}" \
    exec "${SCRIPT_DIR}/run_centralized_prefix_gradient_agreement.sh" "$@"

