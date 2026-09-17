#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLE=mobile_a GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}" \
    bash "$SCRIPT_DIR/run_publication_core_matrix.sh"

