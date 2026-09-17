#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLE=resnet4 GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}" \
    bash "$SCRIPT_DIR/run_publication_core_matrix.sh"

