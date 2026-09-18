#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Final reduced MobileNet scope: seed 0 and FedAvg only. Completed outputs
# copied from mobile_a and exact prior results are skipped automatically.
ROLE=mobile_seed0_fedavg GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}" \
    bash "$SCRIPT_DIR/run_publication_core_matrix.sh"
