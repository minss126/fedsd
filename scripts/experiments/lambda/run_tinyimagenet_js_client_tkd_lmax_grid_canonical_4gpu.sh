#!/usr/bin/env bash

# Four-GPU side: TinyImageNet IID, full T_KD x lambda_max grid.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_SET=server4 \
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}" \
"${SCRIPT_DIR}/run_tinyimagenet_js_client_tkd_lmax_grid_canonical.sh"

