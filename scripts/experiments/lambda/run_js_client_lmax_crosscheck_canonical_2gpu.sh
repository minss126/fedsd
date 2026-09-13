#!/usr/bin/env bash

# Two-GPU side: CIFAR-100 beta=.3 x lambda_max={1,2}.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_SET=server2 \
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}" \
"${SCRIPT_DIR}/run_js_client_lmax_crosscheck_canonical.sh"

