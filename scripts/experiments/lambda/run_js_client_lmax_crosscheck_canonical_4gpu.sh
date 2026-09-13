#!/usr/bin/env bash

# Four-GPU side: CIFAR-10 IID/beta=.1 x lambda_max={1,2}.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_SET=server4 \
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}" \
"${SCRIPT_DIR}/run_js_client_lmax_crosscheck_canonical.sh"

