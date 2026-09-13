#!/usr/bin/env bash

# Four-GPU allocation:
#   adaptive JS-client: IID
#   fixed lambda: IID, beta=.5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SET=server4 \
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}" \
"${SCRIPT_DIR}/run_js_client_retune_fixed_canonical.sh"
