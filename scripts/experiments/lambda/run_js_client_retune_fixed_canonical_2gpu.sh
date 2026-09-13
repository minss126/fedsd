#!/usr/bin/env bash

# Two-GPU allocation:
#   adaptive JS-client: beta=.1
#   fixed lambda: beta=.3, beta=.1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SET=server2 \
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}" \
"${SCRIPT_DIR}/run_js_client_retune_fixed_canonical.sh"
