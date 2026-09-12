#!/usr/bin/env bash

# Keep every IID comparison on the four-GPU server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITIONS_OVERRIDE="iid" \
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}" \
"${SCRIPT_DIR}/run_js_granularity_temperature_canonical.sh"
