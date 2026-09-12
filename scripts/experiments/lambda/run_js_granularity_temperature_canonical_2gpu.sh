#!/usr/bin/env bash

# Keep every beta=.1 comparison on the two-GPU server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITIONS_OVERRIDE="beta_0.1" \
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}" \
"${SCRIPT_DIR}/run_js_granularity_temperature_canonical.sh"
