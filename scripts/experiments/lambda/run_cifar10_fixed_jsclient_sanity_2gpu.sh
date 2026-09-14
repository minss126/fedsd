#!/usr/bin/env bash

# Seed-1 No-JS versus JS-client validation on the two-GPU server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SET=server2 \
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}" \
"${SCRIPT_DIR}/run_cifar10_fixed_jsclient_sanity_4gpu.sh"
