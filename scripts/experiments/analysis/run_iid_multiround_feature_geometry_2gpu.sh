#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}" \
    bash "${SCRIPT_DIR}/run_iid_multiround_feature_geometry.sh" "$@"
