#!/usr/bin/env bash

# Run only the missing seed-0 controls for the KD-necessity decision:
# paired Plain, a strength-matched constant gate, and one client-wise JS gate.
# The main launcher reuses completed Current and branch-wise JS trajectories in
# the final analysis, so those expensive jobs are not repeated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export METHODS_OVERRIDE="${METHODS_OVERRIDE:-plain constant js_client}"
export TIME_ESTIMATE="${TIME_ESTIMATE:-about 10-14 hours}"
exec "${SCRIPT_DIR}/run_kd_need_proxy_seed0_4gpu.sh" "$@"
