#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Conditional follow-up to run only if the entropy diagnostics show that
# high marginal prediction entropy b often co-occurs with high sample
# uncertainty u.  This replaces the current skew proxy b with
#   d = H(mean_i p_i)/log(C) - mean_i H(p_i)/log(C)
# while preserving every other setting of the proxy-temperature diagnostic.
#
# The paired b-proxy results are stored by
# run_proxy_temperature_entropy_diagnostics.sh.  This script runs only the
# two d-proxy corrections (plain d^2 and soft d relaxation) across the same
# three partitions, for six new runs total.  d has a different empirical
# scale from b, so its soft threshold must be chosen from the first
# diagnostic run instead of reusing b's tau=0.85.

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1\" bash $0"
    echo
    echo "Runs d=prediction_mutual_info as the client skew proxy."
    echo "Defaults: IID/beta_0.5/beta_0.1, plain d^2 and soft d relaxation."
    echo "Required: SOFT_TAU=<value selected from the d diagnostic distribution>."
    exit 0
fi

if [ -z "${SOFT_TAU+x}" ]; then
    echo "Set SOFT_TAU after inspecting the d distribution from the entropy diagnostic run." >&2
    echo "Example: SOFT_TAU=0.20 GPUS_OVERRIDE=\"0 1\" bash $0" >&2
    exit 1
fi

LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_prediction_mutual_info_proxy}" \
SKEW_PROXY="prediction_mutual_info" \
bash scripts/experiments/lambda/run_proxy_temperature_entropy_diagnostics.sh "$@"
