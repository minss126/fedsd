#!/bin/bash

set -euo pipefail

# Full re-run for the actual client-model instability question.  In addition
# to pre-local teacher frequency statistics, this evaluates post-local client
# models on the same labeled reference samples before aggregation and logs
# probability-vector JS/L2 divergence by each client's local frequency group.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_postlocal_branch_distribution_r500}"
export ENABLE_POSTLOCAL_BRANCH_DISTRIBUTION=1
export SAVE_POSTLOCAL_FULL_LOGITS=1
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "Running post-local common-reference branch distribution probe"
echo "log_root=${LOG_ROOT}"

exec bash scripts/experiments/analysis/run_client_pretrain_branch_frequency_probe.sh
