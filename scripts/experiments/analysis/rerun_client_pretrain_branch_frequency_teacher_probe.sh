#!/bin/bash

set -euo pipefail

# Re-run the client-wise pre-local branch-frequency probe with teacher group
# statistics (teacher entropy, true-label probability, confidence, and
# accuracy) enabled.  A separate root preserves the existing branch-only run.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_client_pretrain_branch_frequency_teacher_r500}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "Running client pretrain branch-frequency probe with teacher statistics"
echo "log_root=${LOG_ROOT}"

exec bash scripts/experiments/analysis/run_client_pretrain_branch_frequency_probe.sh
