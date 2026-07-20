#!/bin/bash

set -euo pipefail

# Re-run the branch-frequency analysis after enabling its per-batch statistic
# collection in the standard `fedbyot` training path.  Results are written to
# a new directory by default so the previously empty statistics are preserved
# for auditability.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_train_branch_frequency_stats_rerun}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "Re-running branch-frequency statistics into: ${LOG_ROOT}"
echo "Override GPUs with GPUS_OVERRIDE, e.g. GPUS_OVERRIDE='0 1 2 3'."

exec bash scripts/experiments/analysis/run_train_branch_frequency_stats.sh
