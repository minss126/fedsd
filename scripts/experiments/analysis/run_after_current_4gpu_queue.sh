#!/usr/bin/env bash

set -euo pipefail

# Lightweight reservation wrapper: wait only for explicitly supplied current
# training PIDs, then start the largest pending suite on all requested GPUs.
# It does not reserve GPUs or wait for unrelated users' jobs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

WAIT_PIDS="${WAIT_PIDS:-77398 77399 77401 77402}"
POLL_SECONDS="${POLL_SECONDS:-60}"

echo "========== 4-GPU queued CE mechanism suite =========="
echo "Waiting for PIDs: ${WAIT_PIDS}"

while :; do
    alive=()
    for pid in ${WAIT_PIDS}; do
        if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
        fi
    done
    if [ ${#alive[@]} -eq 0 ]; then
        break
    fi
    echo "[$(date '+%F %T')] waiting for: ${alive[*]}"
    sleep "$POLL_SECONDS"
done

echo "[$(date '+%F %T')] current 4-GPU jobs finished; starting CE mechanism ablation."
exec ./scripts/experiments/analysis/run_branch_ce_mechanism_ablation.sh
