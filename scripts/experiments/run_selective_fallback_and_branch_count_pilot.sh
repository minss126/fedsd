#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Fast queue:
#   1) Selective KD with CE fallback on beta_0.1/FedAvg
#   2) Active branch-count pilot on beta_0.1/FedAvg

echo "========== Step 1/2: Selective KD with CE fallback pilot =========="
bash scripts/experiments/selective/run_selective_ce_fallback_pilot.sh

echo "========== Step 2/2: Branch count pilot =========="
bash scripts/experiments/branch/run_branch_count_pilot.sh

echo "Selective fallback + branch count pilot complete."
