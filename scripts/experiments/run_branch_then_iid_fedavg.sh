#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Overnight queue:
#   1) Branch-wise alpha sweep on non-IID beta 0.1 / 0.3 / 0.5
#   2) IID FedAvg comparison set
#
# Usage:
#   bash scripts/experiments/run_branch_then_iid_fedavg.sh
#
# Useful overrides:
#   USE_WANDB=0 bash scripts/experiments/run_branch_then_iid_fedavg.sh
#   GPUS_OVERRIDE="0 1 2 3" bash scripts/experiments/run_branch_then_iid_fedavg.sh

echo "========== Step 1/2: Branch-wise alpha sweep: beta_0.1 beta_0.3 beta_0.5 =========="
ENVS_OVERRIDE="beta_0.1 beta_0.3 beta_0.5" \
BASE_ALGOS="${BASE_ALGOS_BRANCH:-fedavg}" \
LOG_ROOT="${BRANCH_LOG_ROOT:-logs/branch/logs_branch_alpha_sweep}" \
bash scripts/experiments/branch/run_branch_alpha_sweep.sh

echo "========== Step 2/2: IID FedAvg compare =========="
LOG_ROOT="${IID_LOG_ROOT:-logs/reliability/logs_iid_fedavg_compare}" \
bash scripts/experiments/reliability/run_iid_fedavg_compare.sh

echo "Branch-wise then IID FedAvg queue complete."
