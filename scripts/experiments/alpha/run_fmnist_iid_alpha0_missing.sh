#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Re-run the missing FMNIST IID alpha=0 result from the simple dataset
# alpha screen. This writes to the same log location as the original sweep:
#   logs/alpha/logs_simple_dataset_alpha_screen/fmnist_resnet18/iid/fedavg/

GPUS_OVERRIDE="${GPUS_OVERRIDE:-1}" \
SETTINGS_OVERRIDE="fmnist_resnet18" \
PARTITIONS_OVERRIDE="iid" \
ALPHAS_OVERRIDE="0.00:0p00" \
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_simple_dataset_alpha_screen}" \
USE_WANDB="${USE_WANDB:-1}" \
bash scripts/experiments/alpha/run_simple_dataset_alpha_screen.sh
