#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# FMNIST 500-round alpha sweep for a fairer comparison with CIFAR/TinyImageNet
# last-30 metrics.
#
# Runs:
#   partitions: iid, beta_0.5, beta_0.3, beta_0.1
#   alpha:      0.0, 0.3, 0.7, 1.0
#   total:      16 jobs
#
# Output:
#   logs/alpha/logs_fmnist_alpha_r500/fmnist_resnet18/<partition>/fedavg/

GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}" \
SETTINGS_OVERRIDE="fmnist_resnet18" \
PARTITIONS_OVERRIDE="${PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1}" \
ALPHAS_OVERRIDE="${ALPHAS_OVERRIDE:-0.00:0p00 0.30:0p30 0.70:0p70 1.00:1p00}" \
SMALL_ROUNDS="${SMALL_ROUNDS:-500}" \
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_fmnist_alpha_r500}" \
USE_WANDB="${USE_WANDB:-1}" \
bash scripts/experiments/alpha/run_simple_dataset_alpha_screen.sh
