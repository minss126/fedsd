#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Plain FedAvg baseline for FMNIST / ResNet18 at 500 rounds.
#
# This matches the FMNIST alpha-r500 sweep:
#   logs/alpha/logs_fmnist_alpha_r500/fmnist_resnet18/<partition>/fedavg/
#
# Output:
#   logs/baseline/logs_plain_fmnist_r500/fmnist_resnet18/<partition>/fedavg/plain_fedavg.log

GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}" \
SETTINGS_OVERRIDE="fmnist_resnet18" \
PARTITIONS_OVERRIDE="${PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1}" \
FMNIST_ROUNDS="${FMNIST_ROUNDS:-500}" \
LOG_ROOT="${LOG_ROOT:-logs/baseline/logs_plain_fmnist_r500}" \
USE_WANDB="${USE_WANDB:-1}" \
bash scripts/experiments/baseline/run_plain_dataset_generalization.sh
