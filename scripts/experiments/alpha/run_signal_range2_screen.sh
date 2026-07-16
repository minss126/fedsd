#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Quick lambda-range screen for adaptive KD-only BYOT.
#
# Goal:
#   Check whether lambda_max=2 is a better range than lambda_max=1 or 3.
#
# Rationale:
#   fixed lambda=1 has often been strong. lambda_max=1 tends to only reduce
#   below that point, while lambda_max=3 can over-shoot. lambda_max=2 lets
#   reliable clients exceed 1, while still reducing overly strong KD pressure.
#
# Default plan:
#   partition: noniid_grouping
#   methods:
#     label_prob_x_certainty_p1
#     label_prob_x_client_pred_entropy_p1
#     label_prob_x_client_pred_entropy_p2
#   total: 1 * 3 = 3 runs
#
# Set ENVS_OVERRIDE="iid beta_0.5 beta_0.3 beta_0.1 noniid_grouping" if
# you want to rerun the full range comparison with group-lambda logging.
#
# This wraps run_signal_power_ablation.sh so the implementation stays identical
# to the previous lambda_max=1/3 experiments except for the selected range.
#
# Usage:
#   USE_WANDB=1 bash scripts/experiments/alpha/run_signal_range2_screen.sh

GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}" \
LAMBDA_MAX="${LAMBDA_MAX:-2.00}" \
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_signal_power_ablation_max2_screen}" \
METHODS_OVERRIDE="${METHODS_OVERRIDE:-label_prob_x_certainty_p1 label_prob_x_client_pred_entropy_p1 label_prob_x_client_pred_entropy_p2}" \
ENVS_OVERRIDE="${ENVS_OVERRIDE:-noniid_grouping}" \
EXTRA_FLAGS="${EXTRA_FLAGS:---log_client_group_lambda}" \
USE_WANDB="${USE_WANDB:-1}" \
bash scripts/experiments/alpha/run_signal_power_ablation.sh
