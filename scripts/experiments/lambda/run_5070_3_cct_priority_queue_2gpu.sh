#!/usr/bin/env bash

# Priority CCT model-extension queue for the 5070-3 two-GPU server.
#
# Phase 1: CCT on CIFAR-100 for 500 rounds.
# Phase 2: CCT on TinyImageNet and ImageNet100-64 for 300 rounds.
#
# Each phase uses seed 0, FedAvg, IID/beta=.1 and
# Plain/Fixed(lambda=.3)/Adaptive.  Adaptive warm-up is derived inside the
# underlying launcher as exactly half of that phase's communication rounds.
# Completed PKLs are skipped, so this wrapper is safe to rerun after an
# interruption (the interrupted in-flight cell itself restarts from round 0).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1}"
(( ${#GPUS[@]} == 2 )) || {
    echo "Exactly two GPU ids are required; got: ${GPUS[*]:-<empty>}" >&2
    exit 2
}

LOG_ROOT="${LOG_ROOT:-logs/lambda/final/logs_cct_model_extension_seed0_long_horizon}"
SMOKE_LOG_ROOT="${SMOKE_LOG_ROOT:-logs/lambda/smoke/logs_cct_model_extension_seed0}"
DRY_RUN="${DRY_RUN:-0}"

echo "========== 5070-3 CCT priority queue =========="
echo "GPUs=${GPUS[*]} | seed=0 | FedAvg | IID,beta=.1"
echo "methods=plain,fixed(.3),adaptive | C100 R500 | Tiny/Image R300"
echo "log_root=$LOG_ROOT"

echo "[Phase 1/2] CIFAR-100 R500"
GPUS_OVERRIDE="${GPUS[*]}" \
DATASETS_OVERRIDE="cifar100" \
PARTITIONS_OVERRIDE="iid beta_0.1" \
METHODS_OVERRIDE="plain fixed adaptive" \
SEEDS_OVERRIDE="0" \
LOG_ROOT="$LOG_ROOT" \
SMOKE_LOG_ROOT="$SMOKE_LOG_ROOT" \
RUN_SMOKE_FIRST="${RUN_SMOKE_FIRST:-1}" \
REUSE_SMOKE="${REUSE_SMOKE:-1}" \
RUN_FULL_AFTER_SMOKE=1 \
SKIP_EXISTING="${SKIP_EXISTING:-1}" \
DRY_RUN="$DRY_RUN" \
bash scripts/experiments/lambda/run_cct_model_extension_seed0_2gpu.sh

echo "[Phase 2/2] TinyImageNet + ImageNet100-64 R300"
GPUS_OVERRIDE="${GPUS[*]}" \
DATASETS_OVERRIDE="tinyimagenet imagenet100_64" \
PARTITIONS_OVERRIDE="iid beta_0.1" \
METHODS_OVERRIDE="plain fixed adaptive" \
SEEDS_OVERRIDE="0" \
ROUNDS_OVERRIDE=300 \
LOG_ROOT="$LOG_ROOT" \
SMOKE_LOG_ROOT="$SMOKE_LOG_ROOT" \
RUN_SMOKE_FIRST="${RUN_SMOKE_FIRST:-1}" \
REUSE_SMOKE=1 \
RUN_FULL_AFTER_SMOKE=1 \
SKIP_EXISTING="${SKIP_EXISTING:-1}" \
DRY_RUN="$DRY_RUN" \
bash scripts/experiments/lambda/run_cct_model_extension_seed0_2gpu.sh

echo "5070-3 CCT priority queue complete."
