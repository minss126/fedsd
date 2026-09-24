#!/usr/bin/env bash

# Completion queue for the 5070-1 two-GPU server.
#
# Phase 1: MobileNetV2 fixed-lambda R100 on TinyImageNet/ImageNet100-64.
# Phase 2: corrected matched-tokenizer CCT R100 on ImageNet100-64.
#
# The currently running R300 queue must be stopped before launching this file.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1}"
(( ${#GPUS[@]} == 2 )) || {
    echo "Exactly two GPU ids are required; got: ${GPUS[*]:-<empty>}" >&2
    exit 2
}

DRY_RUN="${DRY_RUN:-0}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || {
    echo "DRY_RUN must be 0 or 1; got $DRY_RUN." >&2
    exit 2
}

MOBILENET_LOG_ROOT="${MOBILENET_LOG_ROOT:-logs/lambda/final/logs_mobilenetv2_fixed_r100_canonical_seed0}"
CCT_LOG_ROOT="${CCT_LOG_ROOT:-logs/lambda/final/logs_cct_model_extension_seed0_matched_tokenizer_r100}"
CCT_SMOKE_LOG_ROOT="${CCT_SMOKE_LOG_ROOT:-logs/lambda/smoke/logs_cct_model_extension_seed0_matched_tokenizer}"

echo "========== 5070-1 R100 completion queue =========="
echo "GPUs=${GPUS[*]}"
echo "[Phase 1/2] MobileNetV2 Fixed(lambda=.3), Tiny/Image R100"
GPUS_OVERRIDE="${GPUS[*]}" \
DATASETS_OVERRIDE="tinyimagenet imagenet100_64" \
PARTITIONS_OVERRIDE="iid beta_0.1" \
METHODS_OVERRIDE="fixed" \
SEEDS_OVERRIDE="0" \
ROUNDS_OVERRIDE=100 \
LOG_ROOT="$MOBILENET_LOG_ROOT" \
SKIP_EXISTING="${SKIP_EXISTING:-1}" \
DRY_RUN="$DRY_RUN" \
bash scripts/experiments/lambda/run_mobilenetv2_tiny_r300_canonical_seed0_2gpu.sh

echo "[Phase 2/2] Corrected CCT, ImageNet100-64 R100"
GPUS_OVERRIDE="${GPUS[*]}" \
DATASETS_OVERRIDE="imagenet100_64" \
PARTITIONS_OVERRIDE="iid beta_0.1" \
METHODS_OVERRIDE="plain fixed adaptive" \
SEEDS_OVERRIDE="0" \
ROUNDS_OVERRIDE=100 \
LOG_ROOT="$CCT_LOG_ROOT" \
SMOKE_LOG_ROOT="$CCT_SMOKE_LOG_ROOT" \
RUN_SMOKE_FIRST="${RUN_SMOKE_FIRST:-1}" \
REUSE_SMOKE="${REUSE_SMOKE:-0}" \
RUN_FULL_AFTER_SMOKE=1 \
SKIP_EXISTING="${SKIP_EXISTING:-1}" \
DRY_RUN="$DRY_RUN" \
bash scripts/experiments/lambda/run_cct_model_extension_seed0_2gpu.sh

echo "5070-1 R100 completion queue complete."
