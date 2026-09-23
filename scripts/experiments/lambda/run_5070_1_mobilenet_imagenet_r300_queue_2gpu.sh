#!/usr/bin/env bash

# Queue for the 5070-1 two-GPU server.
#
# 1. Wait until the currently running MobileNetV2/TinyImageNet R300 matrix
#    has been idle for three consecutive checks.
# 2. Run the matching MobileNetV2/ImageNet100-64 R300 matrix.
#
# The invoked launcher keeps the finalized model-extension protocol:
# seed 0, FedAvg, IID/beta=.1, Plain/Fixed(.3)/Adaptive, E=5, C=.1,
# batch=64, KD-only, no feature loss, T_KD=T_proxy=1, and canonical RNG.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1}"
(( ${#GPUS[@]} == 2 )) || {
    echo "Exactly two GPU ids are required; got: ${GPUS[*]:-<empty>}" >&2
    exit 2
}

WAIT_FOR_TINY="${WAIT_FOR_TINY:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
WAIT_IDLE_POLLS="${WAIT_IDLE_POLLS:-3}"
DRY_RUN="${DRY_RUN:-0}"

for value_name in WAIT_FOR_TINY DRY_RUN; do
    value="${!value_name}"
    [[ "$value" == 0 || "$value" == 1 ]] || {
        echo "$value_name must be 0 or 1; got $value." >&2
        exit 2
    }
done
[[ "$WAIT_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
    echo "WAIT_POLL_SECONDS must be a positive integer." >&2; exit 2;
}
[[ "$WAIT_IDLE_POLLS" =~ ^[1-9][0-9]*$ ]] || {
    echo "WAIT_IDLE_POLLS must be a positive integer." >&2; exit 2;
}

TINY_PROCESS_PATTERN='[m]ain.py.*logs/lambda/final/logs_mobilenetv2_tiny_r300_canonical_seed0'

if [[ "$WAIT_FOR_TINY" == 1 && "$DRY_RUN" != 1 ]]; then
    echo "Waiting for the current MobileNetV2/TinyImageNet R300 queue to finish."
    idle_polls=0
    while (( idle_polls < WAIT_IDLE_POLLS )); do
        if pgrep -f "$TINY_PROCESS_PATTERN" >/dev/null; then
            idle_polls=0
        else
            idle_polls=$((idle_polls + 1))
            echo "Tiny queue idle check: ${idle_polls}/${WAIT_IDLE_POLLS}"
        fi
        (( idle_polls >= WAIT_IDLE_POLLS )) || sleep "$WAIT_POLL_SECONDS"
    done
fi

echo "Starting MobileNetV2/ImageNet100-64 R300 on GPUs: ${GPUS[*]}"
GPUS_OVERRIDE="${GPUS[*]}" \
DATASETS_OVERRIDE="imagenet100_64" \
PARTITIONS_OVERRIDE="iid beta_0.1" \
METHODS_OVERRIDE="plain fixed adaptive" \
SEEDS_OVERRIDE="0" \
ROUNDS_OVERRIDE=300 \
LOG_ROOT="${LOG_ROOT:-logs/lambda/final/logs_mobilenetv2_imagenet100_64_r300_canonical_seed0}" \
SKIP_EXISTING="${SKIP_EXISTING:-1}" \
DRY_RUN="$DRY_RUN" \
bash scripts/experiments/lambda/run_mobilenetv2_tiny_r300_canonical_seed0_2gpu.sh

echo "5070-1 priority queue complete."
