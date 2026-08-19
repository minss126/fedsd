#!/usr/bin/env bash

# Two-GPU half of the long OFAT completion queue.
# It owns CIFAR-100's missing protocol/FedAvgM cells and TinyImageNet's
# protocol/model cells.  This balances the longer ImageNet queue on 4 GPUs.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"
if (( $(wc -w <<< "$GPUS_OVERRIDE") != 2 )); then
    echo "Provide exactly two GPU ids through GPUS_OVERRIDE." >&2
    exit 1
fi

echo "========== Phase 1/2: CIFAR-100 missing cells =========="
GPUS_OVERRIDE="$GPUS_OVERRIDE" \
DATASETS_OVERRIDE="cifar100" \
AXES_OVERRIDE="protocol mechanism" \
bash scripts/experiments/lambda/run_ofat_matrix_no_resnet50.sh

echo "========== Phase 2/2: TinyImageNet protocol and model =========="
GPUS_OVERRIDE="$GPUS_OVERRIDE" \
DATASETS_OVERRIDE="tinyimagenet" \
AXES_OVERRIDE="protocol model" \
bash scripts/experiments/lambda/run_ofat_matrix_no_resnet50.sh
