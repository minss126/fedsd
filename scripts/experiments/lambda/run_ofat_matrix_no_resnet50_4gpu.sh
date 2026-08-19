#!/usr/bin/env bash

# Four-GPU half of the long OFAT completion queue.
# It owns all ImageNet100-64 cells and TinyImageNet's mechanism axis.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}"
if (( $(wc -w <<< "$GPUS_OVERRIDE") != 4 )); then
    echo "Provide exactly four GPU ids through GPUS_OVERRIDE." >&2
    exit 1
fi

echo "========== Phase 1/2: ImageNet100-64 (all axes) =========="
GPUS_OVERRIDE="$GPUS_OVERRIDE" \
DATASETS_OVERRIDE="imagenet100_64" \
AXES_OVERRIDE="protocol model mechanism" \
bash scripts/experiments/lambda/run_ofat_matrix_no_resnet50.sh

echo "========== Phase 2/2: TinyImageNet mechanisms =========="
GPUS_OVERRIDE="$GPUS_OVERRIDE" \
DATASETS_OVERRIDE="tinyimagenet" \
AXES_OVERRIDE="mechanism" \
bash scripts/experiments/lambda/run_ofat_matrix_no_resnet50.sh
