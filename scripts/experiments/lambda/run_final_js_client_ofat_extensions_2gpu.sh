#!/usr/bin/env bash

# Two-GPU share: CIFAR-100 MobileNetV2 and mechanism axes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1}"
if (( ${#GPUS[@]} != 2 )); then
    echo "Provide exactly two GPU ids through GPUS_OVERRIDE." >&2
    exit 2
fi

GPUS_OVERRIDE="${GPUS[*]}" \
AXES_OVERRIDE="${AXES_OVERRIDE:-model fedprox moon}" \
exec bash scripts/experiments/lambda/run_final_js_client_ofat_extensions.sh
