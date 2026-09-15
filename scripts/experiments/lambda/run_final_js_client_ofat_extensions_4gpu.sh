#!/usr/bin/env bash

# Four-GPU share: dataset and protocol axes.  These are the heavier queues,
# especially TinyImageNet/ImageNet100-64 and E=10/participation=0.2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3}"
if (( ${#GPUS[@]} != 4 )); then
    echo "Provide exactly four GPU ids through GPUS_OVERRIDE." >&2
    exit 2
fi

GPUS_OVERRIDE="${GPUS[*]}" \
AXES_OVERRIDE="${AXES_OVERRIDE:-dataset local_epochs participation}" \
exec bash scripts/experiments/lambda/run_final_js_client_ofat_extensions.sh
