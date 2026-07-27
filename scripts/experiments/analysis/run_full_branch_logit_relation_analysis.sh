#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then
        PYTHON_BIN="venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_postlocal_branch_distribution_full_logits_r500}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-logs/analysis/full_branch_logit_relation}"
DEVICE="${ANALYSIS_DEVICE:-cpu}"

"$PYTHON_BIN" scripts/experiments/analysis/analyze_full_branch_logits.py \
    --log-root "$LOG_ROOT" \
    --datasets "${DATASETS:-cifar10,cifar100}" \
    --partitions "${PARTITIONS:-iid,beta_0.5,beta_0.1}" \
    --alphas "${ALPHAS:-0p00,1p00}" \
    --seeds "${SEEDS_OVERRIDE:-0,1}" \
    --rounds "${ROUNDS_OVERRIDE:-470,480,490}" \
    --device "$DEVICE" \
    --output-prefix "$OUTPUT_PREFIX"

