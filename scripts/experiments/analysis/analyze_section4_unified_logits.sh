#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x venv/bin/python ]; then PYTHON_BIN=venv/bin/python; else PYTHON_BIN=python3; fi
fi

TEMPERATURES=(${TEMPERATURES_OVERRIDE:-1.0 0.5})
SECTION4_LOG_BASE="${SECTION4_LOG_BASE:-logs/analysis}"
for temperature in "${TEMPERATURES[@]}"; do
    temperature_tag="$(printf '%s' "$temperature" | tr '.' 'p')"
    log_root="${SECTION4_LOG_BASE}/logs_section4_unified_no_feature_t${temperature_tag}_min64"
    output_prefix="${SECTION4_LOG_BASE}/section4_unified_t${temperature_tag}_rare_frequent"
    "$PYTHON_BIN" scripts/experiments/analysis/analyze_full_branch_logits.py \
        --log-root "$log_root" \
        --datasets cifar100 \
        --partitions iid,beta_0.3,beta_0.1 \
        --alphas 0p00,1p00 \
        --seeds "${SEED_OVERRIDE:-0}" \
        --rounds 470,480,490 \
        --kd-temperature "$temperature" \
        --output-prefix "$output_prefix"
done
