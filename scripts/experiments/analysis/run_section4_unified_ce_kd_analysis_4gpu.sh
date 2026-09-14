#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export RUN_SET=server4
export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}"
SECTION4_LOG_BASE="${SECTION4_LOG_BASE:-logs/analysis}"

# T=1 supplies the requested gradient analysis. T=0.5 supplies the empirical
# CE/KD accuracy and rare/frequent comparison without duplicating gradients.
TEMPERATURE=1.0 ENABLE_GRADIENT_ROUTES=1 \
    LOG_ROOT="${SECTION4_LOG_BASE}/logs_section4_unified_no_feature_t1p0_min64" \
    bash scripts/experiments/analysis/run_section4_unified_ce_kd_analysis.sh
TEMPERATURE=0.5 ENABLE_GRADIENT_ROUTES=0 \
    LOG_ROOT="${SECTION4_LOG_BASE}/logs_section4_unified_no_feature_t0p5_min64" \
    bash scripts/experiments/analysis/run_section4_unified_ce_kd_analysis.sh
