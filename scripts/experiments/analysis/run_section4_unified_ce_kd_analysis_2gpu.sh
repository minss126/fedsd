#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export RUN_SET=server2
export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"
SECTION4_LOG_BASE="${SECTION4_LOG_BASE:-logs/analysis}"

TEMPERATURE=1.0 ENABLE_GRADIENT_ROUTES=1 \
    LOG_ROOT="${SECTION4_LOG_BASE}/logs_section4_unified_no_feature_t1p0_min64" \
    bash scripts/experiments/analysis/run_section4_unified_ce_kd_analysis.sh
TEMPERATURE=0.5 ENABLE_GRADIENT_ROUTES=0 \
    LOG_ROOT="${SECTION4_LOG_BASE}/logs_section4_unified_no_feature_t0p5_min64" \
    bash scripts/experiments/analysis/run_section4_unified_ce_kd_analysis.sh
