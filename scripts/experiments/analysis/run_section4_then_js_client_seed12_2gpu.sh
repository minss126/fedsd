#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"
bash scripts/experiments/analysis/run_section4_unified_ce_kd_analysis_2gpu.sh
bash scripts/experiments/lambda/run_js_client_seed12_cifar100_core_2gpu.sh
