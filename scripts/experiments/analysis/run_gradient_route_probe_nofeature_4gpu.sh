#!/usr/bin/env bash

# CE-vs-KD gradient-route analysis under the final no-feature protocol.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}"
export DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-cifar10 cifar100}"
export PARTITIONS_OVERRIDE="${PARTITIONS_OVERRIDE:-iid beta_0.1}"
export VARIANTS_OVERRIDE="${VARIANTS_OVERRIDE:-ce_only kd_only}"
export FEATURE_BETA=0.0
export TEMPERATURE="${TEMPERATURE:-1.0}"
export LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_gradient_route_probe_no_feature_t1_r500}"

exec bash scripts/experiments/analysis/run_gradient_route_probe_4gpu.sh
