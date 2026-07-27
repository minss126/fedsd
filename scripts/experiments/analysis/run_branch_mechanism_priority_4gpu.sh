#!/usr/bin/env bash

set -euo pipefail

# Priority half of the complete evidence package.  Run this concurrently with
# run_branch_mechanism_support_2gpu.sh only when the two GPU sets are disjoint.
# It intentionally contains no post-processing/analysis: it only creates the
# checkpoints and metrics required for later analysis.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS="${GPUS_OVERRIDE:-0 1 2 3}"
export USE_WANDB="${USE_WANDB:-0}"

echo "========== Priority Branch-Mechanism Experiments (4 GPU) =========="
echo "gpus=${GPU_IDS}"
echo "Order: C100 core controls -> C100 detach controls -> C10 controls -> frozen shallow capacity/Off"
echo "No metric post-processing is run here. SKIP_EXISTING=1 makes this resumable."

echo "[1/2] Supervision and shared-trunk causal controls with layer-wise update metrics"
GPUS_OVERRIDE="$GPU_IDS" \
    ./scripts/experiments/analysis/run_branch_supervision_causal_suite.sh

echo "[2/2] Frozen shallow-representation probe from completed teacher-only checkpoints"
GPUS_OVERRIDE="$GPU_IDS" \
    ./scripts/experiments/analysis/run_frozen_shallow_representation_probe.sh

echo "Priority branch-mechanism experiments finished."
