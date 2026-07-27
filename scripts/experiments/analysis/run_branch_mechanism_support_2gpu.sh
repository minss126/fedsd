#!/usr/bin/env bash

set -euo pipefail

# Supporting half of the evidence package.  The 3-seed nested class-count
# study is deliberately sized to finish at roughly the same time as the
# 4-GPU priority suite when launched on a disjoint two-GPU set.
# This launcher only produces experiment logs/checkpoints; it does not analyse
# them.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS="${GPUS_OVERRIDE:-4 5}"
export USE_WANDB="${USE_WANDB:-0}"

echo "========== Supporting Branch-Mechanism Experiments (2 GPU) =========="
echo "gpus=${GPU_IDS}"
echo "Study: C100 nested K={10,20,50,100}, alpha={0,1}, seed={0,1,2}"
echo "No metric post-processing is run here. SKIP_EXISTING=1 makes this resumable."

GPUS_OVERRIDE="$GPU_IDS" SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-0 1 2}" \
    ./scripts/experiments/analysis/run_cifar100_class_count_alpha_control.sh

echo "Supporting branch-mechanism experiments finished."
