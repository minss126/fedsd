#!/usr/bin/env bash

set -euo pipefail

# One-command, resumable execution of every experiment needed to turn the
# C10/C100 alpha explanation from an inference into measured evidence.
# Stages are intentionally sequential: each consumes all selected GPUs, and a
# later stage starts only when the previous one exits successfully.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS="${GPUS_OVERRIDE:-0 1 2 3}"
export USE_WANDB="${USE_WANDB:-0}"

echo "========== Complete Branch-Mechanism Suite (4 GPU) =========="
echo "gpus=${GPU_IDS}"
echo "Stages: existing full-logit analysis -> frozen shallow probe -> causal controls -> class-count control"
echo "All stage scripts use SKIP_EXISTING=1 by default, so rerunning this command resumes safely."

echo "[1/4] Existing full-logit relation / calibration analysis"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-cpu}" \
    ./scripts/experiments/analysis/run_full_branch_logit_relation_analysis.sh

echo "[2/4] Frozen shallow-representation probe and Off checkpoints"
GPUS_OVERRIDE="$GPU_IDS" \
    ./scripts/experiments/analysis/run_frozen_shallow_representation_probe.sh

echo "[3/4] Feature-only, CE/KD, and stop-gradient causal controls with layer-wise update drift"
GPUS_OVERRIDE="$GPU_IDS" \
    ./scripts/experiments/analysis/run_branch_supervision_causal_suite.sh

echo "[4/4] CIFAR-100 nested class-count alpha control"
GPUS_OVERRIDE="$GPU_IDS" \
    ./scripts/experiments/analysis/run_cifar100_class_count_alpha_control.sh

echo "Complete branch-mechanism suite finished."

