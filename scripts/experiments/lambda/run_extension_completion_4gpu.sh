#!/bin/bash

set -euo pipefail

# Completes the missing conditions for TinyImageNet, ImageNet100-64, and
# FedProx on a four-GPU server.  Phase 1 finishes the requested three-method
# table at beta={0.5,0.1}; phase 2 then adds IID and beta=0.3.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}"
export EXTENSIONS_OVERRIDE="tinyimagenet imagenet100_64 fedprox"
export ROUNDS="${ROUNDS:-500}"
export TINYIMAGENET_ROUNDS="${TINYIMAGENET_ROUNDS:-100}"
export IMAGENET100_ROUNDS="${IMAGENET100_ROUNDS:-100}"
export LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_extension_completion}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "========== Phase 1/2: Plain, beta={0.5,0.1} =========="
ENVS_OVERRIDE="beta_0.5 beta_0.1" \
METHODS_OVERRIDE="plain" \
bash scripts/experiments/lambda/run_soft_adaptive_extensions_4gpu.sh

echo "========== Phase 2/2: Plain/Fixed/Adaptive, IID and beta=0.3 =========="
ENVS_OVERRIDE="iid beta_0.3" \
METHODS_OVERRIDE="plain fixed_lambda1 soft_b_adaptive" \
bash scripts/experiments/lambda/run_soft_adaptive_extensions_4gpu.sh

echo "4-GPU extension completion pipeline complete."
