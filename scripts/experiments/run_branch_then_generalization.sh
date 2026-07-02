#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "========== Stage 1/2: Branch Pair and Mean Control =========="
bash scripts/experiments/branch/run_branch_pair_and_mean_control.sh

echo "========== Stage 2/2: Dataset/Model Generalization =========="
bash scripts/experiments/generalization/run_dataset_model_generalization.sh

echo "All queued experiments completed."
