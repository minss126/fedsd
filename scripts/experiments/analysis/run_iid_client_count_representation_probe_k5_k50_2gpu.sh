#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Additional points for the CIFAR-100 IID client-count diagnostic.
# The two jobs are scheduled concurrently, one per GPU:
#   K=5  -> 10,000 training samples/client
#   K=50 ->  1,000 training samples/client
# Only seed 0 is run. All other settings are inherited from the original
# K={1,20,100} diagnostic for a directly comparable result.

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"
export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-0}"
export CLIENT_COUNTS_OVERRIDE="${CLIENT_COUNTS_OVERRIDE:-5 50}"
export USE_WANDB="${USE_WANDB:-0}"

exec bash scripts/experiments/analysis/run_iid_client_count_representation_probe.sh
