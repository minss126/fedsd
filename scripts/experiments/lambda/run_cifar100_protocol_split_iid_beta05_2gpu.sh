#!/bin/bash

set -euo pipefail

# Two-GPU half of the CIFAR-100 protocol extension.  Intended for GPUs 2,3
# on the four-GPU host while GPUs 0,1 are occupied elsewhere.
# Runs local-epoch extension first, then participation-rate extension.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"2 3\" bash $0"
    echo "Partitions: IID, beta=0.5; phases: local epochs then participation."
    echo "Expected wall time on two GPUs: about 40-48 hours."
    exit 0
fi

export GPUS_OVERRIDE="${GPUS_OVERRIDE:-2 3}"
export ENVS_OVERRIDE="${ENVS_OVERRIDE:-iid beta_0.5}"
export LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_cifar100_protocol_extensions}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "========== Phase 1/2: Local Epochs | IID, beta=0.5 =========="
bash scripts/experiments/lambda/run_cifar100_local_epochs_extension.sh

echo "========== Phase 2/2: Participation | IID, beta=0.5 =========="
bash scripts/experiments/lambda/run_cifar100_participation_extension.sh

echo "Two-GPU protocol split (IID, beta=0.5) complete."
