#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Use a fresh root so every target/temperature/scale condition is trained by
# the same corrected source.  The underlying launcher validates completed
# pickles, so this command is safe to resume after an interruption.
export LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_cifar100_kd_factorial_clean_r500}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

exec bash scripts/experiments/analysis/run_cifar100_kd_factorial_ablation.sh
