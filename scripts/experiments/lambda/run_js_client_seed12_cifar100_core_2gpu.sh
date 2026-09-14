#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export RUN_SET=server2
export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"
exec bash scripts/experiments/lambda/run_js_client_seed12_cifar100_core.sh
