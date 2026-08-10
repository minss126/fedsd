#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export AXIS_OVERRIDE="local_epochs"
exec bash scripts/experiments/lambda/run_cifar100_protocol_axis.sh "$@"
