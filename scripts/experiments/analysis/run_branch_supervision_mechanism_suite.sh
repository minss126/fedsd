#!/usr/bin/env bash

set -euo pipefail

# Run the three complementary suites in a reproducible order.  Target ablation
# answers CE-vs-KD; CE-mechanism ablation answers CE-vs-feature-only; paired
# gradients measure the shared-representation mechanism directly.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

./scripts/experiments/analysis/run_branch_target_ablation.sh
./scripts/experiments/analysis/run_branch_ce_mechanism_ablation.sh
./scripts/experiments/analysis/run_paired_branch_gradient_probe.sh
