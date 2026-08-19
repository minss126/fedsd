#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Same centralized checkpoint/local subsets/frozen probes as the CE-only and
# simple-BYOT diagnostics. This wrapper applies the final selected soft-b
# client-adaptive KD-only objective. Since a centralized checkpoint has no FL
# round index, the default resolves the round schedule at its post-warm-up
# value (round scale 1.0); override ADAPTIVE_ROUND_SCALE for an ablation.
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}" \
BUDGETS_OVERRIDE="${BUDGETS_OVERRIDE:-fixed_step}" \
LOCAL_OBJECTIVE="adaptive_kd" \
SD_ALPHA="0.0" \
SD_BETA="${SD_BETA:-0.01}" \
SD_TEMPERATURE="${SD_TEMPERATURE:-1.0}" \
SD_BRANCH_REDUCTION="sum" \
ADAPTIVE_LAMBDA_MAX="${ADAPTIVE_LAMBDA_MAX:-1.0}" \
ADAPTIVE_ROUND_SCALE="${ADAPTIVE_ROUND_SCALE:-1.0}" \
ADAPTIVE_PROXY_TEMPERATURE="${ADAPTIVE_PROXY_TEMPERATURE:-1.0}" \
ADAPTIVE_RELIABILITY_POWER="${ADAPTIVE_RELIABILITY_POWER:-1.0}" \
ADAPTIVE_SKEW_POWER="${ADAPTIVE_SKEW_POWER:-2.0}" \
ADAPTIVE_SOFT_TAU="${ADAPTIVE_SOFT_TAU:-0.85}" \
ADAPTIVE_SOFT_TEMPERATURE="${ADAPTIVE_SOFT_TEMPERATURE:-0.05}" \
REFERENCE_ROOT="${REFERENCE_ROOT:-logs/analysis/logs_centralized_prefix_gradient_agreement}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/analysis/logs_centralized_prefix_gradient_agreement_adaptive}" \
    exec "${SCRIPT_DIR}/run_centralized_prefix_gradient_agreement.sh" "$@"
