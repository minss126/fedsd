#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Exact counterpart of the completed Final-CE-only diagnostic. The same
# centralized checkpoint, frozen probes, deterministic nested local subsets,
# fixed-step budget, and full-test measurements are reused. Only the local
# objective changes to the original BYOT blend:
#   Final CE + sum_b[(1-alpha) Branch CE_b + alpha KD_b] + beta sum_b Feature MSE_b
GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}" \
BUDGETS_OVERRIDE="${BUDGETS_OVERRIDE:-fixed_step}" \
LOCAL_OBJECTIVE="${LOCAL_OBJECTIVE:-blend}" \
SD_ALPHA="${SD_ALPHA:-0.15}" \
SD_BETA="${SD_BETA:-0.05}" \
SD_TEMPERATURE="${SD_TEMPERATURE:-0.5}" \
SD_BRANCH_REDUCTION="${SD_BRANCH_REDUCTION:-sum}" \
REFERENCE_ROOT="${REFERENCE_ROOT:-logs/analysis/logs_centralized_prefix_gradient_agreement}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/analysis/logs_centralized_prefix_gradient_agreement_byot}" \
    exec "${SCRIPT_DIR}/run_centralized_prefix_gradient_agreement.sh" "$@"
