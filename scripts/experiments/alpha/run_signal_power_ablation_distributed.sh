#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Launch the distributed signal/power ablation allocation with one command on
# each server. The child runner waits for every assigned job before advancing
# to the next block, so no manual follow-up command is needed.
#
# Run on the 4-GPU server:
#   ROLE=server4 bash scripts/experiments/alpha/run_signal_power_ablation_distributed.sh
#
# Run on the 2-GPU server (at the same time):
#   ROLE=server2 bash scripts/experiments/alpha/run_signal_power_ablation_distributed.sh
#
# Assignment:
#   server4: all lambda_max=3 jobs (32), then 12 lambda_max=1 jobs
#   server2: remaining 20 lambda_max=1 jobs

ROLE="${ROLE:-}"
RUNNER="scripts/experiments/alpha/run_signal_power_ablation.sh"
LOG_ROOT_MAX3="${LOG_ROOT_MAX3:-logs/alpha/logs_signal_power_ablation_max3}"
LOG_ROOT_MAX1="${LOG_ROOT_MAX1:-logs/alpha/logs_signal_power_ablation_max1}"

ALL_METHODS="label_prob_p1 label_prob_p2 teacher_certainty_p1 teacher_certainty_p2 label_prob_x_certainty_p1 label_prob_x_certainty_p2 label_prob_x_client_pred_entropy_p1 label_prob_x_client_pred_entropy_p2"
SERVER4_MAX1_METHODS="label_prob_p1 label_prob_p2 teacher_certainty_p1"
SERVER2_MAX1_METHODS="teacher_certainty_p2 label_prob_x_certainty_p1 label_prob_x_certainty_p2 label_prob_x_client_pred_entropy_p1 label_prob_x_client_pred_entropy_p2"

run_block() {
    local lambda_max=$1
    local log_root=$2
    local methods=$3

    LAMBDA_MAX="$lambda_max" \
    LOG_ROOT="$log_root" \
    METHODS_OVERRIDE="$methods" \
    GPUS_OVERRIDE="$GPUS_OVERRIDE" \
    bash "$RUNNER"
}

case "$ROLE" in
    server4)
        GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1 2 3}"
        echo "[server4] starting lambda_max=3 signal/power ablation"
        run_block 3.00 "$LOG_ROOT_MAX3" "$ALL_METHODS"
        echo "[server4] starting assigned lambda_max=1 jobs"
        run_block 1.00 "$LOG_ROOT_MAX1" "$SERVER4_MAX1_METHODS"
        ;;
    server2)
        GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"
        echo "[server2] starting assigned lambda_max=1 jobs"
        run_block 1.00 "$LOG_ROOT_MAX1" "$SERVER2_MAX1_METHODS"
        ;;
    *)
        echo "Set ROLE=server4 or ROLE=server2." >&2
        exit 1
        ;;
esac

echo "Distributed signal/power ablation block complete for ${ROLE}."
