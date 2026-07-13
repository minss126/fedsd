#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Beta 0.1 branch-depth control experiment.
#
# Runs:
#   1) Original sum reduction for the missing B1+B3 pair.
#   2) Mean-reduced branch losses for B1, B2, B3, B1+B2, B2+B3, and all branches.
#
# For a single active branch, sum and mean are identical. Therefore the B2
# result also completes the original-reduction single-branch comparison.
#
# Jobs are assigned to four GPU queues. Each queue runs sequentially, while the
# four queues run in parallel:
#   GPU 0: sum B1+B3 -> mean B2+B3
#   GPU 1: mean B1   -> mean all
#   GPU 2: mean B3   -> mean B2
#   GPU 3: mean B1+B2

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if [ "${#GPUS[@]}" -lt 4 ]; then
    echo "This script expects four GPUs. Set GPUS_OVERRIDE with four GPU ids." >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then
        PYTHON_BIN="venv/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

SEED="${SEED:-0}"
BYOT_ALPHA_VAL="${BYOT_ALPHA_VAL:-1.0}"
BYOT_BETA_VAL="${BYOT_BETA_VAL:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs/branch/logs_branch_mean_control}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round 499 result" "$log_file"
}

run_job() {
    local gpu_id=$1
    local method_name=$2
    local active_branches=$3
    local reduction=$4
    local log_dir="${LOG_ROOT}/beta_0.1/fedavg"
    local log_file="${log_dir}/${method_name}.log"

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] completed: ${method_name}"
        return
    fi

    echo "[GPU ${gpu_id}] start: ${method_name} | active=${active_branches} | reduction=${reduction}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --partition noniid --beta 0.1 \
        --logdir "${LOG_ROOT}" \
        --log_file_name "beta_0.1/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${BYOT_ALPHA_VAL}" \
        --byot_beta "${BYOT_BETA_VAL}" \
        --temperature "${TEMP_VAL}" \
        --byot_active_branches "${active_branches}" \
        --byot_branch_loss_reduction "${reduction}" \
        ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${method_name}"
}

echo "========== Branch Pair and Mean Control =========="
echo "gpus=${GPUS[*]}, seed=${SEED}, log_root=${LOG_ROOT}"
echo "alpha=${BYOT_ALPHA_VAL}, byot_beta=${BYOT_BETA_VAL}, temp=${TEMP_VAL}"

(
    run_job "${GPUS[0]}" "sum_active_b1_b3" "1,3" "sum"
    run_job "${GPUS[0]}" "mean_active_b2_b3" "2,3" "mean"
) &
queue0_pid=$!

(
    run_job "${GPUS[1]}" "mean_active_b1" "1" "mean"
    run_job "${GPUS[1]}" "mean_active_all_b1_b2_b3" "1,2,3" "mean"
) &
queue1_pid=$!

(
    run_job "${GPUS[2]}" "mean_active_b3" "3" "mean"
    run_job "${GPUS[2]}" "mean_active_b2" "2" "mean"
) &
queue2_pid=$!

(
    run_job "${GPUS[3]}" "mean_active_b1_b2" "1,2" "mean"
) &
queue3_pid=$!

wait "$queue0_pid" "$queue1_pid" "$queue2_pid" "$queue3_pid"
echo "Branch pair and mean control complete (7 jobs)"
