#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# CIFAR-100 / FedAvg / severe non-IID (Dirichlet beta=0.1).
#
# Sweep A: alpha=1.0, vary BYOT feature-imitation beta.
# Sweep B: remove branch CE and vary the unrestricted branch KD coefficient.
#
# Existing reference shared by both sweeps:
#   logs/reliability/logs_fixed_alpha_high/beta_0.1/fedavg/fixed_alpha1p00.log
#   alpha=1.0, BYOT feature beta=0.01, full branches, sum reduction.
#
# We omit the duplicate reference run and execute 12 new jobs over four
# independent GPU queues (three jobs per GPU).

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
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
TEMP_VAL="${TEMP_VAL:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_byot_beta_kd_only_alpha}"
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
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1
    local method_name=$2
    local branch_objective=$3
    local alpha_val=$4
    local feature_beta=$5
    local log_dir="${LOG_ROOT}/beta_0.1/fedavg"
    local log_file="${log_dir}/${method_name}.log"

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${method_name}"
        return
    fi

    echo "[GPU ${gpu_id}] start: ${method_name} | objective=${branch_objective} | kd_alpha=${alpha_val} | feature_beta=${feature_beta}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr 0.1 --batch_size 64 \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --partition noniid --beta 0.1 \
        --logdir "${LOG_ROOT}" \
        --log_file_name "beta_0.1/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_active_branches "1,2,3" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective "${branch_objective}" \
        --byot_alpha "${alpha_val}" \
        --byot_beta "${feature_beta}" \
        --temperature "${TEMP_VAL}" \
        ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")

    for job in "${jobs[@]}"; do
        IFS='|' read -r method_name objective alpha_val feature_beta <<< "$job"
        run_job "$gpu_id" "$method_name" "$objective" "$alpha_val" "$feature_beta"
    done
}

echo "========== BYOT Feature-Beta and KD-Only Alpha Sweep =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "partition_beta=0.1, temperature=${TEMP_VAL}, log_root=${LOG_ROOT}"
echo "reference: alpha=1.0, feature_beta=0.01 from logs/reliability/logs_fixed_alpha_high"

run_queue "${GPUS[0]}" \
    "feature_beta0p000|blend|1.0|0.0" \
    "feature_beta0p100|blend|1.0|0.1" \
    "kd_only_alpha0p000|kd_only|0.0|0.01" &
queue0_pid=$!

run_queue "${GPUS[1]}" \
    "feature_beta0p001|blend|1.0|0.001" \
    "feature_beta0p500|blend|1.0|0.5" \
    "kd_only_alpha0p010|kd_only|0.01|0.01" &
queue1_pid=$!

run_queue "${GPUS[2]}" \
    "feature_beta0p050|blend|1.0|0.05" \
    "feature_beta1p000|blend|1.0|1.0" \
    "kd_only_alpha0p100|kd_only|0.1|0.01" &
queue2_pid=$!

run_queue "${GPUS[3]}" \
    "kd_only_alpha0p300|kd_only|0.3|0.01" \
    "kd_only_alpha3p000|kd_only|3.0|0.01" \
    "kd_only_alpha10p000|kd_only|10.0|0.01" &
queue3_pid=$!

wait "$queue0_pid" "$queue1_pid" "$queue2_pid" "$queue3_pid"
echo "BYOT feature-beta and KD-only alpha sweep complete (12 new runs)"
