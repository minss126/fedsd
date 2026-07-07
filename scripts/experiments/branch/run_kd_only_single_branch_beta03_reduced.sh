#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Reduced beta=0.3 KD-only single-branch sweep.
#
# Purpose:
#   Check the missing beta=0.3 transition point with a compact matrix.
#
# Matrix:
#   dataset/backbone: CIFAR-100 / ResNet18-BYOT
#   partition: beta=0.3
#   active branch: B1, B2, B3
#   lambda_kd: 0.00, 3.00
#
# Total default jobs: 3 branches x 2 lambdas = 6.
# With 2 GPUs, this is 3 sequential jobs per GPU.

GPUS=(${GPUS_OVERRIDE:-0 1})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPU ids provided. Set GPUS_OVERRIDE." >&2
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
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CIFAR_LR="${CIFAR_LR:-0.1}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LOG_ROOT="${LOG_ROOT:-logs_kd_only_single_branch_generalization}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

BRANCHES=(${BRANCHES_OVERRIDE:-1:B1 2:B2 3:B3})
LAMBDAS=(${LAMBDAS_OVERRIDE:-0.00:0p000 3.00:3p000})

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
    local branch_id=$2
    local branch_tag=$3
    local lambda_val=$4
    local lambda_tag=$5

    local setting="cifar100_resnet18"
    local partition_name="beta_0.3"
    local method_name="kd_only_${branch_tag}_lambda${lambda_tag}"
    local log_dir="${LOG_ROOT}/${setting}/${partition_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${setting} | ${partition_name} | ${method_name}"
        return
    fi

    echo "[GPU ${gpu_id}] start: ${setting} | ${partition_name} | ${method_name} | active=${branch_id}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${CIFAR_LR}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${setting}/${partition_name}/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --partition noniid --beta 0.3 \
        --kd_conf_threshold 0.0 \
        --byot_active_branches "${branch_id}" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only \
        --byot_alpha "${lambda_val}" \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${setting} | ${partition_name} | ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")

    for job in "${jobs[@]}"; do
        [ -z "$job" ] && continue
        IFS='|' read -r branch_id branch_tag lambda_val lambda_tag <<< "$job"
        run_job "$gpu_id" "$branch_id" "$branch_tag" "$lambda_val" "$lambda_tag"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for branch_pair in "${BRANCHES[@]}"; do
    branch_id="${branch_pair%%:*}"
    branch_tag="${branch_pair##*:}"
    for lambda_pair in "${LAMBDAS[@]}"; do
        lambda_val="${lambda_pair%%:*}"
        lambda_tag="${lambda_pair##*:}"
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${branch_id}|${branch_tag}|${lambda_val}|${lambda_tag}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== KD-Only Single-Branch Beta=0.3 Reduced Sweep =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "branches=${BRANCHES[*]}"
echo "lambdas=${LAMBDAS[*]}"
echo "feature_beta=${FEATURE_BETA}, temp=${TEMP_VAL}, log_root=${LOG_ROOT}"
echo "skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"
echo "jobs=${job_count}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done

wait "${pids[@]}"
echo "KD-only single-branch beta=0.3 reduced sweep complete (${job_count} jobs)"
