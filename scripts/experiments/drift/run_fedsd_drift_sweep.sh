#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"


# Drift sweep for the FedSD theorem check.
# Measures whether client update drift grows as non-IID becomes stronger
# and as the fixed BYOT alpha becomes larger.
#
# Runs:
#   partition in {iid, beta_0.5, beta_0.3, beta_0.1}
#   alpha in {0.00, 0.01, 0.05, 0.10, 0.30}
#
# Results are isolated under logs_drift/ to avoid mixing with previous runs.
#
# wandb is enabled by default for experiment tracking.
# Disable if needed:
#   USE_WANDB=0 ./run_fedsd_drift_sweep.sh

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
JOB_COUNT=0

LOG_ROOT="${LOG_ROOT:-logs_drift}"
BASE_ALGO_NAME="fedprox"
BASE_ALGO_FLAGS="--use_fedprox --mu 0.01"

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
BETA_VAL="0.01"
TEMP_VAL="0.5"
DRIFT_LOG_INTERVAL="${DRIFT_LOG_INTERVAL:-1}"

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

run_job() {
    local env_name=$1
    local env_flags=$2
    local alpha_val=$3
    local alpha_tag=$4

    local method_name="fedsd_alpha${alpha_tag}_drift"
    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local log_dir="${LOG_ROOT}/${env_name}/${BASE_ALGO_NAME}"
    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] 시작: ${env_name} | ${BASE_ALGO_NAME} | ${method_name} | alpha=${alpha_val}"

    python main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/${BASE_ALGO_NAME}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${alpha_val}" --byot_beta "${BETA_VAL}" --temperature "${TEMP_VAL}" \
        --log_client_drift --drift_log_interval "${DRIFT_LOG_INTERVAL}" \
        ${env_flags} ${BASE_ALGO_FLAGS} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "✅ 현재 4개 배치 완료"
    fi
}

run_partition() {
    local env_name=$1
    local env_flags=$2

    run_job "$env_name" "$env_flags" "0.00" "0p00"
    run_job "$env_name" "$env_flags" "0.01" "0p01"
    run_job "$env_name" "$env_flags" "0.05" "0p05"
    run_job "$env_name" "$env_flags" "0.10" "0p10"
    run_job "$env_name" "$env_flags" "0.30" "0p30"
}

echo "========== FedSD Drift Sweep =========="
echo "LOG_ROOT=${LOG_ROOT}, ROUNDS=${ROUNDS}, SEED=${SEED}, DRIFT_LOG_INTERVAL=${DRIFT_LOG_INTERVAL}"

run_partition "iid" "--partition iid"
run_partition "beta_0.5" "--partition noniid --beta 0.5"
run_partition "beta_0.3" "--partition noniid --beta 0.3"
run_partition "beta_0.1" "--partition noniid --beta 0.1"

wait
echo "✅ FedSD drift sweep 완료"
