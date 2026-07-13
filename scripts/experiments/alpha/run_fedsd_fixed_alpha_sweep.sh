#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"


# Fixed-alpha sensitivity check for Fedsd.
# This tests whether changing the KD alpha alone is a meaningful performance axis.
#
# Existing alpha=0.05 control:
#   logs_prev/logs_tuning/beta_0.3/fedprox/fedsd_fixed_alpha
#
# New runs below intentionally avoid that name so existing results are not overwritten.
#
# wandb is enabled by default for experiment tracking.
# Disable if needed:
#   USE_WANDB=0 ./run_fedsd_fixed_alpha_sweep.sh

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
JOB_COUNT=0

ENV_NAME="beta_0.3"
ENV_FLAGS="--partition noniid --beta 0.3"
BASE_ALGO_NAME="fedprox"
BASE_ALGO_FLAGS="--use_fedprox --mu 0.01"

SEED="0"
BETA_VAL="0.01"
TEMP_VAL="0.5"

LOG_DIR="logs_prev/logs_tuning/${ENV_NAME}/${BASE_ALGO_NAME}"
mkdir -p "$LOG_DIR"

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

run_job() {
    local alpha_val=$1
    local method_name=$2

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}

    echo "[GPU ${gpu_id}] 시작: ${ENV_NAME} | ${BASE_ALGO_NAME} | ${method_name} | alpha=${alpha_val}"

    python main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "logs_prev/logs_tuning" \
        --log_file_name "${ENV_NAME}/${BASE_ALGO_NAME}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${alpha_val}" --byot_beta "${BETA_VAL}" --temperature "${TEMP_VAL}" \
        ${ENV_FLAGS} ${BASE_ALGO_FLAGS} ${WANDB_FLAGS} \
        > "${LOG_DIR}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "✅ 현재 4개 배치 완료"
    fi
}

echo "========== Fedsd Fixed Alpha Sweep =========="

run_job "0.00" "fedsd_alpha0p00"
run_job "0.01" "fedsd_alpha0p01"
run_job "0.03" "fedsd_alpha0p03"
run_job "0.10" "fedsd_alpha0p10"
run_job "0.20" "fedsd_alpha0p20"
run_job "0.30" "fedsd_alpha0p30"

wait
echo "✅ Fedsd fixed alpha sweep 완료"
