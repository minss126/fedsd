#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"


# Multi-seed check for the current best branch-agreement setting.
# Runs fixed alpha vs branch agreement min_scale=0.2 for seed 1 and 2.
#
# wandb is enabled by default for experiment tracking.
# Disable if needed:
#   USE_WANDB=0 ./run_branch_agreement_seed_check.sh

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
JOB_COUNT=0

ENV_NAME="beta_0.3"
ENV_FLAGS="--partition noniid --beta 0.3"
BASE_ALGO_NAME="fedprox"
BASE_ALGO_FLAGS="--use_fedprox --mu 0.01"

ALPHA_VAL="0.05"
BETA_VAL="0.01"
TEMP_VAL="0.5"
MIN_SCALE="0.2"

LOG_DIR="logs_tuning/${ENV_NAME}/${BASE_ALGO_NAME}"
mkdir -p "$LOG_DIR"

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

run_job() {
    local method_name=$1
    local seed=$2
    local extra_flags=$3

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}

    echo "[GPU ${gpu_id}] 시작: ${ENV_NAME} | ${BASE_ALGO_NAME} | ${method_name} | seed=${seed}"

    python main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${seed}" \
        --device "cuda:${gpu_id}" \
        --logdir "logs_tuning" \
        --log_file_name "${ENV_NAME}/${BASE_ALGO_NAME}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${ALPHA_VAL}" --byot_beta "${BETA_VAL}" --temperature "${TEMP_VAL}" \
        ${ENV_FLAGS} ${BASE_ALGO_FLAGS} ${extra_flags} ${WANDB_FLAGS} \
        > "${LOG_DIR}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "✅ 현재 4개 배치 완료"
    fi
}

echo "========== Branch Agreement Multi-Seed Check =========="

run_job "fedsd_fixed_alpha_s1" "1" ""
run_job "fedsd_branch_agree_min0p2_s1" "1" "--byot_alpha_proxy branch_agreement --alpha_min_scale ${MIN_SCALE}"
run_job "fedsd_fixed_alpha_s2" "2" ""
run_job "fedsd_branch_agree_min0p2_s2" "2" "--byot_alpha_proxy branch_agreement --alpha_min_scale ${MIN_SCALE}"

wait
echo "✅ Branch agreement multi-seed check 완료"
