#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"


# Fedsd-only proxy alpha ablation.
# These methods do not assume known Dirichlet beta during training.

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

run_job() {
    local method_name=$1
    local proxy_name=$2

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}

    echo "[GPU ${gpu_id}] 시작: ${ENV_NAME} | ${BASE_ALGO_NAME} | ${method_name}"

    python main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed 0 \
        --device "cuda:${gpu_id}" \
        --logdir "logs_tuning" \
        --log_file_name "${ENV_NAME}/${BASE_ALGO_NAME}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${ALPHA_VAL}" --byot_beta "${BETA_VAL}" --temperature "${TEMP_VAL}" \
        --byot_alpha_proxy "${proxy_name}" --alpha_min_scale "${MIN_SCALE}" \
        ${ENV_FLAGS} ${BASE_ALGO_FLAGS} \
        > "${LOG_DIR}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))
}

echo "========== Fedsd Proxy Alpha Ablation =========="

run_job "fedsd_proxy_teacher_conf" "teacher_conf"
run_job "fedsd_proxy_teacher_entropy" "teacher_entropy"
run_job "fedsd_proxy_branch_agreement" "branch_agreement"
run_job "fedsd_proxy_teacher_correctness" "teacher_correctness"

wait
echo "✅ Fedsd proxy alpha ablation 완료"
