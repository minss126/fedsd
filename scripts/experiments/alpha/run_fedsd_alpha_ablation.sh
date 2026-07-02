#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"


# Fedsd-only alpha adaptation ablation.
# Four jobs run in parallel on GPU 0..3:
# 1) fixed alpha, 2) beta-aware alpha, 3) entropy-aware alpha, 4) beta+entropy alpha.

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

LOG_DIR="logs_tuning/${ENV_NAME}/${BASE_ALGO_NAME}"
mkdir -p "$LOG_DIR"

run_job() {
    local method_name=$1
    local extra_flags=$2

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
        ${ENV_FLAGS} ${BASE_ALGO_FLAGS} ${extra_flags} \
        > "${LOG_DIR}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))
}

echo "========== Fedsd Alpha Adaptation Ablation =========="

run_job "fedsd_fixed_alpha" ""
run_job "fedsd_beta_alpha" "--beta_aware_byot_alpha --alpha_beta_ref 0.5 --alpha_min_scale 0.2"
run_job "fedsd_entropy_alpha" "--adaptive_byot_alpha --alpha_min_scale 0.2 --alpha_entropy_power 1.0"
run_job "fedsd_beta_entropy_alpha" "--beta_aware_byot_alpha --adaptive_byot_alpha --alpha_beta_ref 0.5 --alpha_min_scale 0.2 --alpha_entropy_power 1.0"

wait
echo "✅ Fedsd alpha adaptation ablation 완료"
