#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"


# Re-run IID/FedAvg BYOT variants with the current tuning hyperparameters.
# This fixes the old logs/iid/fedavg fallback runs that used different epochs/alpha/beta.

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
JOB_COUNT=0

MU_VAL="0.01"
TEMP_VAL="0.5"
ALPHA_VAL="0.05"
BETA_VAL="0.01"

LOG_DIR="logs_tuning/iid/fedavg"
mkdir -p "$LOG_DIR"

run_job() {
    local method_name=$1
    local method_flags=$2

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}

    echo "[GPU ${gpu_id}] 시작: iid | fedavg | ${method_name}"

    python main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed 0 \
        --device "cuda:${gpu_id}" \
        --logdir "logs_tuning" \
        --log_file_name "iid/fedavg/${method_name}" \
        --partition iid \
        $method_flags \
        > "${LOG_DIR}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))
}

echo "========== IID/FedAvg BYOT 3개 작업 재실행 =========="

run_job "fedsd" \
    "--model resnet18_byot --alg fedbyot --kd_conf_threshold 0.0 --byot_alpha ${ALPHA_VAL} --byot_beta ${BETA_VAL} --mu ${MU_VAL} --temperature ${TEMP_VAL}"

run_job "selective" \
    "--model resnet18_byot --alg fedbyot_selective --kd_conf_threshold 0.8 --byot_alpha ${ALPHA_VAL} --byot_beta ${BETA_VAL} --mu ${MU_VAL} --temperature ${TEMP_VAL}"

run_job "warmup" \
    "--model resnet18_byot --alg fedbyot_selective_greedy --kd_conf_threshold 0.8 --min_threshold 0.3 --warmup_epochs 2 --byot_alpha ${ALPHA_VAL} --byot_beta ${BETA_VAL} --mu ${MU_VAL} --temperature ${TEMP_VAL}"

wait
echo "✅ IID/FedAvg BYOT 재실행 완료"
