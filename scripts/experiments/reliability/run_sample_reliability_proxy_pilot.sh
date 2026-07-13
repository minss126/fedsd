#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Sample-wise reliability proxy pilot for BYOT/FedSD.
#
# Default:
#   beta_0.3 / fedavg / seed 0 / alpha 0.05
#
# Examples:
#   USE_WANDB=0 bash scripts/experiments/reliability/run_sample_reliability_proxy_pilot.sh
#   BASE_ALGO=fedprox MIN_SCALE=0.2 bash scripts/experiments/reliability/run_sample_reliability_proxy_pilot.sh

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
JOB_COUNT=0

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then
        PYTHON_BIN="venv/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

ENV_NAME="${ENV_NAME:-beta_0.3}"
ENV_FLAGS="${ENV_FLAGS:---partition noniid --beta 0.3}"

BASE_ALGO="${BASE_ALGO:-fedavg}"
if [ "$BASE_ALGO" = "fedprox" ]; then
    BASE_ALGO_FLAGS="--use_fedprox --mu ${MU_VAL:-0.01}"
else
    BASE_ALGO_FLAGS=""
fi

SEED="${SEED:-0}"
ALPHA_VAL="${ALPHA_VAL:-0.05}"
BETA_VAL="${BETA_VAL:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
MIN_SCALE="${MIN_SCALE:-0.0}"
LOG_ROOT="${LOG_ROOT:-logs/reliability/logs_reliability}"
LOG_DIR="${LOG_ROOT}/${ENV_NAME}/${BASE_ALGO}"

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
    local extra_flags=$2

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}

    echo "[GPU ${gpu_id}] start: ${ENV_NAME} | ${BASE_ALGO} | ${method_name}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${ENV_NAME}/${BASE_ALGO}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${ALPHA_VAL}" --byot_beta "${BETA_VAL}" --temperature "${TEMP_VAL}" \
        --alpha_min_scale "${MIN_SCALE}" \
        ${ENV_FLAGS} ${BASE_ALGO_FLAGS} ${extra_flags} ${WANDB_FLAGS} \
        > "${LOG_DIR}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

echo "========== Sample-wise Reliability Proxy Pilot =========="
echo "env=${ENV_NAME}, base_algo=${BASE_ALGO}, seed=${SEED}, alpha=${ALPHA_VAL}, min_scale=${MIN_SCALE}"

run_job "fedsd_fixed" ""
run_job "sample_teacher_conf" "--byot_sample_proxy teacher_conf"
run_job "sample_teacher_entropy" "--byot_sample_proxy teacher_entropy"
run_job "sample_teacher_margin" "--byot_sample_proxy teacher_margin"
run_job "sample_teacher_label_prob" "--byot_sample_proxy teacher_label_prob"
run_job "sample_teacher_correctness" "--byot_sample_proxy teacher_correctness"
run_job "sample_branch_agreement" "--byot_sample_proxy branch_agreement"
run_job "sample_branch_soft_kl" "--byot_sample_proxy branch_soft_kl"
run_job "sample_branch_js" "--byot_sample_proxy branch_js"

wait
echo "Sample-wise reliability proxy pilot complete"
