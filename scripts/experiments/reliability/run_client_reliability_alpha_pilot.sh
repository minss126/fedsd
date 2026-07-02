#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Client-wise reliability alpha pilot.
#
# Default:
#   beta_0.3 / fedavg / seed 0
#   fixed alpha baselines: 0.00, 0.01, 0.05, 0.10, 0.30
#   client-wise alpha range: 0.01 ~ 0.30

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
BETA_VAL="${BETA_VAL:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
CLIENT_ALPHA_MIN="${CLIENT_ALPHA_MIN:-0.01}"
CLIENT_ALPHA_MAX="${CLIENT_ALPHA_MAX:-0.30}"
LOG_ROOT="${LOG_ROOT:-logs_client_reliability}"
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
    local alpha_val=$2
    local extra_flags=$3

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
        --byot_alpha "${alpha_val}" --byot_beta "${BETA_VAL}" --temperature "${TEMP_VAL}" \
        ${ENV_FLAGS} ${BASE_ALGO_FLAGS} ${extra_flags} ${WANDB_FLAGS} \
        > "${LOG_DIR}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

client_flags() {
    local proxy=$1
    echo "--byot_client_proxy ${proxy} --byot_client_alpha_min ${CLIENT_ALPHA_MIN} --byot_client_alpha_max ${CLIENT_ALPHA_MAX}"
}

echo "========== Client-wise Reliability Alpha Pilot =========="
echo "env=${ENV_NAME}, base_algo=${BASE_ALGO}, seed=${SEED}, client_alpha=${CLIENT_ALPHA_MIN}~${CLIENT_ALPHA_MAX}"

run_job "fixed_alpha0p00" "0.00" ""
run_job "fixed_alpha0p01" "0.01" ""
run_job "fixed_alpha0p05" "0.05" ""
run_job "fixed_alpha0p10" "0.10" ""
run_job "fixed_alpha0p30" "0.30" ""

run_job "client_label_prob_0p01_0p30" "0.05" "$(client_flags teacher_label_prob)"
run_job "client_correctness_0p01_0p30" "0.05" "$(client_flags teacher_correctness)"
run_job "client_branch_js_0p01_0p30" "0.05" "$(client_flags branch_js)"
run_job "client_entropy_0p01_0p30" "0.05" "$(client_flags teacher_entropy)"

wait
echo "Client-wise reliability alpha pilot complete"
