#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Extended client-wise reliability alpha sweep.
#
# Designed for an overnight-ish run on 4 GPUs.
# Default plan:
#   beta_0.3: fixed 0.00/0.01/0.05/0.10/0.30 + 4 client-wise proxies
#   beta_0.1: fixed 0.05/0.30 + 4 client-wise proxies
#   beta_0.5: fixed 0.05/0.30 + 4 client-wise proxies
#
# Client-wise alpha range:
#   alpha_k in [0.01, 0.30]
#
# Usage:
#   USE_WANDB=0 bash scripts/experiments/reliability/run_client_reliability_alpha_extended.sh

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

BASE_ALGO="${BASE_ALGO:-fedavg}"
if [ "$BASE_ALGO" = "fedprox" ]; then
    BASE_ALGO_FLAGS="--use_fedprox --mu ${MU_VAL:-0.01}"
else
    BASE_ALGO_FLAGS=""
fi

SEED="${SEED:-0}"
BETA_VAL="${BYOT_BETA_VAL:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
CLIENT_ALPHA_MIN="${CLIENT_ALPHA_MIN:-0.01}"
CLIENT_ALPHA_MAX="${CLIENT_ALPHA_MAX:-0.30}"
LOG_ROOT="${LOG_ROOT:-logs_client_reliability_extended}"

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

env_flags() {
    local env_name=$1
    case "$env_name" in
        beta_0.1)
            echo "--partition noniid --beta 0.1"
            ;;
        beta_0.3)
            echo "--partition noniid --beta 0.3"
            ;;
        beta_0.5)
            echo "--partition noniid --beta 0.5"
            ;;
        *)
            echo "Unknown env: ${env_name}" >&2
            exit 1
            ;;
    esac
}

client_flags() {
    local proxy=$1
    echo "--byot_client_proxy ${proxy} --byot_client_alpha_min ${CLIENT_ALPHA_MIN} --byot_client_alpha_max ${CLIENT_ALPHA_MAX}"
}

run_job() {
    local env_name=$1
    local method_name=$2
    local alpha_val=$3
    local extra_flags=$4

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local log_dir="${LOG_ROOT}/${env_name}/${BASE_ALGO}"
    local flags
    flags="$(env_flags "${env_name}")"

    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] start: ${env_name} | ${BASE_ALGO} | ${method_name} | alpha=${alpha_val}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/${BASE_ALGO}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${alpha_val}" --byot_beta "${BETA_VAL}" --temperature "${TEMP_VAL}" \
        ${flags} ${BASE_ALGO_FLAGS} ${extra_flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

run_fixed_full() {
    local env_name=$1
    run_job "${env_name}" "fixed_alpha0p00" "0.00" ""
    run_job "${env_name}" "fixed_alpha0p01" "0.01" ""
    run_job "${env_name}" "fixed_alpha0p05" "0.05" ""
    run_job "${env_name}" "fixed_alpha0p10" "0.10" ""
    run_job "${env_name}" "fixed_alpha0p30" "0.30" ""
}

run_fixed_reduced() {
    local env_name=$1
    run_job "${env_name}" "fixed_alpha0p05" "0.05" ""
    run_job "${env_name}" "fixed_alpha0p30" "0.30" ""
}

run_client_methods() {
    local env_name=$1
    run_job "${env_name}" "client_label_prob_0p01_0p30" "0.05" "$(client_flags teacher_label_prob)"
    run_job "${env_name}" "client_correctness_0p01_0p30" "0.05" "$(client_flags teacher_correctness)"
    run_job "${env_name}" "client_branch_js_0p01_0p30" "0.05" "$(client_flags branch_js)"
    run_job "${env_name}" "client_entropy_0p01_0p30" "0.05" "$(client_flags teacher_entropy)"
}

echo "========== Extended Client-wise Reliability Alpha Sweep =========="
echo "base_algo=${BASE_ALGO}, seed=${SEED}, client_alpha=${CLIENT_ALPHA_MIN}~${CLIENT_ALPHA_MAX}, log_root=${LOG_ROOT}"

run_fixed_full "beta_0.3"
run_client_methods "beta_0.3"

run_fixed_reduced "beta_0.1"
run_client_methods "beta_0.1"

run_fixed_reduced "beta_0.5"
run_client_methods "beta_0.5"

wait
echo "Extended client-wise reliability alpha sweep complete (${JOB_COUNT} jobs)"
