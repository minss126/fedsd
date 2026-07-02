#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Fixed high-alpha sweep for FedSD/BYOT.
#
# Purpose:
#   Check whether alpha=0.30 was only the current upper-bound winner,
#   or whether performance keeps improving at larger alpha values.
#
# Default:
#   beta_0.1 / beta_0.3 / beta_0.5
#   fixed alpha in {0.50, 0.70, 1.00}
#
# Usage:
#   USE_WANDB=0 bash scripts/experiments/reliability/run_fixed_alpha_high_sweep.sh
#
# To skip the extreme alpha=1.00 stress test:
#   ALPHAS="0.50:0p50 0.70:0p70" USE_WANDB=0 bash scripts/experiments/reliability/run_fixed_alpha_high_sweep.sh

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
LOG_ROOT="${LOG_ROOT:-logs_fixed_alpha_high}"

ENVS=(${ENVS_OVERRIDE:-beta_0.1 beta_0.3 beta_0.5})
ALPHAS=(${ALPHAS:-0.50:0p50 0.70:0p70 1.00:1p00})

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

run_job() {
    local env_name=$1
    local alpha_val=$2
    local alpha_tag=$3

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local method_name="fixed_alpha${alpha_tag}"
    local log_dir="${LOG_ROOT}/${env_name}/${BASE_ALGO}"
    local flags
    flags="$(env_flags "${env_name}")"

    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] start: ${env_name} | ${BASE_ALGO} | ${method_name}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/${BASE_ALGO}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${alpha_val}" --byot_beta "${BETA_VAL}" --temperature "${TEMP_VAL}" \
        ${flags} ${BASE_ALGO_FLAGS} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

echo "========== Fixed High-Alpha Sweep =========="
echo "base_algo=${BASE_ALGO}, seed=${SEED}, log_root=${LOG_ROOT}"
echo "envs=${ENVS[*]}"
echo "alphas=${ALPHAS[*]}"

for env_name in "${ENVS[@]}"; do
    for alpha_pair in "${ALPHAS[@]}"; do
        alpha_val="${alpha_pair%%:*}"
        alpha_tag="${alpha_pair##*:}"
        run_job "${env_name}" "${alpha_val}" "${alpha_tag}"
    done
done

wait
echo "Fixed high-alpha sweep complete (${JOB_COUNT} jobs)"
