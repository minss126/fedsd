#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Fixed high-alpha FedSD/BYOT sweep for stronger base regularizers.
#
# This complements:
#   scripts/experiments/reliability/run_fixed_alpha_high_sweep.sh
#
# Default plan:
#   base algorithms: fedprox, moon
#   partitions: beta_0.1 / beta_0.3 / beta_0.5
#   fixed alpha: 0.50 / 0.70 / 1.00
#
# Usage:
#   USE_WANDB=0 bash scripts/experiments/reliability/run_fixed_alpha_high_base_sweep.sh
#
# Useful overrides:
#   BASE_ALGOS="fedprox" USE_WANDB=0 bash scripts/experiments/reliability/run_fixed_alpha_high_base_sweep.sh
#   BASE_ALGOS="moon" MOON_MU=1.0 USE_WANDB=0 bash scripts/experiments/reliability/run_fixed_alpha_high_base_sweep.sh
#   ENVS_OVERRIDE="beta_0.3 beta_0.5" ALPHAS="0.70:0p70 1.00:1p00" USE_WANDB=0 bash scripts/experiments/reliability/run_fixed_alpha_high_base_sweep.sh

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

SEED="${SEED:-0}"
BETA_VAL="${BYOT_BETA_VAL:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEDPROX_MU="${FEDPROX_MU:-0.01}"
MOON_MU="${MOON_MU:-1.0}"
LOG_ROOT="${LOG_ROOT:-logs_fixed_alpha_high_base}"

BASE_ALGOS=(${BASE_ALGOS:-fedprox moon})
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

base_flags() {
    local base_algo=$1
    case "$base_algo" in
        fedprox)
            echo "--use_fedprox --mu ${FEDPROX_MU}"
            ;;
        moon)
            echo "--use_moon --mu ${MOON_MU}"
            ;;
        *)
            echo "Unknown base algorithm: ${base_algo}" >&2
            exit 1
            ;;
    esac
}

run_job() {
    local base_algo=$1
    local env_name=$2
    local alpha_val=$3
    local alpha_tag=$4

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local method_name="fixed_alpha${alpha_tag}"
    local log_dir="${LOG_ROOT}/${env_name}/${base_algo}"
    local flags
    local regularizer_flags
    flags="$(env_flags "${env_name}")"
    regularizer_flags="$(base_flags "${base_algo}")"

    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] start: ${env_name} | ${base_algo} | ${method_name} | alpha=${alpha_val}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/${base_algo}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${alpha_val}" --byot_beta "${BETA_VAL}" --temperature "${TEMP_VAL}" \
        ${flags} ${regularizer_flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

echo "========== Fixed High-Alpha Base Sweep =========="
echo "base_algos=${BASE_ALGOS[*]}, seed=${SEED}, log_root=${LOG_ROOT}"
echo "fedprox_mu=${FEDPROX_MU}, moon_mu=${MOON_MU}, temp=${TEMP_VAL}, byot_beta=${BETA_VAL}"
echo "envs=${ENVS[*]}"
echo "alphas=${ALPHAS[*]}"

for base_algo in "${BASE_ALGOS[@]}"; do
    for env_name in "${ENVS[@]}"; do
        for alpha_pair in "${ALPHAS[@]}"; do
            alpha_val="${alpha_pair%%:*}"
            alpha_tag="${alpha_pair##*:}"
            run_job "${base_algo}" "${env_name}" "${alpha_val}" "${alpha_tag}"
        done
    done
done

wait
echo "Fixed high-alpha base sweep complete (${JOB_COUNT} jobs)"
