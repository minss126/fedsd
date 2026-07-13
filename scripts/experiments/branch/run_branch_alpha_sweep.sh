#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Branch-wise BYOT/FedSD alpha sweep.
#
# Purpose:
#   Test whether B1/B2/B3 should use different KD-vs-CE weights.
#   The loss becomes:
#     teacher_CE + sum_i [(1 - alpha_i) * branch_i_CE + alpha_i * branch_i_KD] + beta * feature_loss
#
# Default plan:
#   base algorithm: FedAvg
#   partitions: iid / beta_0.1 / beta_0.3 / beta_0.5
#   branch alpha patterns:
#     uniform high      : 1.0, 1.0, 1.0
#     shallow-heavy     : 1.0, 0.7, 0.5
#     deep-heavy        : 0.5, 0.7, 1.0
#     middle/deep-heavy : 0.7, 1.0, 1.0
#     shallow-only high : 1.0, 0.5, 0.5
#
# Usage:
#   bash scripts/experiments/branch/run_branch_alpha_sweep.sh
#
# Useful overrides:
#   ENVS_OVERRIDE="beta_0.1 beta_0.3" bash scripts/experiments/branch/run_branch_alpha_sweep.sh
#   BASE_ALGOS="fedavg fedprox moon" bash scripts/experiments/branch/run_branch_alpha_sweep.sh
#   USE_WANDB=0 bash scripts/experiments/branch/run_branch_alpha_sweep.sh

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
BYOT_ALPHA_FALLBACK="${BYOT_ALPHA_FALLBACK:-1.0}"
BYOT_BETA_VAL="${BYOT_BETA_VAL:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEDPROX_MU="${FEDPROX_MU:-0.01}"
MOON_MU="${MOON_MU:-1.0}"
LOG_ROOT="${LOG_ROOT:-logs/branch/logs_branch_alpha_sweep}"

BASE_ALGOS=(${BASE_ALGOS:-fedavg})
ENVS=(${ENVS_OVERRIDE:-iid beta_0.1 beta_0.3 beta_0.5})
BRANCH_ALPHA_SETS=(
    "1.00,1.00,1.00:branch_uniform_1_1_1"
    "1.00,0.70,0.50:branch_shallow_heavy_1_0p7_0p5"
    "0.50,0.70,1.00:branch_deep_heavy_0p5_0p7_1"
    "0.70,1.00,1.00:branch_mid_deep_heavy_0p7_1_1"
    "1.00,0.50,0.50:branch_shallow_only_1_0p5_0p5"
)

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
        iid)
            echo "--partition iid"
            ;;
        beta_0.1)
            echo "--partition noniid --beta 0.1"
            ;;
        beta_0.3)
            echo "--partition noniid --beta 0.3"
            ;;
        beta_0.5)
            echo "--partition noniid --beta 0.5"
            ;;
        noniid_grouping)
            echo "--partition noniid_grouping --partition_groups 8"
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
        fedavg)
            echo ""
            ;;
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
    local branch_alphas=$3
    local method_name=$4

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local log_dir="${LOG_ROOT}/${env_name}/${base_algo}"
    local flags
    local regularizer_flags
    flags="$(env_flags "${env_name}")"
    regularizer_flags="$(base_flags "${base_algo}")"

    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] start: ${env_name} | ${base_algo} | ${method_name} | branch_alphas=${branch_alphas}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/${base_algo}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${BYOT_ALPHA_FALLBACK}" \
        --byot_branch_alphas "${branch_alphas}" \
        --byot_beta "${BYOT_BETA_VAL}" --temperature "${TEMP_VAL}" \
        ${flags} ${regularizer_flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

echo "========== Branch-wise Alpha Sweep =========="
echo "base_algos=${BASE_ALGOS[*]}, seed=${SEED}, log_root=${LOG_ROOT}"
echo "envs=${ENVS[*]}"
echo "byot_beta=${BYOT_BETA_VAL}, temp=${TEMP_VAL}, fedprox_mu=${FEDPROX_MU}, moon_mu=${MOON_MU}"
echo "wandb=${USE_WANDB:-1}"

for base_algo in "${BASE_ALGOS[@]}"; do
    for env_name in "${ENVS[@]}"; do
        for alpha_pair in "${BRANCH_ALPHA_SETS[@]}"; do
            branch_alphas="${alpha_pair%%:*}"
            method_name="${alpha_pair##*:}"
            run_job "${base_algo}" "${env_name}" "${branch_alphas}" "${method_name}"
        done
    done
done

wait
echo "Branch-wise alpha sweep complete (${JOB_COUNT} jobs)"
