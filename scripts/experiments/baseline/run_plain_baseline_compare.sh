#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Plain ResNet baselines for fair comparison against BYOT/FedSD runs.
#
# Why this exists:
#   Some existing "baseline" logs use model=resnet18_byot, which is useful as
#   a BYOT-architecture ablation but not as a plain FL baseline. This script
#   runs model=resnet18 without BYOT branches under the same core training
#   settings used by the recent high-alpha FedSD experiments.
#
# Default plan:
#   partitions: beta_0.1 / beta_0.3 / beta_0.5
#   methods: FedAvg / FedProx / MOON
#   model: resnet18
#   dataset: CIFAR-100
#   rounds: 500, local epochs: 5, seed: 0
#
# Usage:
#   bash scripts/experiments/baseline/run_plain_baseline_compare.sh
#
# Useful overrides:
#   ENVS_OVERRIDE="beta_0.3 beta_0.5" bash scripts/experiments/baseline/run_plain_baseline_compare.sh
#   ALGOS="fedavg fedprox" bash scripts/experiments/baseline/run_plain_baseline_compare.sh
#   USE_WANDB=0 bash scripts/experiments/baseline/run_plain_baseline_compare.sh

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
TEMP_VAL="${TEMP_VAL:-0.5}"
FEDPROX_MU="${FEDPROX_MU:-0.01}"
MOON_MU="${MOON_MU:-1.0}"
LOG_ROOT="${LOG_ROOT:-logs/baseline/logs_plain_baseline_compare}"

ENVS=(${ENVS_OVERRIDE:-beta_0.1 beta_0.3 beta_0.5})
ALGOS=(${ALGOS:-fedavg fedprox moon})

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

algo_flags() {
    local algo=$1
    case "$algo" in
        fedavg)
            echo "--alg fedavg"
            ;;
        fedprox)
            echo "--alg fedprox --mu ${FEDPROX_MU}"
            ;;
        moon)
            echo "--alg moon --mu ${MOON_MU} --temperature ${TEMP_VAL}"
            ;;
        *)
            echo "Unknown algorithm: ${algo}" >&2
            exit 1
            ;;
    esac
}

run_job() {
    local env_name=$1
    local algo=$2

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local log_dir="${LOG_ROOT}/${env_name}/${algo}"
    local flags
    local method_flags
    flags="$(env_flags "${env_name}")"
    method_flags="$(algo_flags "${algo}")"

    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] start: ${env_name} | plain resnet18 | ${algo}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/${algo}/baseline" \
        --model resnet18 \
        ${flags} ${method_flags} ${WANDB_FLAGS} \
        > "${log_dir}/baseline_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

echo "========== Plain Baseline Compare =========="
echo "seed=${SEED}, log_root=${LOG_ROOT}"
echo "fedprox_mu=${FEDPROX_MU}, moon_mu=${MOON_MU}, temp=${TEMP_VAL}"
echo "envs=${ENVS[*]}"
echo "algos=${ALGOS[*]}"

for env_name in "${ENVS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        run_job "${env_name}" "${algo}"
    done
done

wait
echo "Plain baseline compare complete (${JOB_COUNT} jobs)"
