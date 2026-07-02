#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Quick beta_0.1 pilot for active BYOT branch count.
#
# This does not remove branch modules from the forward pass. It controls which
# branch CE/KD/feature losses are used for training, which is enough for a fast
# signal on whether shallow/multiple branch supervision is hurting severe
# non-IID performance.

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
BYOT_ALPHA_VAL="${BYOT_ALPHA_VAL:-1.0}"
BYOT_BETA_VAL="${BYOT_BETA_VAL:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs_branch_count_pilot}"
ENVS=(${ENVS_OVERRIDE:-beta_0.1})

METHODS=(
    "active_b3_only:3"
    "active_b2_b3:2,3"
    "active_all_b1_b2_b3:1,2,3"
    "active_b1_only_depth_probe:1"
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
        beta_0.1)
            echo "--partition noniid --beta 0.1"
            ;;
        beta_0.3)
            echo "--partition noniid --beta 0.3"
            ;;
        beta_0.5)
            echo "--partition noniid --beta 0.5"
            ;;
        iid)
            echo "--partition iid"
            ;;
        *)
            echo "Unknown env: ${env_name}" >&2
            exit 1
            ;;
    esac
}

run_job() {
    local env_name=$1
    local method_name=$2
    local active_branches=$3

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local log_dir="${LOG_ROOT}/${env_name}/fedavg"
    local flags
    flags="$(env_flags "${env_name}")"

    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] start: ${env_name} | ${method_name} | active=${active_branches}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${BYOT_ALPHA_VAL}" --byot_beta "${BYOT_BETA_VAL}" --temperature "${TEMP_VAL}" \
        --byot_active_branches "${active_branches}" \
        ${flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

echo "========== Branch Count Pilot =========="
echo "envs=${ENVS[*]}, seed=${SEED}, log_root=${LOG_ROOT}"
echo "alpha=${BYOT_ALPHA_VAL}, byot_beta=${BYOT_BETA_VAL}, temp=${TEMP_VAL}"

for env_name in "${ENVS[@]}"; do
    for method_pair in "${METHODS[@]}"; do
        method_name="${method_pair%%:*}"
        active_branches="${method_pair#*:}"
        run_job "${env_name}" "${method_name}" "${active_branches}"
    done
done

wait
echo "Branch count pilot complete (${JOB_COUNT} jobs)"
