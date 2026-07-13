#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"


# Partition pilot for the current best branch-agreement setting.
# Runs fixed alpha vs branch agreement min_scale=0.2 on:
#   1) beta_0.5
#   2) noniid_grouping
#
# wandb is enabled by default for experiment tracking.
# Disable if needed:
#   USE_WANDB=0 ./run_branch_agreement_partition_pilot.sh

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
JOB_COUNT=0

BASE_ALGO_NAME="fedprox"
BASE_ALGO_FLAGS="--use_fedprox --mu 0.01"

SEED="0"
ALPHA_VAL="0.05"
BETA_VAL="0.01"
TEMP_VAL="0.5"
MIN_SCALE="0.2"

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

run_job() {
    local env_name=$1
    local env_flags=$2
    local method_name=$3
    local extra_flags=$4

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local log_dir="logs_prev/logs_tuning/${env_name}/${BASE_ALGO_NAME}"
    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] 시작: ${env_name} | ${BASE_ALGO_NAME} | ${method_name}"

    python main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "logs_prev/logs_tuning" \
        --log_file_name "${env_name}/${BASE_ALGO_NAME}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${ALPHA_VAL}" --byot_beta "${BETA_VAL}" --temperature "${TEMP_VAL}" \
        ${env_flags} ${BASE_ALGO_FLAGS} ${extra_flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "✅ 현재 4개 배치 완료"
    fi
}

echo "========== Branch Agreement Partition Pilot =========="

run_job "beta_0.5" "--partition noniid --beta 0.5" \
    "fedsd_fixed_alpha_partition_pilot" ""
run_job "beta_0.5" "--partition noniid --beta 0.5" \
    "fedsd_branch_agree_min0p2_partition_pilot" "--byot_alpha_proxy branch_agreement --alpha_min_scale ${MIN_SCALE}"

run_job "noniid_grouping" "--partition noniid_grouping --partition_groups 8" \
    "fedsd_fixed_alpha_partition_pilot" ""
run_job "noniid_grouping" "--partition noniid_grouping --partition_groups 8" \
    "fedsd_branch_agree_min0p2_partition_pilot" "--byot_alpha_proxy branch_agreement --alpha_min_scale ${MIN_SCALE}"

wait
echo "✅ Branch agreement partition pilot 완료"
