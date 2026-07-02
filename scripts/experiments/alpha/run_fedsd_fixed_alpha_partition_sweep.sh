#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"


# Fixed-alpha sensitivity check across partitions.
# This tests whether the best KD alpha changes with the non-IID setting.
#
# Reused alpha=0.05 controls:
#   beta_0.3/fedprox/fedsd_fixed_alpha
#   beta_0.5/fedprox/fedsd_fixed_alpha_partition_pilot
#   noniid_grouping/fedprox/fedsd_fixed_alpha_partition_pilot
#
# New runs intentionally use *_partition_sweep names so previous results are not overwritten.
#
# wandb is enabled by default for experiment tracking.
# Disable if needed:
#   USE_WANDB=0 ./run_fedsd_fixed_alpha_partition_sweep.sh

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
JOB_COUNT=0

BASE_ALGO_NAME="fedprox"
BASE_ALGO_FLAGS="--use_fedprox --mu 0.01"

SEED="0"
BETA_VAL="0.01"
TEMP_VAL="0.5"

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
    local alpha_val=$3
    local alpha_tag=$4

    local method_name="fedsd_alpha${alpha_tag}_partition_sweep"
    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local log_dir="logs_tuning/${env_name}/${BASE_ALGO_NAME}"
    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] 시작: ${env_name} | ${BASE_ALGO_NAME} | ${method_name} | alpha=${alpha_val}"

    python main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "logs_tuning" \
        --log_file_name "${env_name}/${BASE_ALGO_NAME}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${alpha_val}" --byot_beta "${BETA_VAL}" --temperature "${TEMP_VAL}" \
        ${env_flags} ${BASE_ALGO_FLAGS} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "✅ 현재 4개 배치 완료"
    fi
}

run_partition() {
    local env_name=$1
    local env_flags=$2

    run_job "$env_name" "$env_flags" "0.00" "0p00"
    run_job "$env_name" "$env_flags" "0.01" "0p01"
    run_job "$env_name" "$env_flags" "0.03" "0p03"
    run_job "$env_name" "$env_flags" "0.10" "0p10"
    run_job "$env_name" "$env_flags" "0.20" "0p20"
    run_job "$env_name" "$env_flags" "0.30" "0p30"
}

echo "========== Fedsd Fixed Alpha Partition Sweep =========="

run_partition "beta_0.3" "--partition noniid --beta 0.3"
run_partition "beta_0.5" "--partition noniid --beta 0.5"
run_partition "noniid_grouping" "--partition noniid_grouping --partition_groups 8"

wait
echo "✅ Fedsd fixed alpha partition sweep 완료"
