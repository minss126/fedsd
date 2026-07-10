#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Plain FedAvg baselines for ImageNet100-64 / ResNet18.
#
# This is the plain counterpart to the ImageNet100-64 BYOT/FedSD alpha sweep
# in scripts/experiments/alpha/run_simple_dataset_alpha_screen.sh.
#
# Default matrix:
#   dataset/model : imagenet100_64 / resnet18
#   partitions    : iid, beta_0.5, beta_0.3, beta_0.1
#   rounds        : 300
#   gpus          : 0 1
#
# Usage:
#   bash scripts/experiments/baseline/run_plain_imagenet100_64.sh
#
# Useful overrides:
#   GPUS_OVERRIDE="0 1" bash scripts/experiments/baseline/run_plain_imagenet100_64.sh
#   PARTITIONS_OVERRIDE="iid beta_0.5" bash scripts/experiments/baseline/run_plain_imagenet100_64.sh
#   IMAGENET100_ROUNDS=500 bash scripts/experiments/baseline/run_plain_imagenet100_64.sh

GPUS=(${GPUS_OVERRIDE:-0 1})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPU ids provided. Set GPUS_OVERRIDE." >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then
        PYTHON_BIN="venv/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

SEED="${SEED:-0}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ROUNDS="${IMAGENET100_ROUNDS:-300}"
LR="${IMAGENET100_LR:-0.01}"
LOG_ROOT="${LOG_ROOT:-logs_plain_imagenet100_64}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DATA_DIR="${IMAGENET100_DATADIR:-/data/imagenet100_resized_64_png}"

PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

partition_flags() {
    local partition_name=$1
    case "$partition_name" in
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
        *)
            echo "Unknown partition: ${partition_name}" >&2
            exit 1
            ;;
    esac
}

run_job() {
    local gpu_id=$1
    local partition_name=$2

    local setting="imagenet100_64_resnet18"
    local method_name="plain_fedavg"
    local log_dir="${LOG_ROOT}/${setting}/${partition_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local partition_args

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${setting} | ${partition_name} | ${method_name}"
        return
    fi

    partition_args="$(partition_flags "$partition_name")"

    echo "[GPU ${gpu_id}] start: ${setting} | ${partition_name} | ${method_name} | rounds=${ROUNDS}"

    if ! "${PYTHON_BIN}" main.py \
        --dataset imagenet100_64 --datadir "${DATA_DIR}" \
        --in_channels 3 --num_classes 100 \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${LR}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${setting}/${partition_name}/fedavg/${method_name}" \
        ${partition_args} \
        --model resnet18 --alg fedavg \
        ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu_id}] FAILED: ${setting} | ${partition_name} | ${method_name}"
        return
    fi

    echo "[GPU ${gpu_id}] complete: ${setting} | ${partition_name} | ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")

    for partition_name in "${jobs[@]}"; do
        [ -z "$partition_name" ] && continue
        run_job "$gpu_id" "$partition_name"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for partition_name in "${PARTITIONS[@]}"; do
    gpu_idx=$((job_count % NUM_GPUS))
    QUEUES[$gpu_idx]+="${partition_name}"$'\n'
    job_count=$((job_count + 1))
done

echo "========== Plain ImageNet100-64 Baselines =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "partitions=${PARTITIONS[*]}"
echo "lr=${LR}, data_dir=${DATA_DIR}"
echo "log_root=${LOG_ROOT}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"
echo "jobs=${job_count}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done

wait "${pids[@]}"
echo "Plain ImageNet100-64 baselines complete (${job_count} jobs)"
