#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Plain FedAvg baselines for dataset-level alpha-trend comparisons.
#
# This runs the non-BYOT ResNet18 model so the results can be compared against
# earlier BYOT/FedSD alpha sweeps on CIFAR-10, FMNIST, and TinyImageNet.
#
# Default matrix:
#   settings:
#     - CIFAR-10 / ResNet18        : 500 rounds
#     - FMNIST / ResNet18          : 100 rounds
#     - TinyImageNet / ResNet18    : 500 rounds
#   partitions:
#     - iid, beta_0.5, beta_0.3, beta_0.1
#
# Usage:
#   bash scripts/experiments/baseline/run_plain_dataset_generalization.sh
#
# Useful overrides:
#   GPUS_OVERRIDE="0 1" bash scripts/experiments/baseline/run_plain_dataset_generalization.sh
#   SETTINGS_OVERRIDE="cifar10_resnet18 fmnist_resnet18" bash scripts/experiments/baseline/run_plain_dataset_generalization.sh
#   PARTITIONS_OVERRIDE="iid beta_0.1" bash scripts/experiments/baseline/run_plain_dataset_generalization.sh
#   TINYIMAGENET_ROUNDS=300 bash scripts/experiments/baseline/run_plain_dataset_generalization.sh

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
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
LOG_ROOT="${LOG_ROOT:-logs_plain_dataset_generalization}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SETTINGS=(${SETTINGS_OVERRIDE:-cifar10_resnet18 fmnist_resnet18 tinyimagenet_resnet18})
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
    local rounds=$2
    [ -f "$log_file" ] && grep -q "Round $((rounds - 1)) result" "$log_file"
}

dataset_flags() {
    local setting=$1
    case "$setting" in
        cifar10_resnet18)
            echo "--dataset cifar10 --datadir ./data"
            ;;
        fmnist_resnet18)
            echo "--dataset fmnist --datadir ./data --in_channels 1 --num_classes 10"
            ;;
        tinyimagenet_resnet18)
            echo "--dataset tinyimagenet --datadir ./data/tiny-imagenet-200"
            ;;
        *)
            echo "Unknown setting: ${setting}" >&2
            exit 1
            ;;
    esac
}

setting_rounds() {
    local setting=$1
    case "$setting" in
        cifar10_resnet18)
            echo "${CIFAR10_ROUNDS:-500}"
            ;;
        fmnist_resnet18)
            echo "${FMNIST_ROUNDS:-100}"
            ;;
        tinyimagenet_resnet18)
            echo "${TINYIMAGENET_ROUNDS:-500}"
            ;;
        *)
            echo "Unknown setting: ${setting}" >&2
            exit 1
            ;;
    esac
}

setting_lr() {
    local setting=$1
    case "$setting" in
        tinyimagenet_resnet18)
            echo "${TINYIMAGENET_LR:-0.01}"
            ;;
        *)
            echo "${LR:-0.1}"
            ;;
    esac
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
    local setting=$2
    local partition_name=$3

    local rounds
    local lr
    local method_name="plain_fedavg"
    local log_dir="${LOG_ROOT}/${setting}/${partition_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local data_args
    local partition_args

    rounds="$(setting_rounds "$setting")"
    lr="$(setting_lr "$setting")"

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file" "$rounds"; then
        echo "[skip] ${setting} | ${partition_name} | ${method_name}"
        return
    fi

    data_args="$(dataset_flags "$setting")"
    partition_args="$(partition_flags "$partition_name")"

    echo "[GPU ${gpu_id}] start: ${setting} | ${partition_name} | ${method_name} | rounds=${rounds}"

    if ! "${PYTHON_BIN}" main.py \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${lr}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${rounds}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${setting}/${partition_name}/fedavg/${method_name}" \
        ${data_args} ${partition_args} \
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

    for job in "${jobs[@]}"; do
        [ -z "$job" ] && continue
        IFS='|' read -r setting partition_name <<< "$job"
        run_job "$gpu_id" "$setting" "$partition_name"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for setting in "${SETTINGS[@]}"; do
    for partition_name in "${PARTITIONS[@]}"; do
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${setting}|${partition_name}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== Plain Dataset Generalization Baselines =========="
echo "gpus=${GPUS[*]}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "settings=${SETTINGS[*]}"
echo "partitions=${PARTITIONS[*]}"
echo "cifar10_rounds=${CIFAR10_ROUNDS:-500}, fmnist_rounds=${FMNIST_ROUNDS:-100}, tinyimagenet_rounds=${TINYIMAGENET_ROUNDS:-500}"
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
echo "Plain dataset generalization baselines complete (${job_count} jobs)"
