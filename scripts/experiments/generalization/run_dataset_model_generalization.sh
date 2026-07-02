#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Dataset/backbone generalization sweep.
#
# Isolates one generalization axis at a time:
#   - Dataset: CIFAR-10 and TinyImageNet with ResNet18
#   - Backbone: CIFAR-100 with MobileNetV2
#
# Each setting compares plain FedAvg against full-branch BYOT/FedSD under:
#   IID, Dirichlet beta=0.5, and Dirichlet beta=0.1.
#
# Total: 3 experiment groups x 3 partitions x 2 methods = 18 runs.
# Runs are executed in batches of four to use GPUs 0-3.

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if [ "${#GPUS[@]}" -lt 4 ]; then
    echo "This script expects four GPUs. Set GPUS_OVERRIDE with four GPU ids." >&2
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
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BYOT_ALPHA_VAL="${BYOT_ALPHA_VAL:-1.0}"
BYOT_BETA_VAL="${BYOT_BETA_VAL:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs_dataset_model_generalization}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

partition_flags() {
    local partition_name=$1
    case "$partition_name" in
        iid)
            echo "--partition iid"
            ;;
        beta_0.5)
            echo "--partition noniid --beta 0.5"
            ;;
        beta_0.1)
            echo "--partition noniid --beta 0.1"
            ;;
        *)
            echo "Unknown partition: ${partition_name}" >&2
            exit 1
            ;;
    esac
}

dataset_flags() {
    local dataset=$1
    case "$dataset" in
        cifar10)
            echo "--dataset cifar10 --datadir ./data"
            ;;
        cifar100)
            echo "--dataset cifar100 --datadir ./data"
            ;;
        tinyimagenet)
            echo "--dataset tinyimagenet --datadir ./data/tiny-imagenet-200"
            ;;
        *)
            echo "Unknown dataset: ${dataset}" >&2
            exit 1
            ;;
    esac
}

dataset_lr() {
    local dataset=$1
    case "$dataset" in
        tinyimagenet)
            echo "${TINYIMAGENET_LR:-0.01}"
            ;;
        *)
            echo "${CIFAR_LR:-0.1}"
            ;;
    esac
}

model_flags() {
    local backbone=$1
    local method=$2
    case "${backbone}:${method}" in
        resnet18:plain)
            echo "--model resnet18 --alg fedavg"
            ;;
        resnet18:sd)
            echo "--model resnet18_byot --alg fedbyot"
            ;;
        mobilenet:plain)
            echo "--model mobilenet --last_fc --alg fedavg"
            ;;
        mobilenet:sd)
            echo "--model mobilenet_byot --alg fedbyot"
            ;;
        *)
            echo "Unknown backbone/method: ${backbone}/${method}" >&2
            exit 1
            ;;
    esac
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1
    local dataset=$2
    local backbone=$3
    local partition_name=$4
    local method=$5

    local experiment="${dataset}_${backbone}"
    local method_name="plain"
    if [ "$method" = "sd" ]; then
        method_name="fedsd_alpha1p00"
    fi

    local log_dir="${LOG_ROOT}/${experiment}/${partition_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local data_args
    local partition_args
    local model_args
    local learning_rate
    data_args="$(dataset_flags "$dataset")"
    partition_args="$(partition_flags "$partition_name")"
    model_args="$(model_flags "$backbone" "$method")"
    learning_rate="$(dataset_lr "$dataset")"

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${experiment} | ${partition_name} | ${method_name}"
        return
    fi

    echo "[GPU ${gpu_id}] start: ${experiment} | ${partition_name} | ${method_name}"

    local sd_args=""
    if [ "$method" = "sd" ]; then
        sd_args="--kd_conf_threshold 0.0 --byot_alpha ${BYOT_ALPHA_VAL} --byot_beta ${BYOT_BETA_VAL} --temperature ${TEMP_VAL}"
    fi

    "${PYTHON_BIN}" main.py \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${learning_rate}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${experiment}/${partition_name}/fedavg/${method_name}" \
        ${data_args} ${partition_args} ${model_args} ${sd_args} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${experiment} | ${partition_name} | ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")

    for job in "${jobs[@]}"; do
        IFS='|' read -r dataset backbone partition_name method <<< "$job"
        run_job "$gpu_id" "$dataset" "$backbone" "$partition_name" "$method"
    done
}

echo "========== Dataset/Model Generalization Sweep =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "cifar_lr=${CIFAR_LR:-0.1}, tinyimagenet_lr=${TINYIMAGENET_LR:-0.01}, workers=${NUM_WORKERS}"
echo "alpha=${BYOT_ALPHA_VAL}, byot_beta=${BYOT_BETA_VAL}, temp=${TEMP_VAL}"
echo "log_root=${LOG_ROOT}, wandb=${USE_WANDB:-1}"

# Independent per-GPU queues avoid a global batch barrier. The six long
# TinyImageNet runs are distributed first; shorter CIFAR runs fill the tails.
run_queue "${GPUS[0]}" \
    "tinyimagenet|resnet18|iid|plain" \
    "tinyimagenet|resnet18|beta_0.1|plain" \
    "cifar10|resnet18|iid|plain" &
queue0_pid=$!

run_queue "${GPUS[1]}" \
    "tinyimagenet|resnet18|iid|sd" \
    "tinyimagenet|resnet18|beta_0.1|sd" \
    "cifar10|resnet18|iid|sd" &
queue1_pid=$!

run_queue "${GPUS[2]}" \
    "tinyimagenet|resnet18|beta_0.5|plain" \
    "cifar100|mobilenet|iid|plain" \
    "cifar100|mobilenet|beta_0.5|plain" \
    "cifar100|mobilenet|beta_0.1|plain" \
    "cifar10|resnet18|beta_0.5|plain" \
    "cifar10|resnet18|beta_0.1|plain" &
queue2_pid=$!

run_queue "${GPUS[3]}" \
    "tinyimagenet|resnet18|beta_0.5|sd" \
    "cifar100|mobilenet|iid|sd" \
    "cifar100|mobilenet|beta_0.5|sd" \
    "cifar100|mobilenet|beta_0.1|sd" \
    "cifar10|resnet18|beta_0.5|sd" \
    "cifar10|resnet18|beta_0.1|sd" &
queue3_pid=$!

wait "$queue0_pid" "$queue1_pid" "$queue2_pid" "$queue3_pid"

echo "Dataset/model generalization sweep complete (18 runs)"
