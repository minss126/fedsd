#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Alpha-trend screening for simple/small datasets and ImageNet100-64.
#
# Goal:
#   Check whether the CIFAR-10 trend, where larger branch KD alpha can hurt,
#   also appears on other simple datasets.
#
# Matrix:
#   settings:
#     - FMNIST / ResNet18-BYOT          : 100 rounds
#     - BloodMNIST / ResNet18-BYOT      : 100 rounds
#     - ImageNet100-64 / ResNet18-BYOT  : 300 rounds
#   partitions:
#     - iid, beta_0.5, beta_0.3, beta_0.1
#   alphas:
#     - 0.0, 0.3, 0.7, 1.0
#
# Total default jobs: 3 settings x 4 partitions x 4 alphas = 48.

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
FEATURE_BETA="${FEATURE_BETA:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs_simple_dataset_alpha_screen}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SETTINGS=(${SETTINGS_OVERRIDE:-fmnist_resnet18 bloodmnist_resnet18 imagenet100_64_resnet18})
PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})
ALPHAS=(${ALPHAS_OVERRIDE:-0.00:0p00 0.30:0p30 0.70:0p70 1.00:1p00})

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
        fmnist_resnet18)
            echo "--dataset fmnist --datadir ./data --in_channels 1 --num_classes 10"
            ;;
        bloodmnist_resnet18)
            echo "--dataset Bloodmnist --datadir ./data --in_channels 3 --num_classes 8"
            ;;
        imagenet100_64_resnet18)
            echo "--dataset imagenet100_64 --datadir /data/imagenet100_resized_64_png --in_channels 3 --num_classes 100"
            ;;
        *)
            echo "Unknown setting: ${setting}" >&2
            exit 1
            ;;
    esac
}

model_flags() {
    local setting=$1
    case "$setting" in
        fmnist_resnet18|bloodmnist_resnet18|imagenet100_64_resnet18)
            echo "--model resnet18_byot --alg fedbyot"
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
        fmnist_resnet18|bloodmnist_resnet18)
            echo "${SMALL_ROUNDS:-100}"
            ;;
        imagenet100_64_resnet18)
            echo "${IMAGENET100_ROUNDS:-300}"
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
        fmnist_resnet18|bloodmnist_resnet18)
            echo "${SMALL_LR:-0.1}"
            ;;
        imagenet100_64_resnet18)
            echo "${IMAGENET100_LR:-0.01}"
            ;;
        *)
            echo "Unknown setting: ${setting}" >&2
            exit 1
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
    local alpha_val=$4
    local alpha_tag=$5

    local rounds
    local lr
    local method_name="fedsd_alpha${alpha_tag}"
    local log_dir="${LOG_ROOT}/${setting}/${partition_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local data_args
    local model_args
    local partition_args

    rounds="$(setting_rounds "$setting")"
    lr="$(setting_lr "$setting")"

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file" "$rounds"; then
        echo "[skip] ${setting} | ${partition_name} | ${method_name}"
        return
    fi

    data_args="$(dataset_flags "$setting")"
    model_args="$(model_flags "$setting")"
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
        ${data_args} ${partition_args} ${model_args} \
        --kd_conf_threshold 0.0 \
        --byot_active_branches "1,2,3" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective blend \
        --byot_alpha "${alpha_val}" \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
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
        IFS='|' read -r setting partition_name alpha_val alpha_tag <<< "$job"
        run_job "$gpu_id" "$setting" "$partition_name" "$alpha_val" "$alpha_tag"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for setting in "${SETTINGS[@]}"; do
    for partition_name in "${PARTITIONS[@]}"; do
        for alpha_pair in "${ALPHAS[@]}"; do
            alpha_val="${alpha_pair%%:*}"
            alpha_tag="${alpha_pair##*:}"
            gpu_idx=$((job_count % NUM_GPUS))
            QUEUES[$gpu_idx]+="${setting}|${partition_name}|${alpha_val}|${alpha_tag}"$'\n'
            job_count=$((job_count + 1))
        done
    done
done

echo "========== Simple Dataset Alpha Screen =========="
echo "gpus=${GPUS[*]}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "settings=${SETTINGS[*]}"
echo "partitions=${PARTITIONS[*]}"
echo "alphas=${ALPHAS[*]}"
echo "small_rounds=${SMALL_ROUNDS:-100}, imagenet100_rounds=${IMAGENET100_ROUNDS:-300}"
echo "feature_beta=${FEATURE_BETA}, temp=${TEMP_VAL}, log_root=${LOG_ROOT}"
echo "skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"
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
echo "Simple dataset alpha screen complete (${job_count} jobs)"
