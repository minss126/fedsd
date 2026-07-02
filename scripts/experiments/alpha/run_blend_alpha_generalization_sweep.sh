#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# BYOT blend-alpha sweep across dataset/model/partition settings.
#
# Objective:
#   L = CE_teacher + (1-alpha) CE_branch + alpha KD_branch + beta_feat L_feat
#
# This script collects all outputs under LOG_ROOT. When a matching completed
# run already exists in older experiment folders, it symlinks the old
# log/pkl/json files into LOG_ROOT and only runs the missing combinations.
#
# Default matrix:
#   settings:
#     - CIFAR-100 / ResNet18-BYOT
#     - CIFAR-10 / ResNet18-BYOT
#     - CIFAR-100 / MobileNetV2-BYOT
#     - TinyImageNet / ResNet18-BYOT
#   partitions: iid, beta_0.5, beta_0.3, beta_0.1
#   alphas: 0.0, 0.3, 0.7, 1.0

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
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs_blend_alpha_generalization}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"

SETTINGS=(${SETTINGS_OVERRIDE:-cifar100_resnet18 cifar10_resnet18 cifar100_mobilenet tinyimagenet_resnet18})
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
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

dataset_flags() {
    local setting=$1
    case "$setting" in
        cifar10_resnet18)
            echo "--dataset cifar10 --datadir ./data"
            ;;
        cifar100_resnet18|cifar100_mobilenet)
            echo "--dataset cifar100 --datadir ./data"
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

model_flags() {
    local setting=$1
    case "$setting" in
        cifar10_resnet18|cifar100_resnet18|tinyimagenet_resnet18)
            echo "--model resnet18_byot --alg fedbyot"
            ;;
        cifar100_mobilenet)
            echo "--model mobilenet_byot --last_fc --alg fedbyot"
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

dataset_lr() {
    local setting=$1
    case "$setting" in
        tinyimagenet_resnet18)
            echo "${TINYIMAGENET_LR:-0.01}"
            ;;
        *)
            echo "${CIFAR_LR:-0.1}"
            ;;
    esac
}

existing_source_base() {
    local setting=$1
    local partition_name=$2
    local alpha_tag=$3

    case "${setting}:${partition_name}:${alpha_tag}" in
        cifar100_resnet18:iid:0p70)
            echo "logs_iid_fedavg_compare/iid/fedavg/fixed_alpha0p70"
            ;;
        cifar100_resnet18:iid:1p00)
            echo "logs_iid_fedavg_compare/iid/fedavg/fixed_alpha1p00"
            ;;
        cifar100_resnet18:beta_0.1:0p30|cifar100_resnet18:beta_0.3:0p30|cifar100_resnet18:beta_0.5:0p30)
            echo "logs_client_reliability_extended/${partition_name}/fedavg/fixed_alpha0p30"
            ;;
        cifar100_resnet18:beta_0.3:0p00)
            echo "logs_client_reliability_extended/beta_0.3/fedavg/fixed_alpha0p00"
            ;;
        cifar100_resnet18:beta_0.1:0p70|cifar100_resnet18:beta_0.3:0p70|cifar100_resnet18:beta_0.5:0p70)
            echo "logs_fixed_alpha_high/${partition_name}/fedavg/fixed_alpha0p70"
            ;;
        cifar100_resnet18:beta_0.1:1p00|cifar100_resnet18:beta_0.3:1p00|cifar100_resnet18:beta_0.5:1p00)
            echo "logs_fixed_alpha_high/${partition_name}/fedavg/fixed_alpha1p00"
            ;;
        cifar10_resnet18:iid:1p00|cifar10_resnet18:beta_0.5:1p00|cifar10_resnet18:beta_0.1:1p00)
            echo "logs_dataset_model_generalization/cifar10_resnet18/${partition_name}/fedavg/fedsd_alpha1p00"
            ;;
        cifar100_mobilenet:iid:1p00|cifar100_mobilenet:beta_0.5:1p00|cifar100_mobilenet:beta_0.1:1p00)
            echo "logs_dataset_model_generalization/cifar100_mobilenet/${partition_name}/fedavg/fedsd_alpha1p00"
            ;;
        tinyimagenet_resnet18:iid:1p00|tinyimagenet_resnet18:beta_0.5:1p00|tinyimagenet_resnet18:beta_0.1:1p00)
            echo "logs_dataset_model_generalization/tinyimagenet_resnet18/${partition_name}/fedavg/fedsd_alpha1p00"
            ;;
        *)
            echo ""
            ;;
    esac
}

link_existing_run() {
    local source_base=$1
    local dest_base=$2

    if ! has_completed_log "${source_base}.log"; then
        return 1
    fi

    mkdir -p "$(dirname "$dest_base")"
    for ext in log pkl json; do
        if [ -e "${source_base}.${ext}" ] && [ ! -e "${dest_base}.${ext}" ]; then
            ln -s "$(realpath "${source_base}.${ext}")" "${dest_base}.${ext}"
        fi
    done
    if [ -e "${source_base}_terminal.log" ] && [ ! -e "${dest_base}_terminal.log" ]; then
        ln -s "$(realpath "${source_base}_terminal.log")" "${dest_base}_terminal.log"
    fi
    return 0
}

try_reuse_existing() {
    local setting=$1
    local partition_name=$2
    local alpha_tag=$3
    local dest_base=$4
    local source_base

    if [ "$REUSE_EXISTING" != "1" ]; then
        return 1
    fi

    source_base="$(existing_source_base "$setting" "$partition_name" "$alpha_tag")"
    if [ -z "$source_base" ]; then
        return 1
    fi

    if link_existing_run "$source_base" "$dest_base"; then
        echo "[reuse] ${setting} | ${partition_name} | alpha=${alpha_tag} <- ${source_base}"
        return 0
    fi

    return 1
}

run_job() {
    local gpu_id=$1
    local setting=$2
    local partition_name=$3
    local alpha_val=$4
    local alpha_tag=$5
    local method_name="fedsd_alpha${alpha_tag}"
    local log_dir="${LOG_ROOT}/${setting}/${partition_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local dest_base="${log_dir}/${method_name}"
    local data_args
    local model_args
    local partition_args
    local learning_rate

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${setting} | ${partition_name} | ${method_name}"
        return
    fi

    if try_reuse_existing "$setting" "$partition_name" "$alpha_tag" "$dest_base"; then
        return
    fi

    data_args="$(dataset_flags "$setting")"
    model_args="$(model_flags "$setting")"
    partition_args="$(partition_flags "$partition_name")"
    learning_rate="$(dataset_lr "$setting")"

    echo "[GPU ${gpu_id}] start: ${setting} | ${partition_name} | ${method_name}"

    "${PYTHON_BIN}" main.py \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${learning_rate}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
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
        > "${log_dir}/${method_name}_terminal.log" 2>&1

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

echo "========== BYOT Blend Alpha Generalization Sweep =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "settings=${SETTINGS[*]}"
echo "partitions=${PARTITIONS[*]}"
echo "alphas=${ALPHAS[*]}"
echo "feature_beta=${FEATURE_BETA}, temp=${TEMP_VAL}, log_root=${LOG_ROOT}"
echo "reuse_existing=${REUSE_EXISTING}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"
echo "matrix_jobs=${job_count}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done

wait "${pids[@]}"
echo "BYOT blend alpha generalization sweep complete (${job_count} matrix entries; reused entries were symlinked)."
