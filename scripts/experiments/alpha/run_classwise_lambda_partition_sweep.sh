#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Class-wise KD-only BYOT lambda sweep for the remaining partitions.
#
# This extends the beta=0.1 pilot to:
#   iid, beta_0.5, beta_0.3
#
# Already-run beta=0.1 results are expected in:
#   logs/alpha/logs_classwise_lambda_beta01/
#
# Methods:
#   class_count:
#     reliability_{k,c} is proportional to local class count.
#   class_label_prob:
#     reliability_{k,c} is teacher true-label probability averaged per class.
#   class_label_prob_count:
#     reliability_{k,c} = class_label_prob_{k,c} * count_reliability_{k,c}.
#
# Default:
#   CIFAR-100 / ResNet18-BYOT / FedAvg / kd_only
#   GPU: cuda:3
#   total: 9 runs

GPUS=(${GPUS_OVERRIDE:-3})
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
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_classwise_lambda_partition_sweep}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ENVS=(${ENVS_OVERRIDE:-iid beta_0.5 beta_0.3})
PROXIES=(${PROXIES_OVERRIDE:-label_count teacher_label_prob teacher_label_prob_count})

WANDB_FLAGS=""
if [ "${USE_WANDB:-0}" = "1" ]; then
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
        *)
            echo "Unknown env: ${env_name}" >&2
            exit 1
            ;;
    esac
}

proxy_tag() {
    local proxy=$1
    case "$proxy" in
        label_count) echo "class_count" ;;
        teacher_label_prob) echo "class_label_prob" ;;
        teacher_correctness) echo "class_correctness" ;;
        teacher_label_prob_count) echo "class_label_prob_count" ;;
        *) echo "$proxy" ;;
    esac
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1
    local env_name=$2
    local proxy=$3
    local tag
    local method_name
    local log_dir
    local log_file
    local flags

    tag="$(proxy_tag "$proxy")"
    method_name="kd_only_${tag}_0_1"
    log_dir="${LOG_ROOT}/${env_name}/fedavg"
    log_file="${log_dir}/${method_name}.log"
    flags="$(env_flags "$env_name")"

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | ${method_name}"
        return
    fi

    echo "[GPU ${gpu_id}] start: ${env_name} | ${method_name} | class_proxy=${proxy}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${LR}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_active_branches "1,2,3" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only \
        --byot_alpha 1.00 \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        --byot_class_proxy "${proxy}" \
        --byot_class_alpha_min 0.00 \
        --byot_class_alpha_max 1.00 \
        --byot_class_alpha_mode map \
        --byot_class_count_smoothing 1.00 \
        ${flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${env_name} | ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")

    for job in "${jobs[@]}"; do
        [ -z "$job" ] && continue
        IFS='|' read -r env_name proxy <<< "$job"
        run_job "$gpu_id" "$env_name" "$proxy"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for env_name in "${ENVS[@]}"; do
    for proxy in "${PROXIES[@]}"; do
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${env_name}|${proxy}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== Class-wise KD Lambda Partition Sweep =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "envs=${ENVS[*]}"
echo "proxies=${PROXIES[*]}"
echo "feature_beta=${FEATURE_BETA}, temp=${TEMP_VAL}, log_root=${LOG_ROOT}"
echo "skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-0}"
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
echo "Class-wise KD lambda partition sweep complete (${job_count} jobs)"
