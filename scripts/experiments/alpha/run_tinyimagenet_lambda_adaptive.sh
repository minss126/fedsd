#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# TinyImageNet / ResNet18-BYOT lambda comparison.
#
# Complements the existing TinyImageNet alpha/plain logs with:
#   - fixed lambda=3 under kd_only branch objective
#   - adaptive lambda using label_prob x client_pred_entropy^2
#
# Output layout:
#   logs/alpha/logs_tinyimagenet_lambda_adaptive/<partition>/fedavg/

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
        PYTHON_BIN="python3"
    fi
fi

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LAMBDA_MAX="${LAMBDA_MAX:-3.00}"
WARMUP="${WARMUP:-250}"
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_tinyimagenet_lambda_adaptive}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})
METHODS=(${METHODS_OVERRIDE:-fixed_lambda3 label_prob_x_client_pred_entropy_p2})

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

partition_flags() {
    case "$1" in
        iid) echo "--partition iid" ;;
        beta_0.1) echo "--partition noniid --beta 0.1" ;;
        beta_0.3) echo "--partition noniid --beta 0.3" ;;
        beta_0.5) echo "--partition noniid --beta 0.5" ;;
        *) echo "Unknown partition: $1" >&2; exit 1 ;;
    esac
}

method_flags() {
    case "$1" in
        fixed_lambda3)
            echo ""
            ;;
        label_prob_x_client_pred_entropy_p2)
            echo "--byot_round_lambda_schedule linear --byot_round_lambda_min 0.00 --byot_round_lambda_warmup ${WARMUP} --byot_client_proxy teacher_label_prob --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00 --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0 --alpha_min_scale 0.0 --byot_client_skew_proxy prediction_entropy --byot_client_skew_min_scale 0.00 --byot_client_skew_power 2.0"
            ;;
        *) echo "Unknown method: $1" >&2; exit 1 ;;
    esac
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1
    local partition_name=$2
    local method_name=$3
    local log_dir="${LOG_ROOT}/${partition_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local partition_args
    local extra_args

    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${partition_name} | ${method_name}"
        return
    fi

    partition_args="$(partition_flags "$partition_name")"
    extra_args="$(method_flags "$method_name")"

    echo "[GPU ${gpu_id}] start: tinyimagenet_resnet18 | ${partition_name} | ${method_name}"

    "${PYTHON_BIN}" main.py \
        --dataset tinyimagenet --datadir ./data/tiny-imagenet-200 \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${LR}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${partition_name}/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_active_branches "1,2,3" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only \
        --byot_alpha "${LAMBDA_MAX}" \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        ${extra_args} ${partition_args} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: tinyimagenet_resnet18 | ${partition_name} | ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")

    for job in "${jobs[@]}"; do
        [ -z "$job" ] && continue
        IFS='|' read -r partition_name method_name <<< "$job"
        run_job "$gpu_id" "$partition_name" "$method_name"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for partition_name in "${PARTITIONS[@]}"; do
    for method_name in "${METHODS[@]}"; do
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${partition_name}|${method_name}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== TinyImageNet Lambda/Adaptive Sweep =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "partitions=${PARTITIONS[*]}"
echo "methods=${METHODS[*]}"
echo "lambda_max=${LAMBDA_MAX}, feature_beta=${FEATURE_BETA}, temp=${TEMP_VAL}, warmup=${WARMUP}"
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
echo "TinyImageNet lambda/adaptive sweep complete (${job_count} jobs)"
