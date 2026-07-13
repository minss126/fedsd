#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Simple adaptive lambda method comparison.
#
# This is a method-level probe, not a fine-grained ablation.
#
# Shared base:
#   CIFAR-100 / ResNet18-BYOT / FedAvg / KD-only / full B1+B2+B3
#   lambda_round(t) = lambda_max * min(1, t / warmup)
#   lambda_max=3, warmup=250
#
# Methods:
#   reliability_only:
#     lambda_k(t) = lambda_round(t) * E[p_T(y|x)]
#     Uses only teacher true-label probability on the client.
#
#   predskew_only:
#     lambda_k(t) = lambda_round(t) * H(mean_x p_T(.|x))^2
#     Uses only teacher prediction distribution entropy. This is label-free
#     after the normal forward pass, and measures prediction spread/skew.
#
#   reliability_predskew:
#     lambda_k(t) = lambda_round(t) * E[p_T(y|x)] * H(mean_x p_T(.|x))^2
#     Combines the two simplest useful signals.
#
#   server_drift_soft:
#     lambda(t) = lambda_round(t) / (1 + 0.1 * relative_update_drift(t-1))
#     A server-side global adaptation using client update drift, with a softer
#     tau than the previous server_drift candidate.
#
# Default jobs: 4 partitions * 4 methods = 16.

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
        PYTHON_BIN="python3"
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
LAMBDA_MAX="${LAMBDA_MAX:-3.00}"
WARMUP="${WARMUP:-250}"
SERVER_TAU_SOFT="${SERVER_TAU_SOFT:-0.1}"
SERVER_MIN_SCALE="${SERVER_MIN_SCALE:-0.0}"
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_adaptive_lambda_simple_methods}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ENVS=(${ENVS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})
METHODS=(${METHODS_OVERRIDE:-reliability_only predskew_only reliability_predskew server_drift_soft})

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

method_flags() {
    local method_name=$1
    case "$method_name" in
        reliability_only)
            echo "--byot_client_proxy teacher_label_prob --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00 --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0 --alpha_min_scale 0.0"
            ;;
        predskew_only)
            echo "--byot_client_skew_proxy prediction_entropy --byot_client_skew_min_scale 0.00 --byot_client_skew_power 2.0"
            ;;
        reliability_predskew)
            echo "--byot_client_proxy teacher_label_prob --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00 --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0 --alpha_min_scale 0.0 --byot_client_skew_proxy prediction_entropy --byot_client_skew_min_scale 0.00 --byot_client_skew_power 2.0"
            ;;
        server_drift_soft)
            echo "--byot_server_lambda_adaptive --byot_server_lambda_tau ${SERVER_TAU_SOFT} --byot_server_lambda_min_scale ${SERVER_MIN_SCALE} --log_client_drift --drift_log_interval 1"
            ;;
        *)
            echo "Unknown method: ${method_name}" >&2
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
    local env_name=$2
    local method_name=$3
    local log_dir="${LOG_ROOT}/${env_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local partition_args
    local adaptive_args

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | ${method_name}"
        return
    fi

    partition_args="$(env_flags "$env_name")"
    adaptive_args="$(method_flags "$method_name")"

    echo "[GPU ${gpu_id}] start: ${env_name} | ${method_name}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --datadir ./data \
        --num_classes 100 \
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
        --byot_alpha "${LAMBDA_MAX}" \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        --byot_round_lambda_schedule linear \
        --byot_round_lambda_min 0.00 \
        --byot_round_lambda_warmup "${WARMUP}" \
        ${adaptive_args} ${partition_args} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${env_name} | ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")

    for job in "${jobs[@]}"; do
        [ -z "$job" ] && continue
        IFS='|' read -r env_name method_name <<< "$job"
        run_job "$gpu_id" "$env_name" "$method_name"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for env_name in "${ENVS[@]}"; do
    for method_name in "${METHODS[@]}"; do
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${env_name}|${method_name}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== Simple Adaptive Lambda Methods =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, seed=${SEED}, jobs=${job_count}"
echo "envs=${ENVS[*]}"
echo "methods=${METHODS[*]}"
echo "lambda_max=${LAMBDA_MAX}, warmup=${WARMUP}, server_tau_soft=${SERVER_TAU_SOFT}"
echo "log_root=${LOG_ROOT}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done

wait "${pids[@]}"
echo "Simple adaptive lambda methods complete (${job_count} jobs)"
