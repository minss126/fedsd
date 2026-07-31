#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Tests the prediction-entropy client-skew proxy after separating its
# measurement temperature from the branch-KD temperature.
#
# Both methods use the same adaptive client reliability:
#   lambda_k = lambda_t * r_k * correction(b_k)
#   r_k = mean_i p_proxy(y_i | x_i)
#   b_k = H(mean_i p_proxy(. | x_i)) / log(C)
#
# The training loss uses T_KD=0.5, while every proxy uses T_proxy=1.0.
# `byot_log_prediction_entropy_components` additionally saves, for every
# selected client and round:
#   b_k = H(mean p_i)/log(C)
#   u_k = mean_i H(p_i)/log(C)
#   d_k = b_k - u_k
# By default lambda uses b_k; set SKEW_PROXY=prediction_mutual_info to use
# d_k as a later, matched ablation.
#
# Default matrix: 2 corrections x 3 partitions = 6 runs.
# On two GPUs it should take roughly 6.5-8 hours at 500 rounds.

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1\" bash $0"
    echo
    echo "Defaults: IID/beta_0.5/beta_0.1, fixed lambda_max=2, warm-up=250"
    echo "Methods: plain b^2 correction and soft relaxation (tau=0.85)."
    echo "Overrides: ENVS_OVERRIDE, ROUNDS, LOCAL_EPOCHS, LOG_ROOT, USE_WANDB, SKIP_EXISTING"
    exit 0
fi

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
KD_TEMPERATURE="${KD_TEMPERATURE:-0.5}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-2.0}"
WARMUP="${WARMUP:-250}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
SKEW_PROXY="${SKEW_PROXY:-prediction_entropy}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_proxy_temperature_entropy_diagnostics}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ENVS=(${ENVS_OVERRIDE:-iid beta_0.5 beta_0.1})
METHODS=(plain_b2 soft_relax)

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

env_flags() {
    case "$1" in
        iid) echo "--partition iid" ;;
        beta_0.5) echo "--partition noniid --beta 0.5" ;;
        beta_0.1) echo "--partition noniid --beta 0.1" ;;
        *) echo "Unknown environment: $1" >&2; exit 1 ;;
    esac
}

method_flags() {
    case "$1" in
        plain_b2)
            echo "--byot_client_skew_correction_mode multiply"
            ;;
        soft_relax)
            echo "--byot_client_skew_correction_mode soft_relax --byot_client_skew_soft_tau ${SOFT_TAU} --byot_client_skew_soft_temperature ${SOFT_TEMPERATURE}"
            ;;
        *) echo "Unknown method: $1" >&2; exit 1 ;;
    esac
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1 env_name=$2 method=$3
    local log_dir="${LOG_ROOT}/${env_name}/fedavg"
    local log_file="${log_dir}/${method}.log"
    local partition_args correction_args
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | ${method}"
        return
    fi

    partition_args="$(env_flags "$env_name")"
    correction_args="$(method_flags "$method")"
    echo "[GPU ${gpu_id}] start: ${env_name} | ${method}"

    "$PYTHON_BIN" main.py \
        --dataset cifar100 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED" \
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT" \
        --log_file_name "${env_name}/fedavg/${method}" \
        --model resnet18_byot --alg fedbyot \
        --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA" \
        --byot_alpha "$LAMBDA_MAX" \
        --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00 \
        --byot_round_lambda_warmup "$WARMUP" \
        --temperature "$KD_TEMPERATURE" \
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE" \
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE" \
        --byot_proxy_temperature "$PROXY_TEMPERATURE" \
        --byot_log_prediction_entropy_components \
        --alpha_min_scale 0.0 \
        --byot_client_proxy teacher_label_prob \
        --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00 \
        --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0 \
        --byot_client_skew_proxy "$SKEW_PROXY" \
        --byot_client_skew_power "$SKEW_POWER" \
        --byot_client_skew_min_scale 0.00 \
        ${correction_args} ${partition_args} ${WANDB_FLAGS} \
        > "${log_dir}/${method}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${env_name} | ${method}"
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done

job_count=0
for method in "${METHODS[@]}"; do
    for env_name in "${ENVS[@]}"; do
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${env_name}|${method}"$'\n'
        job_count=$((job_count + 1))
    done
done

run_queue() {
    local gpu_id=$1
    shift
    local job env_name method
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r env_name method <<< "$job"
        run_job "$gpu_id" "$env_name" "$method"
    done
}

echo "========== Proxy Temperature + Entropy Diagnostics =========="
echo "gpus=${GPUS[*]}, jobs=${job_count}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "envs=${ENVS[*]}, methods=${METHODS[*]}"
echo "KD temperature=${KD_TEMPERATURE}, proxy temperature=${PROXY_TEMPERATURE}"
echo "lambda_max=${LAMBDA_MAX}, warmup=${WARMUP}, skew_proxy=${SKEW_PROXY}, skew_power=${SKEW_POWER}, soft_tau=${SOFT_TAU}"
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
echo "Proxy-temperature entropy diagnostic complete (${job_count} jobs)"
