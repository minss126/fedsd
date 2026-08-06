#!/bin/bash

set -euo pipefail

# Adds lambda=0.5 and 0.7 to the existing CIFAR-100 fixed-lambda analysis.
# The command is intentionally identical to the compact T_KD=1 grid:
# ResNet18-BYOT, FedBYOT, KD-only branches, constant lambda, no warm-up.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1 2 3\" bash $0"
    echo "Adds fixed lambda={0.5,0.7} for IID/beta={0.5,0.3,0.1}."
    echo "Expected wall time on four GPUs: about 4-5 hours."
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if [ "${#GPUS[@]}" -ne 4 ]; then
    echo "This launcher requires exactly four GPU ids; received: ${GPUS[*]}" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -x venv/bin/python ]; then PYTHON_BIN="venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.00}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/analysis/logs_cifar100_fixed_lambda_t1_compact}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
LAMBDAS=(${LAMBDAS_OVERRIDE:-0.5 0.7})
ENVS=(${ENVS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})

WANDB_FLAGS=()
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    [ -n "${WANDB_ENTITY:-}" ] && WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
fi

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

env_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown environment: $1" >&2; exit 1 ;;
    esac
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1 lambda=$2 env_name=$3
    local name log_dir log_file
    local -a PARTITION_ARGS CMD
    name="fixed_lambda$(value_tag "$lambda")_tkd$(value_tag "$KD_TEMPERATURE")"
    log_dir="${LOG_ROOT}/${env_name}/fedavg"
    log_file="${log_dir}/${name}.log"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | lambda=${lambda}"
        return
    fi

    mapfile -t PARTITION_ARGS < <(env_args "$env_name")
    CMD=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data
        --n_clients 100 --sample_fraction 0.1
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED"
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT"
        --log_file_name "${env_name}/fedavg/${name}"
        --model resnet18_byot --alg fedbyot
        --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
        --byot_alpha "$lambda"
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
    )
    CMD+=("${PARTITION_ARGS[@]}" "${WANDB_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: ${env_name} | lambda=${lambda} | no warm-up"
    "${CMD[@]}" > "${log_dir}/${name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${env_name} | lambda=${lambda}"
}

run_queue() {
    local gpu_id=$1
    shift
    local job lambda env_name
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r lambda env_name <<< "$job"
        run_job "$gpu_id" "$lambda" "$env_name"
    done
}

declare -a QUEUES
for ((i = 0; i < ${#GPUS[@]}; i++)); do QUEUES[$i]=""; done
job_count=0
for lambda in "${LAMBDAS[@]}"; do
    for env_name in "${ENVS[@]}"; do
        gpu_idx=$((job_count % ${#GPUS[@]}))
        QUEUES[$gpu_idx]+="${lambda}|${env_name}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== CIFAR-100 Fixed Lambda Midpoint Grid =========="
echo "gpus=${GPUS[*]}, lambdas=${LAMBDAS[*]}, envs=${ENVS[*]}, rounds=${ROUNDS}, seed=${SEED}"
echo "T_KD=${KD_TEMPERATURE}, constant lambda, no warm-up, log_root=${LOG_ROOT}"

pids=()
for ((i = 0; i < ${#GPUS[@]}; i++)); do
    mapfile -t queue_jobs <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
    pids+=("$!")
done
wait "${pids[@]}"
echo "Fixed midpoint grid complete (${job_count} jobs)."
