#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Standard fixed-lambda baseline for the selected CIFAR-100 configuration.
# Unlike the adaptive method, lambda is constant from round 0:
#   lambda^(t) = 1.
# No --byot_round_lambda_schedule flags are passed here.

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1 2 3\" bash $0"
    echo
    echo "Runs no-warm-up fixed lambda=1 on IID/beta={0.5,0.3,0.1}."
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -ne 4 ]; then
    echo "This launcher requires exactly four GPU ids; received: ${GPUS[*]}" >&2
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
FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.00}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_cifar100_fixed_lambda1_no_warmup}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ENVS=(${ENVS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})

WANDB_FLAGS=()
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
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
    local gpu_id=$1 env_name=$2
    local name log_dir log_file
    local -a PARTITION_ARGS CMD
    name="fixed_lambda1_tkd$(value_tag "$KD_TEMPERATURE")"
    log_dir="${LOG_ROOT}/${env_name}/fedavg"
    log_file="${log_dir}/${name}.log"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | ${name}"
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
        --byot_alpha 1.00
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
    )
    CMD+=("${PARTITION_ARGS[@]}")
    CMD+=("${WANDB_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: ${env_name} | ${name} | no warm-up"
    "${CMD[@]}" > "${log_dir}/${name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${env_name} | ${name}"
}

echo "========== CIFAR-100 Fixed Lambda=1 (No Warm-up) =========="
echo "gpus=${GPUS[*]}, envs=${ENVS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "T_KD=${KD_TEMPERATURE}, T_proxy=${PROXY_TEMPERATURE}, fixed_lambda=1.00"
echo "log_root=${LOG_ROOT}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"

pids=()
for i in "${!ENVS[@]}"; do
    run_job "${GPUS[$i]}" "${ENVS[$i]}" &
    pids+=("$!")
done

wait "${pids[@]}"
echo "CIFAR-100 fixed lambda=1 baseline complete (${#ENVS[@]} jobs)"
