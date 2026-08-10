#!/bin/bash

set -euo pipefail

# One-factor protocol extension for the finalized adaptive-lambda method.
# Supported axes:
#   local_epochs:      E={1,10}, with participation fixed at 0.1
#   participation:     C={0.05,0.20}, with local epochs fixed at 5
# Existing E=5/C=0.1 results are the shared reference and are not repeated.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: AXIS_OVERRIDE=<local_epochs|participation> GPUS_OVERRIDE=\"0 1 2 3\" bash $0"
    echo
    echo "Methods: plain, fixed lambda={0.1,0.3}, soft-b adaptive"
    echo "Partitions: IID, beta={0.5,0.3,0.1}; dataset=CIFAR-100; rounds=500."
    echo "Adaptive warm-up is computed as 0.5 x rounds (250 rounds here)."
    exit 0
fi

AXIS="${AXIS_OVERRIDE:?Set AXIS_OVERRIDE to local_epochs or participation.}"
GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "Set GPUS_OVERRIDE to one or more GPU ids." >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -x venv/bin/python ]; then PYTHON_BIN="venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
BASE_EPOCHS="${BASE_EPOCHS:-5}"
BASE_SAMPLE_FRACTION="${BASE_SAMPLE_FRACTION:-0.1}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.00}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.00}"
LAMBDA_WARMUP_RATIO="${LAMBDA_WARMUP_RATIO:-0.5}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_cifar100_protocol_extensions}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ENVS=(${ENVS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})
METHODS=(${METHODS_OVERRIDE:-plain fixed_lambda0p10 fixed_lambda0p30 soft_b_adaptive})

case "$AXIS" in
    local_epochs)
        VALUES=(${VALUES_OVERRIDE:-1 10})
        ;;
    participation)
        VALUES=(${VALUES_OVERRIDE:-0.05 0.20})
        ;;
    *)
        echo "Unknown axis: $AXIS" >&2
        exit 1
        ;;
esac

ADAPTIVE_WARMUP=$(awk -v rounds="$ROUNDS" -v ratio="$LAMBDA_WARMUP_RATIO" 'BEGIN { printf "%d", int(rounds * ratio + 0.5) }')

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

axis_label() {
    case "$AXIS" in
        local_epochs) echo "e$1" ;;
        participation) echo "c$(value_tag "$1")" ;;
    esac
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

method_name() {
    case "$1" in
        plain) echo "plain" ;;
        fixed_lambda0p10) echo "fixed_lambda0p10_tkd$(value_tag "$KD_TEMPERATURE")" ;;
        fixed_lambda0p30) echo "fixed_lambda0p30_tkd$(value_tag "$KD_TEMPERATURE")" ;;
        soft_b_adaptive) echo "soft_b_tkd$(value_tag "$KD_TEMPERATURE")_lmax$(value_tag "$LAMBDA_MAX")_warm${ADAPTIVE_WARMUP}_tau$(value_tag "$SOFT_TAU")" ;;
        *) echo "Unknown method: $1" >&2; exit 1 ;;
    esac
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1 value=$2 env_name=$3 method=$4
    local epochs="$BASE_EPOCHS" sample_fraction="$BASE_SAMPLE_FRACTION"
    local label name log_dir log_file
    local -a PARTITION_ARGS CMD

    case "$AXIS" in
        local_epochs) epochs="$value" ;;
        participation) sample_fraction="$value" ;;
    esac
    label="$(axis_label "$value")"
    name="$(method_name "$method")_r${ROUNDS}"
    log_dir="${LOG_ROOT}/${AXIS}/${label}/${env_name}/fedavg"
    log_file="${log_dir}/${name}.log"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${AXIS}=${value} | ${env_name} | ${method}"
        return
    fi

    mapfile -t PARTITION_ARGS < <(env_args "$env_name")
    CMD=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data --num_classes 100
        --n_clients 100 --sample_fraction "$sample_fraction"
        --epochs "$epochs" --lr "$LR" --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED"
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT"
        --log_file_name "${AXIS}/${label}/${env_name}/fedavg/${name}"
    )
    CMD+=("${PARTITION_ARGS[@]}")

    case "$method" in
        plain)
            CMD+=(--model resnet18 --alg fedavg)
            ;;
        fixed_lambda0p10|fixed_lambda0p30|soft_b_adaptive)
            CMD+=(
                --model resnet18_byot --alg fedbyot
                --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum
                --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
                --temperature "$KD_TEMPERATURE"
                --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
                --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
                --byot_proxy_temperature "$PROXY_TEMPERATURE"
            )
            case "$method" in
                fixed_lambda0p10) CMD+=(--byot_alpha 0.10) ;;
                fixed_lambda0p30) CMD+=(--byot_alpha 0.30) ;;
                soft_b_adaptive)
                    CMD+=(
                        --byot_alpha "$LAMBDA_MAX"
                        --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00
                        --byot_round_lambda_warmup "$ADAPTIVE_WARMUP"
                        --alpha_min_scale 0.0
                        --byot_client_proxy teacher_label_prob
                        --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00
                        --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
                        --byot_client_skew_proxy prediction_entropy
                        --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.00
                        --byot_client_skew_correction_mode soft_relax
                        --byot_client_skew_soft_tau "$SOFT_TAU"
                        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
                    )
                    ;;
            esac
            ;;
    esac
    CMD+=("${WANDB_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: ${AXIS}=${value} | ${env_name} | ${method} | E=${epochs}, C=${sample_fraction}"
    "${CMD[@]}" > "${log_dir}/${name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${AXIS}=${value} | ${env_name} | ${method}"
}

run_queue() {
    local gpu_id=$1
    shift
    local job value env_name method
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r value env_name method <<< "$job"
        run_job "$gpu_id" "$value" "$env_name" "$method"
    done
}

declare -a QUEUES
for ((i = 0; i < ${#GPUS[@]}; i++)); do QUEUES[$i]=""; done
job_count=0
for value in "${VALUES[@]}"; do
    for env_name in "${ENVS[@]}"; do
        for method in "${METHODS[@]}"; do
            gpu_idx=$((job_count % ${#GPUS[@]}))
            QUEUES[$gpu_idx]+="${value}|${env_name}|${method}"$'\n'
            job_count=$((job_count + 1))
        done
    done
done

echo "========== CIFAR-100 Protocol Extension =========="
echo "axis=${AXIS}, values=${VALUES[*]}, gpus=${GPUS[*]}, jobs=${job_count}"
echo "rounds=${ROUNDS}, E_base=${BASE_EPOCHS}, C_base=${BASE_SAMPLE_FRACTION}, adaptive_warmup=${ADAPTIVE_WARMUP} (${LAMBDA_WARMUP_RATIO}R)"
echo "envs=${ENVS[*]}, methods=${METHODS[*]}, seed=${SEED}, log_root=${LOG_ROOT}"

pids=()
for ((i = 0; i < ${#GPUS[@]}; i++)); do
    mapfile -t queue_jobs <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
    pids+=("$!")
done
wait "${pids[@]}"
echo "CIFAR-100 ${AXIS} extension complete (${job_count} jobs)."
