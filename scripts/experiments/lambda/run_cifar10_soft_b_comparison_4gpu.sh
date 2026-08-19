#!/usr/bin/env bash

# CIFAR-10 confirmation experiment for the selected adaptive-lambda method.
#
# For each of IID, Dirichlet beta=0.5, and beta=0.1, this compares:
#   1. Plain FedAvg (no BYOT branches)
#   2. BYOT with constant lambda=0.3 (no lambda warm-up)
#   3. BYOT with the selected soft-b adaptive lambda
#
# The adaptive configuration is transferred unchanged from the CIFAR-100
# selection: T_KD=T_proxy=1, lambda_max=1, p=2, tau=.85, T_soft=.05,
# and a linear warm-up over the first half of training.  This is a
# confirmation run, not a CIFAR-10-specific hyperparameter search.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  GPUS_OVERRIDE="0 1 2 3" bash scripts/experiments/lambda/run_cifar10_soft_b_comparison_4gpu.sh

Optional overrides:
  SEED=0 USE_WANDB=0 SKIP_EXISTING=1
  ROUNDS=500 LOCAL_EPOCHS=5 BATCH_SIZE=64
  LOG_ROOT=logs/lambda/adaptive/logs_cifar10_soft_b_comparison
  DRY_RUN=1
EOF
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if (( ${#GPUS[@]} != 4 )); then
    echo "This launcher requires exactly four GPU ids; received: ${GPUS[*]:-(none)}" >&2
    exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"

FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
FIXED_LAMBDA="${FIXED_LAMBDA:-0.30}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.5}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"

LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_cifar10_soft_b_comparison}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.1})
METHODS=(plain fixed_lambda0p30 soft_b_adaptive)

WARMUP_ROUNDS="$(awk -v rounds="$ROUNDS" -v ratio="$WARMUP_RATIO" 'BEGIN { printf "%d", int(rounds * ratio + 0.5) }')"

WANDB_FLAGS=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
fi

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

method_name() {
    case "$1" in
        plain) printf '%s' plain ;;
        fixed_lambda0p30) printf 'fixed_lambda%s_tkd%s' "$(value_tag "$FIXED_LAMBDA")" "$(value_tag "$KD_TEMPERATURE")" ;;
        soft_b_adaptive)
            printf 'soft_b_tkd%s_lmax%s_warm%s_tau%s' \
                "$(value_tag "$KD_TEMPERATURE")" "$(value_tag "$LAMBDA_MAX")" \
                "$WARMUP_ROUNDS" "$(value_tag "$SOFT_TAU")"
            ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

has_completed_log() {
    local log_file="$1"
    [[ -f "$log_file" ]] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

append_adaptive_args() {
    CMD+=(
        --byot_alpha "$LAMBDA_MAX"
        --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00
        --byot_round_lambda_warmup "$WARMUP_ROUNDS"
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
}

run_job() {
    local gpu_id="$1" partition="$2" method="$3"
    local name log_dir log_file
    local -a PARTITION_FLAGS CMD

    name="$(method_name "$method")_r${ROUNDS}"
    log_dir="${LOG_ROOT}/${partition}/fedavg"
    log_file="${log_dir}/${name}.log"
    mkdir -p "$log_dir"

    if [[ "$SKIP_EXISTING" == "1" ]] && has_completed_log "$log_file"; then
        echo "[GPU ${gpu_id}] skip: cifar10 | ${partition} | ${method}"
        return 0
    fi

    mapfile -t PARTITION_FLAGS < <(partition_args "$partition")
    CMD=(
        "$PYTHON_BIN" main.py
        --dataset cifar10 --datadir ./data --in_channels 3 --num_classes 10
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$SEED" --device "cuda:${gpu_id}"
        --logdir "$LOG_ROOT" --log_file_name "${partition}/fedavg/${name}"
    )
    CMD+=("${PARTITION_FLAGS[@]}")

    if [[ "$method" == "plain" ]]; then
        CMD+=(--model resnet18 --alg fedavg)
    else
        CMD+=(
            --model resnet18_byot --alg fedbyot
            --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum
            --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
            --temperature "$KD_TEMPERATURE"
            --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
            --byot_proxy_temperature "$PROXY_TEMPERATURE"
        )
        if [[ "$method" == "fixed_lambda0p30" ]]; then
            # Intentionally no round schedule or warm-up for the fixed baseline.
            CMD+=(--byot_alpha "$FIXED_LAMBDA")
        else
            append_adaptive_args
        fi
    fi
    CMD+=("${WANDB_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: cifar10 | ${partition} | ${method} | R=${ROUNDS}, warm=${WARMUP_ROUNDS}"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run][GPU %s] ' "$gpu_id"
        printf '%q ' "${CMD[@]}"
        printf '\n'
        return 0
    fi
    if ! "${CMD[@]}" > "${log_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu_id}] failed: cifar10 | ${partition} | ${method}" >&2
        tail -30 "${log_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_log "$log_file"; then
        echo "[GPU ${gpu_id}] incomplete log: ${log_file}" >&2
        return 1
    fi
    echo "[GPU ${gpu_id}] complete: cifar10 | ${partition} | ${method}"
}

declare -a QUEUES LOADS
for ((i = 0; i < 4; i++)); do
    QUEUES[$i]=""
    LOADS[$i]=0
done

enqueue() {
    local partition="$1" method="$2" target=0 weight=100
    [[ "$method" == "soft_b_adaptive" ]] && weight=110
    for ((i = 1; i < 4; i++)); do
        (( LOADS[i] < LOADS[target] )) && target=$i
    done
    QUEUES[$target]+="${partition}|${method}"$'\n'
    LOADS[$target]=$((LOADS[$target] + weight))
}

# Schedule adaptive jobs first so the unavoidable three-job queue contains
# cheaper plain/fixed runs rather than three adaptive runs.
for method in soft_b_adaptive plain fixed_lambda0p30; do
    for partition in "${PARTITIONS[@]}"; do
        enqueue "$partition" "$method"
    done
done

run_queue() {
    local gpu_id="$1"
    shift
    local job partition method failed=0
    for job in "$@"; do
        [[ -z "$job" ]] && continue
        IFS='|' read -r partition method <<< "$job"
        if ! run_job "$gpu_id" "$partition" "$method"; then
            failed=1
        fi
    done
    return "$failed"
}

echo "========== CIFAR-10 soft-b adaptive-lambda comparison =========="
echo "jobs=9, gpus=${GPUS[*]}, seed=${SEED}"
echo "partitions=${PARTITIONS[*]} | methods=${METHODS[*]}"
echo "base=ResNet18, FedAvg, clients=${NUM_CLIENTS}, participation=${SAMPLE_FRACTION}, E=${LOCAL_EPOCHS}, R=${ROUNDS}"
echo "adaptive: T_KD=${KD_TEMPERATURE}, T_proxy=${PROXY_TEMPERATURE}, lambda_max=${LAMBDA_MAX}, p=${SKEW_POWER}, tau=${SOFT_TAU}, T_soft=${SOFT_TEMPERATURE}, warm=${WARMUP_ROUNDS}"
echo "fixed: lambda=${FIXED_LAMBDA}, no warm-up"
echo "logs=${LOG_ROOT}"
for ((i = 0; i < 4; i++)); do
    jobs_in_queue=$(printf '%s' "${QUEUES[$i]}" | sed '/^$/d' | wc -l)
    echo "GPU ${GPUS[$i]}: ${jobs_in_queue} jobs (relative load ${LOADS[$i]})"
done

pids=()
for ((i = 0; i < 4; i++)); do
    mapfile -t queue_jobs <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if (( failed )); then
    echo "One or more CIFAR-10 jobs failed; completed jobs were kept under ${LOG_ROOT}." >&2
    exit 1
fi
echo "CIFAR-10 soft-b adaptive-lambda comparison complete (9 jobs)."
