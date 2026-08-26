#!/usr/bin/env bash

# Leave-one-component-out ablation of the final soft-b adaptive lambda.
#
# Default stage:
#   dataset    = CIFAR-100
#   partition  = IID, beta=0.3
#   methods    = full, w/o warm-up, w/o reliability, w/o bias correction
#
# The already completed seed-0 full method is reused by default. Set
# REUSE_PRIOR_FULL=0 to run it again under LOG_ROOT.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  GPUS_OVERRIDE="0 1 2 3" \
    bash scripts/experiments/lambda/run_cifar100_component_ablation_stage1.sh

Defaults:
  PARTITIONS_OVERRIDE="iid beta_0.3"
  METHODS_OVERRIDE="wo_warmup wo_reliability wo_bias full"

Examples:
  PARTITIONS_OVERRIDE="beta_0.5 beta_0.1" GPUS_OVERRIDE="0 1 2 3" bash $0
  REUSE_PRIOR_FULL=0 GPUS_OVERRIDE="0 1 2 3" bash $0
  DRY_RUN=1 GPUS_OVERRIDE="0 1 2 3" bash $0
EOF
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if (( ${#GPUS[@]} == 0 )); then
    echo "Set GPUS_OVERRIDE to one or more GPU ids." >&2
    exit 1
fi
NUM_GPUS=${#GPUS[@]}

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.3})
METHODS=(${METHODS_OVERRIDE:-wo_warmup wo_reliability wo_bias full})

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
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
LAMBDA_WARMUP="${LAMBDA_WARMUP:-250}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"

LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_cifar100_component_ablation}"
PRIOR_FULL_ROOT="${PRIOR_FULL_ROOT:-logs/lambda/adaptive/logs_soft_adaptive_tuning_stage1}"
REUSE_PRIOR_FULL="${REUSE_PRIOR_FULL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

WANDB_FLAGS=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
fi

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

partition_flags() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

method_name() {
    local method="$1"
    case "$method" in
        wo_warmup)
            printf 'wo_warmup_r1_bsoft_tau%s' "$(value_tag "$SOFT_TAU")"
            ;;
        wo_reliability)
            printf 'wo_reliability_warm%s_bsoft_tau%s' "$LAMBDA_WARMUP" "$(value_tag "$SOFT_TAU")"
            ;;
        wo_bias)
            printf 'wo_bias_warm%s_r1' "$LAMBDA_WARMUP"
            ;;
        full)
            printf 'full_soft_b_warm%s_tau%s' "$LAMBDA_WARMUP" "$(value_tag "$SOFT_TAU")"
            ;;
        *) echo "Unknown method: $method" >&2; return 1 ;;
    esac
}

pkl_complete() {
    local path="$1"
    [[ -s "$path" ]] || return 1
    "$PYTHON_BIN" -c '
import pickle, sys
try:
    with open(sys.argv[1], "rb") as handle:
        payload = pickle.load(handle)
    values = payload.get("acc_global", [])
    complete = isinstance(values, (list, tuple)) and len(values) >= int(sys.argv[2])
except Exception:
    complete = False
raise SystemExit(0 if complete else 1)
' "$path" "$ROUNDS"
}

prior_full_path() {
    local partition="$1"
    printf '%s/%s/fedavg/soft_b_tkd%s_lmax%s_warm%s_tau%s.pkl' \
        "$PRIOR_FULL_ROOT" "$partition" \
        "$(value_tag "$KD_TEMPERATURE")" "$(value_tag "$LAMBDA_MAX")" \
        "$LAMBDA_WARMUP" "$(value_tag "$SOFT_TAU")"
}

current_path() {
    local partition="$1" method="$2" name
    name="$(method_name "$method")"
    printf '%s/%s/fedavg/%s.pkl' "$LOG_ROOT" "$partition" "$name"
}

job_is_needed() {
    local partition="$1" method="$2" output prior
    output="$(current_path "$partition" "$method")"
    if [[ "$SKIP_EXISTING" == "1" ]] && pkl_complete "$output"; then
        echo "[skip-current] ${partition} | ${method}: ${output}" >&2
        return 1
    fi
    if [[ "$method" == "full" && "$REUSE_PRIOR_FULL" == "1" ]]; then
        prior="$(prior_full_path "$partition")"
        if pkl_complete "$prior"; then
            echo "[reuse-prior] ${partition} | full: ${prior}" >&2
            return 1
        fi
    fi
    return 0
}

append_warmup_flags() {
    CMD+=(
        --byot_round_lambda_schedule linear
        --byot_round_lambda_min 0.00
        --byot_round_lambda_warmup "$LAMBDA_WARMUP"
    )
}

append_reliability_flags() {
    CMD+=(
        --byot_client_proxy teacher_label_prob
        --byot_client_alpha_min 0.00
        --byot_client_alpha_max 1.00
        --byot_client_alpha_mode multiply
        --byot_client_reliability_power 1.0
    )
}

append_bias_flags() {
    CMD+=(
        --byot_client_skew_proxy prediction_entropy
        --byot_client_skew_power "$SKEW_POWER"
        --byot_client_skew_min_scale 0.00
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau "$SOFT_TAU"
        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
    )
}

run_job() {
    local gpu_id="$1" partition="$2" method="$3" name log_dir output
    local -a PARTITION_FLAGS CMD

    name="$(method_name "$method")"
    log_dir="${LOG_ROOT}/${partition}/fedavg"
    output="${log_dir}/${name}.pkl"
    mkdir -p "$log_dir"
    mapfile -t PARTITION_FLAGS < <(partition_flags "$partition")

    CMD=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR"
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED"
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT"
        --log_file_name "${partition}/fedavg/${name}"
        --model resnet18_byot --alg fedbyot
        --byot_active_branches "1,2,3"
        --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only
        --byot_beta "$FEATURE_BETA"
        --byot_alpha "$LAMBDA_MAX"
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
        --alpha_min_scale 0.0
    )

    case "$method" in
        wo_warmup)
            append_reliability_flags
            append_bias_flags
            ;;
        wo_reliability)
            append_warmup_flags
            append_bias_flags
            ;;
        wo_bias)
            append_warmup_flags
            append_reliability_flags
            ;;
        full)
            append_warmup_flags
            append_reliability_flags
            append_bias_flags
            ;;
        *) echo "Unknown method: $method" >&2; return 1 ;;
    esac

    CMD+=("${PARTITION_FLAGS[@]}" "${WANDB_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: ${partition} | ${method}"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  command:'
        printf ' %q' "${CMD[@]}"
        printf '\n'
        return 0
    fi

    if ! "${CMD[@]}" > "${log_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu_id}] failed: ${partition} | ${method}" >&2
        tail -30 "${log_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! pkl_complete "$output"; then
        echo "[GPU ${gpu_id}] incomplete result: ${output}" >&2
        return 1
    fi
    echo "[GPU ${gpu_id}] complete: ${partition} | ${method}"
}

run_queue() {
    local gpu_id="$1"
    shift
    local job partition method
    for job in "$@"; do
        [[ -n "$job" ]] || continue
        IFS='|' read -r partition method <<< "$job"
        run_job "$gpu_id" "$partition" "$method"
    done
}

JOBS=()
for partition in "${PARTITIONS[@]}"; do
    for method in "${METHODS[@]}"; do
        if job_is_needed "$partition" "$method"; then
            JOBS+=("${partition}|${method}")
        fi
    done
done

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done
for ((i = 0; i < ${#JOBS[@]}; i++)); do
    gpu_idx=$((i % NUM_GPUS))
    QUEUES[$gpu_idx]+="${JOBS[$i]}"$'\n'
done

echo "========== CIFAR-100 Component Ablation: Stage 1 =========="
echo "gpus=${GPUS[*]}"
echo "partitions=${PARTITIONS[*]}"
echo "methods=${METHODS[*]}"
echo "new_jobs=${#JOBS[@]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "full=lambda_round * reliability * soft_bias"
echo "fixed_params=lambda_max=${LAMBDA_MAX}, T_kd=${KD_TEMPERATURE}, T_proxy=${PROXY_TEMPERATURE}, tau=${SOFT_TAU}, T_soft=${SOFT_TEMPERATURE}, p=${SKEW_POWER}"
echo "log_root=${LOG_ROOT}, reuse_prior_full=${REUSE_PRIOR_FULL}, dry_run=${DRY_RUN}"

if (( ${#JOBS[@]} == 0 )); then
    echo "No new jobs are required."
    exit 0
fi

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    mapfile -t queue_jobs <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if (( status != 0 )); then
    echo "One or more component-ablation jobs failed." >&2
    exit "$status"
fi

echo "CIFAR-100 component ablation stage 1 complete (${#JOBS[@]} new jobs)."
