#!/usr/bin/env bash

# Diagnostic CIFAR-10 screen for dataset-dependent soft-b calibration.
# Existing tau=.85 results are reused; this runs only tau={.45,.65} on
# beta={.5,.1}.  Four jobs are mapped one-to-one onto four GPUs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if (( ${#GPUS[@]} != 4 )); then
    echo "Provide exactly four GPU ids through GPUS_OVERRIDE." >&2
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
FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-250}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
TAUS=(${TAUS_OVERRIDE:-0.45 0.65})
PARTITIONS=(${PARTITIONS_OVERRIDE:-beta_0.5 beta_0.1})
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_cifar10_soft_tau_quick_screen}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

WANDB_FLAGS=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
fi

tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

partition_args() {
    case "$1" in
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

pkl_complete() {
    local path="$1"
    [[ -s "$path" ]] || return 1
    "$PYTHON_BIN" -c '
import pickle, sys
try:
    with open(sys.argv[1], "rb") as f:
        values = pickle.load(f).get("acc_global", [])
    ok = isinstance(values, (list, tuple)) and len(values) >= int(sys.argv[2])
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
' "$path" "$ROUNDS"
}

run_job() {
    local gpu_id="$1" partition="$2" tau="$3"
    local name log_dir result
    local -a PARTITION_FLAGS CMD

    name="soft_b_tkd$(tag "$KD_TEMPERATURE")_lmax$(tag "$LAMBDA_MAX")_warm${WARMUP_ROUNDS}_tau$(tag "$tau")_r${ROUNDS}"
    log_dir="${LOG_ROOT}/${partition}/fedavg"
    result="${log_dir}/${name}.pkl"
    mkdir -p "$log_dir"

    if [[ "$SKIP_EXISTING" == "1" ]] && pkl_complete "$result"; then
        echo "[GPU ${gpu_id}] skip: ${partition} | tau=${tau}"
        return 0
    fi

    mapfile -t PARTITION_FLAGS < <(partition_args "$partition")
    CMD=(
        "$PYTHON_BIN" main.py
        --dataset cifar10 --datadir ./data --in_channels 3 --num_classes 10
        --n_clients 100 --sample_fraction 0.1
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$SEED" --device "cuda:${gpu_id}"
        --logdir "$LOG_ROOT" --log_file_name "${partition}/fedavg/${name}"
        --model resnet18_byot --alg fedbyot
        --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
        --byot_alpha "$LAMBDA_MAX"
        --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00
        --byot_round_lambda_warmup "$WARMUP_ROUNDS" --alpha_min_scale 0.0
        --byot_client_proxy teacher_label_prob
        --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00
        --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
        --byot_client_skew_proxy prediction_entropy
        --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.00
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau "$tau"
        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
    )
    CMD+=("${PARTITION_FLAGS[@]}" "${WANDB_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: ${partition} | tau=${tau} | R=${ROUNDS}, warm=${WARMUP_ROUNDS}"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run] '
        printf '%q ' "${CMD[@]}"
        printf '\n'
        return 0
    fi
    if ! "${CMD[@]}" > "${log_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu_id}] failed: ${partition} | tau=${tau}" >&2
        tail -25 "${log_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! pkl_complete "$result"; then
        echo "[GPU ${gpu_id}] incomplete result: ${result}" >&2
        return 1
    fi
    echo "[GPU ${gpu_id}] complete: ${partition} | tau=${tau}"
}

JOBS=()
for tau in "${TAUS[@]}"; do
    for partition in "${PARTITIONS[@]}"; do
        JOBS+=("${partition}|${tau}")
    done
done
if (( ${#JOBS[@]} != 4 )); then
    echo "This quick launcher expects exactly four jobs; received ${#JOBS[@]}." >&2
    exit 1
fi

echo "========== CIFAR-10 quick soft-tau screen =========="
echo "partitions=${PARTITIONS[*]}, tau=${TAUS[*]}, gpus=${GPUS[*]}"
echo "fixed: T_KD=${KD_TEMPERATURE}, T_proxy=${PROXY_TEMPERATURE}, lmax=${LAMBDA_MAX}, p=${SKEW_POWER}, T_soft=${SOFT_TEMPERATURE}, warm=${WARMUP_ROUNDS}"
echo "logs=${LOG_ROOT}"

pids=()
for ((i = 0; i < 4; i++)); do
    IFS='|' read -r partition tau <<< "${JOBS[$i]}"
    run_job "${GPUS[$i]}" "$partition" "$tau" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then
    echo "One or more quick-screen jobs failed." >&2
    exit 1
fi
echo "CIFAR-10 quick soft-tau screen complete."
