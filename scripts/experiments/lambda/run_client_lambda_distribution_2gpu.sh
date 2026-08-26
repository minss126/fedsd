#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    cat <<'EOF'
Usage:
  GPUS_OVERRIDE="0 1" bash scripts/experiments/lambda/run_client_lambda_distribution_2gpu.sh

Runs the final CIFAR-100 soft-b adaptive method on IID and beta=0.1 in
parallel, saves selected-client lambda/r/b values for every round, and then
creates the last-30 client-wise lambda box/strip plot and CSV.

Expected wall time: approximately 2 h 20 min to 2 h 40 min on two GPUs.
Optional overrides: SEED, USE_WANDB, LOG_ROOT, SKIP_EXISTING, DRY_RUN.
EOF
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1})
if [ "${#GPUS[@]}" -ne 2 ]; then
    echo "This launcher requires exactly two GPU ids; received: ${GPUS[*]}" >&2
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
KD_TEMPERATURE="${KD_TEMPERATURE:-1.00}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.00}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LAMBDA_MAX="${LAMBDA_MAX:-1.00}"
LAMBDA_WARMUP="${LAMBDA_WARMUP:-250}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_client_lambda_distribution}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/adaptive_lambda/client_distribution}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_NAME="soft_b_clientstats_tkd1p00_lmax1p00_warm250_tau0p85"

WANDB_ARGS=()
if [ "${USE_WANDB:-0}" = "1" ]; then
    WANDB_ARGS+=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
fi

partition_args() {
    case "$1" in
        iid) printf '%s\n' "--partition" "iid" ;;
        beta_0.1) printf '%s\n' "--partition" "noniid" "--beta" "0.1" ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1
    local env_name=$2
    local log_dir="${LOG_ROOT}/${env_name}/fedavg"
    local log_file="${log_dir}/${RUN_NAME}.log"
    local terminal_file="${log_dir}/${RUN_NAME}_terminal.log"
    local -a env_args cmd

    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | completed client-wise log already exists"
        return
    fi

    mapfile -t env_args < <(partition_args "$env_name")
    cmd=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data
        --n_clients 100 --sample_fraction 0.1
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED"
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT"
        --log_file_name "${env_name}/fedavg/${RUN_NAME}"
        --model resnet18_byot --alg fedbyot
        --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
        --byot_alpha "$LAMBDA_MAX"
        --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00
        --byot_round_lambda_warmup "$LAMBDA_WARMUP"
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
        --alpha_min_scale 0.0
        --byot_client_proxy teacher_label_prob
        --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00
        --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
        --byot_client_skew_proxy prediction_entropy
        --byot_log_prediction_entropy_components
        --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.00
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau "$SOFT_TAU"
        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
        "${env_args[@]}" "${WANDB_ARGS[@]}"
    )

    echo "[GPU ${gpu_id}] start: CIFAR-100 | ${env_name} | ${RUN_NAME}"
    if [ "$DRY_RUN" = "1" ]; then
        printf '  %q' "${cmd[@]}"
        printf '\n'
        return
    fi
    "${cmd[@]}" > "$terminal_file" 2>&1
    echo "[GPU ${gpu_id}] complete: CIFAR-100 | ${env_name} | ${RUN_NAME}"
}

echo "========== Client-wise Effective Lambda Distribution =========="
echo "GPUs=${GPUS[*]} | partitions=IID,beta_0.1 | seed=${SEED}"
echo "rounds=${ROUNDS} | local_epochs=${LOCAL_EPOCHS} | warmup=${LAMBDA_WARMUP}"
echo "lambda_max=${LAMBDA_MAX} | KD_T=${KD_TEMPERATURE} | proxy_T=${PROXY_TEMPERATURE}"
echo "tau=${SOFT_TAU} | soft_T=${SOFT_TEMPERATURE} | p=${SKEW_POWER}"
echo "logs=${LOG_ROOT}"
echo "estimated wall time: 2 h 20 min to 2 h 40 min"

run_job "${GPUS[0]}" iid &
pid_iid=$!
run_job "${GPUS[1]}" beta_0.1 &
pid_beta=$!

wait "$pid_iid"
wait "$pid_beta"

if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run complete; plot generation skipped."
    exit 0
fi

echo "Both training runs completed; generating client-wise distribution plot."
"$PYTHON_BIN" scripts/experiments/analysis/plot_client_effective_lambda_distribution.py \
    --log-root "$LOG_ROOT" \
    --run-name "$RUN_NAME" \
    --window 30 \
    --output-dir "$OUTPUT_DIR"

echo "Client-wise effective-lambda experiment and plot complete."
