#!/usr/bin/env bash

# Step 1 of the final JS-branch validation plan.
#
# Fill only the missing beta=0.3 adaptive rows for the basic matrix:
#   datasets : CIFAR-10, CIFAR-100
#   model    : ResNet18-BYOT
#   FL       : FedAvg
#   method   : final local-teacher JS-branch adaptive lambda
#   seed     : 0
#
# Existing Plain and fixed-lambda=0.3 results are intentionally not rerun.
# The final protocol uses the original Dirichlet partition rule
# (min_require_size=64 and client_keep_last_batch disabled).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  GPUS_OVERRIDE="0 1 2 3" bash scripts/experiments/lambda/run_js_branch_base_beta03.sh
  GPUS_OVERRIDE="0 1"     bash scripts/experiments/lambda/run_js_branch_base_beta03.sh

Optional overrides:
  DATASETS_OVERRIDE="cifar10 cifar100"
  SEED=0 ROUNDS=500 LOCAL_EPOCHS=5
  SKIP_EXISTING=1 DRY_RUN=1
  LOG_ROOT=logs/lambda/adaptive/logs_js_branch_base_completion
EOF
    exit 0
fi

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3}"
(( ${#GPUS[@]} > 0 )) || {
    echo "Set GPUS_OVERRIDE to at least one GPU id." >&2
    exit 1
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

read -r -a DATASETS <<< "${DATASETS_OVERRIDE:-cifar10 cifar100}"

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"

FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.5}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
JS_GAIN="${JS_GAIN:-1.0}"
WARMUP_ROUNDS="$(awk -v rounds="$ROUNDS" -v ratio="$WARMUP_RATIO" \
    'BEGIN { printf "%d", int(rounds * ratio + 0.5) }')"

LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_js_branch_base_completion}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$LOG_ROOT"

dataset_args() {
    case "$1" in
        cifar10) printf '%s\n' --dataset cifar10 --num_classes 10 ;;
        cifar100) printf '%s\n' --dataset cifar100 --num_classes 100 ;;
        *) echo "Unknown dataset: $1" >&2; return 1 ;;
    esac
}

has_completed_run() {
    local log_file="$1" pkl_file="$2"
    [[ -f "$log_file" ]] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file" \
        && [[ -f "$pkl_file" ]]
}

run_job() {
    local gpu="$1" dataset="$2"
    local run_name log_name run_dir log_file pkl_file
    local -a dataset_flags cmd

    run_name="${dataset}_beta_0.3_js_branch_seed${SEED}_r${ROUNDS}"
    log_name="runs/js_branch/${dataset}/beta_0.3/seed${SEED}/${run_name}"
    run_dir="${LOG_ROOT}/runs/js_branch/${dataset}/beta_0.3/seed${SEED}"
    log_file="${LOG_ROOT}/${log_name}.log"
    pkl_file="${LOG_ROOT}/${log_name}.pkl"
    mkdir -p "$run_dir"

    if [[ "$SKIP_EXISTING" == 1 ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] skip: ${dataset} | beta_0.3 | JS-branch adaptive"
        return 0
    fi

    mapfile -t dataset_flags < <(dataset_args "$dataset")
    cmd=(
        "$PYTHON_BIN" main.py
        "${dataset_flags[@]}" --datadir ./data --in_channels 3
        --partition noniid --beta 0.3 --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$SEED" --device "cuda:${gpu}"
        --logdir "$LOG_ROOT" --log_file_name "$log_name"
        --sequential_client_execution
        --paired_resnet_init --paired_execution_rng --preserve_byot_proxy_rng
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
        --byot_teacher_source local
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
        --byot_alpha "$LAMBDA_MAX"
        --byot_round_lambda_schedule linear --byot_round_lambda_min 0.0
        --byot_round_lambda_warmup "$WARMUP_ROUNDS"
        --alpha_min_scale 0.0
        --byot_client_proxy teacher_label_prob
        --byot_client_alpha_min 0.0 --byot_client_alpha_max 1.0
        --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
        --byot_client_skew_proxy prediction_entropy
        --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.0
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau "$SOFT_TAU"
        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
        --byot_branch_need_proxy js --byot_branch_need_gain "$JS_GAIN"
        --byot_branch_need_min_gate 0.0
        --byot_branch_need_temperature "$PROXY_TEMPERATURE"
    )

    echo "[GPU ${gpu}] start: ${dataset} | beta_0.3 | JS-branch adaptive"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run][GPU %s] ' "$gpu"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    if ! "${cmd[@]}" > "${run_dir}/${run_name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${dataset} | beta_0.3" >&2
        tail -40 "${run_dir}/${run_name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] incomplete: ${run_name}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${dataset} | beta_0.3 | JS-branch adaptive"
}

JOBS=()
for dataset in "${DATASETS[@]}"; do
    JOBS+=("$dataset")
done

echo "========== Step 1: JS-branch base beta=0.3 completion =========="
echo "GPUs=${GPUS[*]} | jobs=${#JOBS[@]} | datasets=${DATASETS[*]}"
echo "CIFAR protocol: R=${ROUNDS}, E=${LOCAL_EPOCHS}, seed=${SEED}, min=${MIN_REQUIRE_SIZE}, keep_last=0"
echo "adaptive: lambda_max=${LAMBDA_MAX}, warmup=${WARMUP_ROUNDS}, tau=${SOFT_TAU}, JS gain=${JS_GAIN}"
echo "Plain/fixed-lambda baselines are not rerun."

run_queue() {
    local queue_index="$1" gpu="${GPUS[$1]}" job_index failed=0
    for ((job_index=queue_index; job_index<${#JOBS[@]}; job_index+=${#GPUS[@]})); do
        run_job "$gpu" "${JOBS[$job_index]}" || failed=1
    done
    return "$failed"
}

pids=()
for ((i=0; i<${#GPUS[@]} && i<${#JOBS[@]}; i++)); do
    run_queue "$i" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
(( status == 0 )) || exit "$status"

if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry run complete."
else
    echo "Step 1 complete: ${LOG_ROOT}"
fi
