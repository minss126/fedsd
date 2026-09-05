#!/usr/bin/env bash

# Seed-0 comparison of KD-necessity proxies and their controls.  The existing
# soft-b trajectories from stage 1 are reused as the current-rule reference.
# JS/advantage/combined gains are calibrated on the pre-local stage-1 rows so
# all three adaptive gates have the same pooled mean strength.  A constant
# gate at that mean is included to separate adaptive routing from merely
# reducing the overall KD strength. ``js_client`` uses the same calibrated JS
# signal but shares one client-wise gate across all three branches. ``plain``
# is a name-paired FedAvg baseline without the BYOT branches.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

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

SEED=0
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
DATASETS=(${DATASETS_OVERRIDE:-cifar10 cifar100})
PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.1})
METHODS=(${METHODS_OVERRIDE:-constant js advantage combined})

FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.5}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
NEED_MIN_GATE="${NEED_MIN_GATE:-0.0}"
WARMUP_ROUNDS="$(awk -v rounds="$ROUNDS" -v ratio="$WARMUP_RATIO" \
    'BEGIN { printf "%d", int(rounds * ratio + 0.5) }')"

STAGE1_ROOT="${STAGE1_ROOT:-logs/analysis/logs_kd_necessity_stage1}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_kd_need_proxy_seed0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${LOG_ROOT}/.matplotlib}"
export MPLCONFIGDIR
mkdir -p "$LOG_ROOT" "$MPLCONFIGDIR"

CALIBRATION_JSON="${LOG_ROOT}/need_proxy_calibration.json"
if [[ ! -d "${STAGE1_ROOT}/diagnostics" ]]; then
    echo "Missing stage-1 diagnostics: ${STAGE1_ROOT}/diagnostics" >&2
    exit 1
fi
"$PYTHON_BIN" scripts/experiments/analysis/calibrate_kd_need_proxy_gains.py \
    --input-root "${STAGE1_ROOT}/diagnostics" \
    --output "$CALIBRATION_JSON" \
    > "${LOG_ROOT}/need_proxy_calibration_terminal.txt"

read_gain() {
    "$PYTHON_BIN" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["proxies"][sys.argv[2]]["gain"])' \
        "$CALIBRATION_JSON" "$1"
}
TARGET_GATE_MEAN="$(
    "$PYTHON_BIN" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["target_gate_mean"])' \
        "$CALIBRATION_JSON"
)"
CONSTANT_GAIN="${CONSTANT_GAIN_OVERRIDE:-$TARGET_GATE_MEAN}"
JS_GAIN="${JS_GAIN_OVERRIDE:-$(read_gain js)}"
ADVANTAGE_GAIN="${ADVANTAGE_GAIN_OVERRIDE:-$(read_gain advantage)}"
COMBINED_GAIN="${COMBINED_GAIN_OVERRIDE:-$(read_gain combined)}"

method_gain() {
    case "$1" in
        plain) echo "0.0" ;;
        constant) echo "$CONSTANT_GAIN" ;;
        js) echo "$JS_GAIN" ;;
        js_client) echo "$JS_GAIN" ;;
        advantage) echo "$ADVANTAGE_GAIN" ;;
        combined) echo "$COMBINED_GAIN" ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

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
    local gpu_id="$1" method="$2" dataset="$3" partition="$4"
    local gain run_name log_name log_dir log_file pkl_file
    local -a CMD DATASET_FLAGS PARTITION_FLAGS
    gain="$(method_gain "$method")"
    run_name="${dataset}_${partition}_${method}_seed0_r${ROUNDS}"
    log_name="runs/${method}/${dataset}/${partition}/${run_name}"
    log_dir="${LOG_ROOT}/runs/${method}/${dataset}/${partition}"
    log_file="${LOG_ROOT}/${log_name}.log"
    pkl_file="${LOG_ROOT}/${log_name}.pkl"
    mkdir -p "$log_dir"

    if [[ "$SKIP_EXISTING" == "1" ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu_id}] skip: ${method} | ${dataset} | ${partition}"
        return 0
    fi

    mapfile -t DATASET_FLAGS < <(dataset_args "$dataset")
    mapfile -t PARTITION_FLAGS < <(partition_args "$partition")
    CMD=(
        "$PYTHON_BIN" main.py
        --datadir ./data --in_channels 3
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$SEED" --device "cuda:${gpu_id}"
        --logdir "$LOG_ROOT" --log_file_name "$log_name"
        --sequential_client_execution
        --paired_resnet_init --paired_execution_rng --preserve_byot_proxy_rng
    )
    if [[ "$method" == "plain" ]]; then
        CMD+=(--model resnet18 --alg fedavg)
    else
        CMD+=(
            --model resnet18_byot --alg fedbyot
            --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
            --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
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
            --byot_branch_need_proxy "$method"
            --byot_branch_need_gain "$gain"
            --byot_branch_need_min_gate "$NEED_MIN_GATE"
            --byot_branch_need_temperature "$PROXY_TEMPERATURE"
        )
    fi
    CMD+=("${DATASET_FLAGS[@]}" "${PARTITION_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: ${method} | ${dataset} | ${partition} | gain=${gain}"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run][GPU %s] ' "$gpu_id"
        printf '%q ' "${CMD[@]}"
        printf '\n'
        return 0
    fi
    if ! "${CMD[@]}" > "${log_dir}/${run_name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu_id}] failed: ${method} | ${dataset} | ${partition}" >&2
        tail -40 "${log_dir}/${run_name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu_id}] incomplete: ${run_name}" >&2
        return 1
    fi
    echo "[GPU ${gpu_id}] complete: ${method} | ${dataset} | ${partition}"
}

JOBS=()
for method in "${METHODS[@]}"; do
    for partition in "${PARTITIONS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            JOBS+=("${method}|${dataset}|${partition}")
        done
    done
done

declare -a QUEUES LOADS
for ((gpu_index = 0; gpu_index < 4; gpu_index++)); do
    QUEUES[$gpu_index]=""
    LOADS[$gpu_index]=0
done
for job in "${JOBS[@]}"; do
    target=0
    for ((gpu_index = 1; gpu_index < 4; gpu_index++)); do
        (( LOADS[gpu_index] < LOADS[target] )) && target=$gpu_index
    done
    QUEUES[$target]+="${job}"$'\n'
    LOADS[$target]=$((LOADS[$target] + 1))
done

echo "========== Seed-0 KD-need proxy/control comparison =========="
echo "gpus=${GPUS[*]}, jobs=${#JOBS[@]}, methods=${METHODS[*]}"
echo "datasets=${DATASETS[*]}, partitions=${PARTITIONS[*]}, R=${ROUNDS}, E=${LOCAL_EPOCHS}"
echo "mean-matched gates: constant=${CONSTANT_GAIN}, JS gain=${JS_GAIN}, advantage gain=${ADVANTAGE_GAIN}, combined gain=${COMBINED_GAIN}"
echo "current-rule reference=${STAGE1_ROOT}/runs"
echo "estimated 4-GPU wall time: ${TIME_ESTIMATE:-about 15-19 hours}"
for ((gpu_index = 0; gpu_index < 4; gpu_index++)); do
    echo "GPU ${GPUS[$gpu_index]}: ${LOADS[$gpu_index]} job(s)"
done

run_queue() {
    local gpu_id="$1" queue="$2" failed=0
    while IFS='|' read -r method dataset partition; do
        [[ -z "${method:-}" ]] && continue
        if ! run_job "$gpu_id" "$method" "$dataset" "$partition"; then
            failed=1
        fi
    done <<< "$queue"
    return "$failed"
}

pids=()
for ((gpu_index = 0; gpu_index < 4; gpu_index++)); do
    if [[ -n "${QUEUES[$gpu_index]}" ]]; then
        run_queue "${GPUS[$gpu_index]}" "${QUEUES[$gpu_index]}" &
        pids+=("$!")
    fi
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if (( status != 0 )); then
    echo "At least one KD-need proxy job failed." >&2
    exit "$status"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete; analysis skipped."
    exit 0
fi

"$PYTHON_BIN" scripts/experiments/analysis/analyze_kd_need_proxy_seed0.py \
    --experiment-root "$LOG_ROOT" \
    --current-root "$STAGE1_ROOT" \
    --output-dir "${LOG_ROOT}/summary"

echo "Comparison complete: ${LOG_ROOT}/summary/report.md"
