#!/usr/bin/env bash

# Canonical-initialization, feature-free pilot for selecting both the
# JS granularity and the branch-KD temperature.
#
# Compared methods:
#   adaptive_no_js : reliability x soft prediction-entropy correction
#   js_client      : adaptive_no_js x one client-level JS need gate
#   js_branch      : adaptive_no_js x a separate JS need gate per branch
#
# This runner deliberately DOES NOT pass --paired_resnet_init.  All methods
# use the same ResNet18-BYOT architecture and seed, so the ordinary model
# construction already gives them the same complete canonical initial state.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3}"
(( ${#GPUS[@]} > 0 )) || { echo "GPUS_OVERRIDE is empty." >&2; exit 1; }

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.1})
METHODS=(${METHODS_OVERRIDE:-adaptive_no_js js_client js_branch})
KD_TEMPERATURES=(${KD_TEMPERATURES_OVERRIDE:-0.5 1.0})

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-$((ROUNDS / 2))}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"

LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
JS_GAIN="${JS_GAIN:-1.0}"

LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_js_granularity_temperature_canonical_no_feature}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$MIN_REQUIRE_SIZE" != 64 ]]; then
    echo "This final-protocol pilot requires MIN_REQUIRE_SIZE=64; got ${MIN_REQUIRE_SIZE}." >&2
    exit 1
fi

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

need_proxy_for_method() {
    case "$1" in
        adaptive_no_js) printf '%s' none ;;
        js_client) printf '%s' js_client ;;
        js_branch) printf '%s' js ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

has_completed_run() {
    local log_file="$1" pkl_file="$2"
    [[ -f "$log_file" ]] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file" \
        && [[ -s "$pkl_file" ]]
}

# job: partition|method|KD temperature
JOBS=()
for partition in "${PARTITIONS[@]}"; do
    for method in "${METHODS[@]}"; do
        for temperature in "${KD_TEMPERATURES[@]}"; do
            JOBS+=("${partition}|${method}|${temperature}")
        done
    done
done

run_job() {
    local gpu="$1" job="$2"
    local partition method temperature need_proxy tag name rel_dir log_file pkl_file
    local -a part_flags cmd
    IFS='|' read -r partition method temperature <<< "$job"

    need_proxy="$(need_proxy_for_method "$method")"
    tag="tkd$(value_tag "$temperature")"
    name="cifar100_${partition}_${method}_${tag}_canonical_nofeat_seed${SEED}_r${ROUNDS}"
    rel_dir="${partition}/seed${SEED}/${method}"
    log_file="${LOG_ROOT}/${rel_dir}/${name}.log"
    pkl_file="${LOG_ROOT}/${rel_dir}/${name}.pkl"
    mkdir -p "${LOG_ROOT}/${rel_dir}"

    if [[ "$SKIP_EXISTING" == 1 ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] skip: ${partition} | ${method} | T_KD=${temperature}"
        return 0
    fi

    mapfile -t part_flags < <(partition_args "$partition")
    cmd=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data --in_channels 3 --num_classes 100
        "${part_flags[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS"
        --optimizer sgd --lr "$LR" --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --seed "$SEED" --device "cuda:${gpu}"
        --sequential_client_execution --paired_execution_rng --preserve_byot_proxy_rng
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta 0.0
        --byot_teacher_source local --byot_alpha "$LAMBDA_MAX" --alpha_min_scale 0.0
        --temperature "$temperature"
        --byot_branch_kd_teacher_temperature "$temperature"
        --byot_branch_kd_student_temperature "$temperature"
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
        --byot_round_lambda_schedule linear
        --byot_round_lambda_min 0.0 --byot_round_lambda_warmup "$WARMUP_ROUNDS"
        --byot_client_proxy teacher_label_prob
        --byot_client_alpha_min 0.0 --byot_client_alpha_max 1.0
        --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
        --byot_client_skew_proxy prediction_entropy
        --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.0
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau "$SOFT_TAU"
        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
        --byot_branch_need_proxy "$need_proxy"
        --byot_branch_need_gain "$JS_GAIN" --byot_branch_need_min_gate 0.0
        --byot_branch_need_temperature "$PROXY_TEMPERATURE"
        --logdir "$LOG_ROOT" --log_file_name "${rel_dir}/${name}"
    )

    echo "[GPU ${gpu}] start: ${partition} | ${method} | T_KD=${temperature} | canonical init"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run][GPU %s] ' "$gpu"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    if ! "${cmd[@]}" > "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${partition} | ${method} | T_KD=${temperature}" >&2
        tail -40 "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] incomplete: ${partition} | ${method} | T_KD=${temperature}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${partition} | ${method} | T_KD=${temperature}"
}

echo "========== JS granularity x KD-temperature pilot =========="
echo "GPUs=${GPUS[*]} | jobs=${#JOBS[@]} | seed=${SEED}"
echo "CIFAR-100 / ResNet18-BYOT / FedAvg / R=${ROUNDS} / E=${LOCAL_EPOCHS}"
echo "partitions=${PARTITIONS[*]} | methods=${METHODS[*]} | T_KD=${KD_TEMPERATURES[*]}"
echo "feature_beta=0 | min_require_size=${MIN_REQUIRE_SIZE} | warm-up=${WARMUP_ROUNDS}"
echo "lambda_max=${LAMBDA_MAX} | tau=${SOFT_TAU} | JS_gain=${JS_GAIN}"
echo "initialization=canonical BYOT (--paired_resnet_init is intentionally absent)"
echo "log_root=${LOG_ROOT}"

if [[ "$DRY_RUN" == 1 ]]; then
    for job in "${JOBS[@]}"; do
        run_job "${GPUS[0]}" "$job"
    done
    echo "Dry run complete."
    exit 0
fi

run_queue() {
    local gpu_index="$1" job_index failed=0
    local gpu="${GPUS[$gpu_index]}"
    for ((job_index=gpu_index; job_index<${#JOBS[@]}; job_index+=${#GPUS[@]})); do
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

if (( status != 0 )); then
    echo "At least one run failed; inspect *_terminal.log." >&2
    exit "$status"
fi

echo "JS granularity x KD-temperature pilot complete (${#JOBS[@]} jobs)."
