#!/usr/bin/env bash

# Cross-check lambda_max={1,2} for the final JS-client, feature-free protocol.
#
# server4: CIFAR-10, IID/beta=.1, lambda_max={1,2} (four parallel jobs)
# server2: CIFAR-100, beta=.3, lambda_max={1,2} (two parallel jobs)
#
# No paired/preserved RNG-control flags are passed.  All other settings match
# the canonical JS-client retune exactly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

RUN_SET="${RUN_SET:?Set RUN_SET to server4 or server2}"
read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3}"
(( ${#GPUS[@]} > 0 )) || { echo "GPUS_OVERRIDE is empty." >&2; exit 1; }

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
WARMUP_ROUNDS="${WARMUP_ROUNDS:-$((ROUNDS / 2))}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"

KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
JS_GAIN="${JS_GAIN:-1.0}"

LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_js_client_lmax_crosscheck_canonical_no_feature}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

[[ "$MIN_REQUIRE_SIZE" == 64 ]] || {
    echo "Final protocol requires MIN_REQUIRE_SIZE=64; got ${MIN_REQUIRE_SIZE}." >&2
    exit 1
}
[[ "$KD_TEMPERATURE" == 1 || "$KD_TEMPERATURE" == 1.0 ]] || {
    echo "This cross-check fixes KD_TEMPERATURE=1; got ${KD_TEMPERATURE}." >&2
    exit 1
}
[[ "$PROXY_TEMPERATURE" == 1 || "$PROXY_TEMPERATURE" == 1.0 ]] || {
    echo "This cross-check fixes PROXY_TEMPERATURE=1; got ${PROXY_TEMPERATURE}." >&2
    exit 1
}
[[ "$JS_GAIN" == 1 || "$JS_GAIN" == 1.0 ]] || {
    echo "This cross-check fixes JS_GAIN=1; got ${JS_GAIN}." >&2
    exit 1
}

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

dataset_args() {
    case "$1" in
        cifar10) printf '%s\n' --dataset cifar10 --in_channels 3 --num_classes 10 ;;
        cifar100) printf '%s\n' --dataset cifar100 --in_channels 3 --num_classes 100 ;;
        *) echo "Unknown dataset: $1" >&2; return 1 ;;
    esac
}

has_completed_run() {
    local log_file="$1" pkl_file="$2"
    [[ -f "$log_file" ]] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file" \
        && [[ -s "$pkl_file" ]]
}

# job format: dataset|partition|lambda_max
JOBS=()
case "$RUN_SET" in
    server4)
        JOBS=(
            'cifar10|iid|1.0'
            'cifar10|iid|2.0'
            'cifar10|beta_0.1|1.0'
            'cifar10|beta_0.1|2.0'
        )
        ;;
    server2)
        JOBS=(
            'cifar100|beta_0.3|1.0'
            'cifar100|beta_0.3|2.0'
        )
        ;;
    *)
        echo "Unknown RUN_SET=${RUN_SET}; expected server4 or server2." >&2
        exit 1
        ;;
esac

run_job() {
    local gpu="$1" job="$2"
    local dataset partition lambda_max name rel_dir log_file pkl_file
    local -a data_flags part_flags cmd
    IFS='|' read -r dataset partition lambda_max <<< "$job"
    mapfile -t data_flags < <(dataset_args "$dataset")
    mapfile -t part_flags < <(partition_args "$partition")

    name="${dataset}_${partition}_js_client_lmax$(value_tag "$lambda_max")_tau0p85_tkd1p00_canonical_nofeat_seed${SEED}_r${ROUNDS}"
    rel_dir="${dataset}/${partition}/seed${SEED}/lmax$(value_tag "$lambda_max")"
    log_file="${LOG_ROOT}/${rel_dir}/${name}.log"
    pkl_file="${LOG_ROOT}/${rel_dir}/${name}.pkl"
    mkdir -p "${LOG_ROOT}/${rel_dir}"

    if [[ "$SKIP_EXISTING" == 1 ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] skip: ${dataset} | ${partition} | lambda_max=${lambda_max}"
        return 0
    fi

    cmd=(
        "$PYTHON_BIN" main.py
        "${data_flags[@]}" --datadir ./data
        "${part_flags[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS"
        --optimizer sgd --lr "$LR" --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --seed "$SEED" --device "cuda:${gpu}"
        --sequential_client_execution
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta 0.0
        --byot_teacher_source local --byot_alpha "$lambda_max" --alpha_min_scale 0.0
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
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
        --byot_branch_need_proxy js_client
        --byot_branch_need_gain "$JS_GAIN" --byot_branch_need_min_gate 0.0
        --byot_branch_need_temperature "$PROXY_TEMPERATURE"
        --logdir "$LOG_ROOT" --log_file_name "${rel_dir}/${name}"
    )

    echo "[GPU ${gpu}] start: ${dataset} | ${partition} | lambda_max=${lambda_max}"
    if [[ "$DRY_RUN" == 1 ]]; then
        local quoted
        printf -v quoted '%q ' "${cmd[@]}"
        printf '[dry-run][GPU %s] %s\n' "$gpu" "$quoted"
        return 0
    fi

    if ! "${cmd[@]}" > "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${dataset} | ${partition} | lambda_max=${lambda_max}" >&2
        tail -40 "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] incomplete: ${dataset} | ${partition} | lambda_max=${lambda_max}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${dataset} | ${partition} | lambda_max=${lambda_max}"
}

echo "========== JS-client lambda_max cross-check =========="
echo "run_set=${RUN_SET} | GPUs=${GPUS[*]} | jobs=${#JOBS[@]} | seed=${SEED}"
echo "ResNet18-BYOT / FedAvg / R=${ROUNDS} / E=${LOCAL_EPOCHS} / participation=${SAMPLE_FRACTION}"
echo "KD-only | T_KD=1 | T_proxy=1 | feature_beta=0 | min_require_size=64"
echo "warm-up=${WARMUP_ROUNDS} | tau=${SOFT_TAU} | JS-client gain=1"
echo "canonical execution: paired/preserved RNG controls are absent"
echo "log_root=${LOG_ROOT}"

declare -a QUEUES
for ((i=0; i<${#GPUS[@]}; i++)); do QUEUES[$i]=''; done
for ((i=0; i<${#JOBS[@]}; i++)); do
    target=$((i % ${#GPUS[@]}))
    QUEUES[$target]+="${JOBS[$i]}"$'\n'
done

run_queue() {
    local gpu_index="$1"
    local gpu="${GPUS[$gpu_index]}"
    local failed=0
    local job
    while IFS= read -r job; do
        [[ -n "$job" ]] || continue
        run_job "$gpu" "$job" || failed=1
    done <<< "${QUEUES[$gpu_index]}"
    return "$failed"
}

if [[ "$DRY_RUN" == 1 ]]; then
    status=0
    for ((i=0; i<${#GPUS[@]} && i<${#JOBS[@]}; i++)); do
        run_queue "$i" || status=1
    done
    (( status == 0 )) || exit "$status"
    echo "Dry run complete (${#JOBS[@]} jobs through the real queue path)."
    exit 0
fi

pids=()
for ((i=0; i<${#GPUS[@]} && i<${#JOBS[@]}; i++)); do
    run_queue "$i" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then
    echo "At least one run failed; inspect *_terminal.log." >&2
    exit "$status"
fi
echo "JS-client lambda_max cross-check complete (${#JOBS[@]} jobs)."

