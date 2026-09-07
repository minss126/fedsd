#!/usr/bin/env bash

# CIFAR-10/CIFAR-100 pilot for changing only the BYOT teacher source:
#   local_teacher  : concurrently trained local final classifier -> local branches
#   global_teacher : frozen round-start global final classifier -> local branches
#
# The local final classifier remains CE-only in both methods. Feature imitation
# also remains local-final-feature -> local-branch-feature, so the comparison
# isolates the source of the logit KD target and its adaptive proxies.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

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
read -r -a PARTITIONS <<< "${PARTITIONS_OVERRIDE:-iid beta_0.1}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-0}"

ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"
KEEP_LAST_BATCH="${KEEP_LAST_BATCH:-0}"
[[ "$KEEP_LAST_BATCH" == 0 || "$KEEP_LAST_BATCH" == 1 ]] || {
    echo "KEEP_LAST_BATCH must be 0 or 1." >&2
    exit 1
}

FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.5}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
JS_GAIN="${JS_GAIN_OVERRIDE:-1.0}"
WARMUP_ROUNDS="$(awk -v rounds="$ROUNDS" -v ratio="$WARMUP_RATIO" \
    'BEGIN { printf "%d", int(rounds * ratio + 0.5) }')"

LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_global_teacher_js_branch_pilot}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REUSE_EXISTING_LOCAL="${REUSE_EXISTING_LOCAL:-1}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$LOG_ROOT"

dataset_args() {
    case "$1" in
        cifar10) printf '%s\n' --dataset cifar10 --num_classes 10 ;;
        cifar100) printf '%s\n' --dataset cifar100 --num_classes 100 ;;
        *) echo "Unknown dataset: $1" >&2; return 1 ;;
    esac
}

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

teacher_source() {
    case "$1" in
        local_teacher) echo local ;;
        global_teacher) echo global ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

has_completed_run() {
    local log_file="$1" pkl_file="$2"
    [[ -f "$log_file" ]] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file" \
        && [[ -f "$pkl_file" ]]
}

# Return a completed, configuration-matched historical local-teacher run.
# The old JSON has no byot_teacher_source field because local was the only
# behavior at that point; it is equivalent to the new explicit `local` value.
existing_local_reference() {
    local dataset="$1" partition="$2" seed="$3" base=""
    [[ "$REUSE_EXISTING_LOCAL" == 1 ]] || return 1
    [[ "$seed" == 0 && "$ROUNDS" == 500 && "$LOCAL_EPOCHS" == 5 ]] || return 1
    [[ "$LR" == 0.1 && "$BATCH_SIZE" == 64 && "$TEST_BATCH_SIZE" == 512 ]] || return 1
    [[ "$NUM_CLIENTS" == 100 && "$SAMPLE_FRACTION" == 0.1 && "$MIN_REQUIRE_SIZE" == 64 ]] || return 1
    [[ "$KEEP_LAST_BATCH" == 0 ]] || return 1
    [[ "$FEATURE_BETA" == 0.01 && "$KD_TEMPERATURE" == 1.0 && "$PROXY_TEMPERATURE" == 1.0 ]] || return 1
    [[ "$LAMBDA_MAX" == 1.0 && "$WARMUP_ROUNDS" == 250 && "$SKEW_POWER" == 2.0 ]] || return 1
    [[ "$SOFT_TAU" == 0.85 && "$SOFT_TEMPERATURE" == 0.05 && "$JS_GAIN" == 1.0 ]] || return 1

    case "${dataset}|${partition}" in
        cifar10\|iid|cifar10\|beta_0.1|cifar100\|iid|cifar100\|beta_0.1)
            base="logs/analysis/logs_kd_need_proxy_seed0/runs/js/${dataset}/${partition}/${dataset}_${partition}_js_seed0_r500"
            ;;
        *) return 1 ;;
    esac
    [[ -f "${base}.json" ]] && has_completed_run "${base}.log" "${base}.pkl" || return 1
    printf '%s\n' "$base"
}

new_run_completed() {
    local dataset="$1" partition="$2" method="$3" seed="$4"
    local name log_name
    name="${dataset}_${partition}_${method}_js_branch_min${MIN_REQUIRE_SIZE}_seed${seed}_r${ROUNDS}"
    log_name="runs/${method}/${dataset}/${partition}/seed${seed}/${name}"
    has_completed_run "${LOG_ROOT}/${log_name}.log" "${LOG_ROOT}/${log_name}.pkl"
}

run_job() {
    local gpu="$1" dataset="$2" partition="$3" method="$4" seed="$5"
    local source name log_name run_dir log_file pkl_file
    local -a dataset_flags part keep_last_flags cmd

    source="$(teacher_source "$method")"
    name="${dataset}_${partition}_${method}_js_branch_min${MIN_REQUIRE_SIZE}_seed${seed}_r${ROUNDS}"
    log_name="runs/${method}/${dataset}/${partition}/seed${seed}/${name}"
    run_dir="${LOG_ROOT}/runs/${method}/${dataset}/${partition}/seed${seed}"
    log_file="${LOG_ROOT}/${log_name}.log"
    pkl_file="${LOG_ROOT}/${log_name}.pkl"
    mkdir -p "$run_dir"

    if [[ "$SKIP_EXISTING" == 1 ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] skip: ${method} | ${dataset} | ${partition} | seed=${seed}"
        return 0
    fi

    mapfile -t dataset_flags < <(dataset_args "$dataset")
    mapfile -t part < <(partition_args "$partition")
    keep_last_flags=()
    if [[ "$KEEP_LAST_BATCH" == 1 ]]; then
        keep_last_flags+=(--client_keep_last_batch)
    fi
    cmd=(
        "$PYTHON_BIN" main.py
        "${dataset_flags[@]}" --datadir ./data --in_channels 3
        "${part[@]}" --min_require_size "$MIN_REQUIRE_SIZE" "${keep_last_flags[@]}"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$seed" --device "cuda:${gpu}"
        --logdir "$LOG_ROOT" --log_file_name "$log_name"
        --sequential_client_execution
        --paired_resnet_init --paired_execution_rng --preserve_byot_proxy_rng
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
        --byot_teacher_source "$source"
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

    echo "[GPU ${gpu}] start: ${method} | ${dataset} | ${partition} | seed=${seed}"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run][GPU %s] ' "$gpu"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    if ! "${cmd[@]}" > "${run_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${method} | ${dataset} | ${partition} | seed=${seed}" >&2
        tail -40 "${run_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] incomplete: ${name}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${method} | ${dataset} | ${partition} | seed=${seed}"
}

JOBS=()
# Add only local-teacher settings that cannot be reused. Put these shorter jobs
# first so a four-GPU run balances the fifth job behind one global-teacher job.
for dataset in "${DATASETS[@]}"; do
    for partition in "${PARTITIONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if reference="$(existing_local_reference "$dataset" "$partition" "$seed")"; then
                echo "[reuse] local_teacher | ${dataset} | ${partition} | seed=${seed}"
                echo "        ${reference}"
            elif ! new_run_completed "$dataset" "$partition" local_teacher "$seed"; then
                JOBS+=("${dataset}|${partition}|local_teacher|${seed}")
            fi
        done
    done
done
# Global-teacher is the new experiment and is required for every setting.
for dataset in "${DATASETS[@]}"; do
    for partition in "${PARTITIONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if ! new_run_completed "$dataset" "$partition" global_teacher "$seed"; then
                JOBS+=("${dataset}|${partition}|global_teacher|${seed}")
            fi
        done
    done
done

echo "========== Global-teacher JS-branch pilot =========="
echo "GPUs=${GPUS[*]} | pending jobs=${#JOBS[@]}"
echo "datasets=${DATASETS[*]} | methods=local_teacher(missing only) global_teacher"
echo "partitions=${PARTITIONS[*]} | seeds=${SEEDS[*]} | R=${ROUNDS} | E=${LOCAL_EPOCHS}"
echo "warmup=${WARMUP_ROUNDS}/${ROUNDS} | lambda_max=${LAMBDA_MAX} | min=${MIN_REQUIRE_SIZE} | keep_last=${KEEP_LAST_BATCH}"

if (( ${#JOBS[@]} == 0 )); then
    echo "All requested runs are already complete."
    exit 0
fi

run_queue() {
    local queue_index="$1" gpu="${GPUS[$1]}"
    local job_index dataset partition method seed failed=0
    for ((job_index=queue_index; job_index<${#JOBS[@]}; job_index+=${#GPUS[@]})); do
        IFS='|' read -r dataset partition method seed <<< "${JOBS[$job_index]}"
        run_job "$gpu" "$dataset" "$partition" "$method" "$seed" || failed=1
    done
    return "$failed"
}

pids=()
for ((index=0; index<${#GPUS[@]} && index<${#JOBS[@]}; index++)); do
    run_queue "$index" &
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
    echo "Global-teacher JS-branch pilot complete."
fi
