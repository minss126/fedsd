#!/usr/bin/env bash

# Fixed-lambda teacher-source control:
#   local_kd  : concurrently trained local final classifier -> local branches
#   global_kd : frozen round-start global final classifier -> local branches
#
# No warm-up, reliability, skew correction, sample weighting, or JS gate is
# used. The branch objective is KD-only with lambda=1 from round 0. Feature
# imitation is retained equally in both methods to match the fixed-lambda BYOT
# baseline; set FEATURE_BETA=0 for a pure logit-KD-only auxiliary objective.

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
read -r -a METHODS <<< "${METHODS_OVERRIDE:-local_kd global_kd}"
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
LAMBDA="${LAMBDA:-1.0}"

LOG_ROOT="${LOG_ROOT:-logs/lambda/analysis/logs_global_teacher_fixed_lambda1_pilot}"
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

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

teacher_source() {
    case "$1" in
        local_kd) echo local ;;
        global_kd) echo global ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

has_completed_run() {
    local log_file="$1" pkl_file="$2"
    [[ -f "$log_file" ]] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file" \
        && [[ -f "$pkl_file" ]]
}

run_job() {
    local gpu="$1" dataset="$2" partition="$3" method="$4" seed="$5"
    local source name log_name run_dir log_file pkl_file
    local -a dataset_flags part keep_last_flags cmd

    source="$(teacher_source "$method")"
    name="${dataset}_${partition}_${method}_lambda1_min${MIN_REQUIRE_SIZE}_seed${seed}_r${ROUNDS}"
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
        --sequential_client_execution --paired_resnet_init --paired_execution_rng
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
        --byot_teacher_source "$source"
        --byot_alpha "$LAMBDA"
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_sample_proxy none --byot_client_proxy none
        --byot_client_skew_proxy none --byot_branch_need_proxy none
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

# Four local jobs followed by four global jobs. With four GPUs each GPU gets
# one job of each type, avoiding a slow global/global queue on one device.
JOBS=()
for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for partition in "${PARTITIONS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                JOBS+=("${dataset}|${partition}|${method}|${seed}")
            done
        done
    done
done

echo "========== Fixed lambda=1 teacher-source pilot =========="
echo "GPUs=${GPUS[*]} | jobs=${#JOBS[@]} | methods=${METHODS[*]}"
echo "datasets=${DATASETS[*]} | partitions=${PARTITIONS[*]} | seeds=${SEEDS[*]}"
echo "R=${ROUNDS} | E=${LOCAL_EPOCHS} | lambda=${LAMBDA} | warmup=none"
echo "min=${MIN_REQUIRE_SIZE} | keep_last=${KEEP_LAST_BATCH} | feature_beta=${FEATURE_BETA}"

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
    echo "Fixed lambda=1 teacher-source pilot complete."
fi
