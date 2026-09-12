#!/usr/bin/env bash

# Controlled follow-up to the TinyImageNet no-feature CE/KD screen.
#
# The existing no-feature results use:
#   dataset=TinyImageNet, model=ResNet18-BYOT, FedAvg, seed=0,
#   partitions={IID,beta=.1}, R=100, E=5.
# This script changes exactly one experimental factor:
#   byot_beta: 0.0 -> 0.01
# and evaluates branch CE-only (alpha=0) and KD-only (alpha=1).
#
# Four jobs are distributed over the requested GPUs. With four GPUs, all
# conditions run concurrently. Completed outputs are skipped on re-launch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3}"
NUM_GPUS=${#GPUS[@]}
(( NUM_GPUS > 0 )) || { echo "GPUS_OVERRIDE is empty." >&2; exit 1; }

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

DATA_DIR="${TINYIMAGENET_DATADIR:-./data/tiny-imagenet-200}"
PAIRED_RESNET_INIT="${PAIRED_RESNET_INIT:-1}"
if [[ -z "${LOG_ROOT:-}" ]]; then
    if [[ "$PAIRED_RESNET_INIT" == 1 ]]; then
        LOG_ROOT="logs/alpha/logs_tinyimagenet_feature_interaction_control"
    else
        LOG_ROOT="logs/alpha/logs_tinyimagenet_legacy_init_alpha_endpoint_control"
    fi
fi

PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.1})
ALPHAS=(${ALPHAS_OVERRIDE:-0.0 1.0})
SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"
CE_TEMPERATURE="${CE_TEMPERATURE:-1.0}"
KD_TEMPERATURE="${KD_TEMPERATURE:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

[[ -d "$DATA_DIR" ]] || {
    echo "TinyImageNet directory is missing: ${DATA_DIR}" >&2
    exit 1
}
[[ "$MIN_REQUIRE_SIZE" == 64 ]] || {
    echo "This control must use MIN_REQUIRE_SIZE=64; got ${MIN_REQUIRE_SIZE}." >&2
    exit 1
}
[[ "$PAIRED_RESNET_INIT" == 0 || "$PAIRED_RESNET_INIT" == 1 ]] || {
    echo "PAIRED_RESNET_INIT must be 0 or 1; got ${PAIRED_RESNET_INIT}." >&2
    exit 1
}

PAIRED_INIT_FLAGS=()
if [[ "$PAIRED_RESNET_INIT" == 1 ]]; then
    PAIRED_INIT_FLAGS+=(--paired_resnet_init)
fi

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

is_zero() {
    awk -v value="$1" 'BEGIN { exit !(value == 0.0) }'
}

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

has_completed_run() {
    local log_file="$1" pkl_file="$2"
    [[ -f "$log_file" ]] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file" \
        && [[ -s "$pkl_file" ]]
}

run_job() {
    local gpu="$1" partition="$2" alpha="$3"
    local alpha_tag feature_tag method_temperature temperature_tag name rel_dir log_file pkl_file
    local -a partition_flags command

    alpha_tag="$(value_tag "$alpha")"
    feature_tag="$(value_tag "$FEATURE_BETA")"
    if is_zero "$alpha"; then
        # Match the existing no-feature CE control exactly. Temperature is
        # mathematically inactive when alpha=0, but keeping the CLI identical
        # makes the feature-beta intervention explicit.
        method_temperature="$CE_TEMPERATURE"
    else
        method_temperature="$KD_TEMPERATURE"
    fi
    temperature_tag="$(value_tag "$method_temperature")"
    name="tinyimagenet_${partition}_alpha${alpha_tag}_tkd${temperature_tag}_feat${feature_tag}_seed${SEED}_r${ROUNDS}"
    rel_dir="${partition}/seed${SEED}"
    log_file="${LOG_ROOT}/${rel_dir}/${name}.log"
    pkl_file="${LOG_ROOT}/${rel_dir}/${name}.pkl"
    mkdir -p "${LOG_ROOT}/${rel_dir}"

    if [[ "$SKIP_EXISTING" == 1 ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] skip: ${partition} | alpha=${alpha} | feature_beta=${FEATURE_BETA}"
        return 0
    fi

    mapfile -t partition_flags < <(partition_args "$partition")
    command=(
        "$PYTHON_BIN" main.py
        --dataset tinyimagenet --datadir "$DATA_DIR" --in_channels 3 --num_classes 200
        "${partition_flags[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients 100 --sample_fraction 0.1
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS"
        --lr "$LR" --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --seed "$SEED" --device "cuda:${gpu}"
        --sequential_client_execution --paired_execution_rng
        "${PAIRED_INIT_FLAGS[@]}"
        --preserve_byot_proxy_rng
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective blend --byot_alpha "$alpha"
        --byot_beta "$FEATURE_BETA" --byot_teacher_source local
        --temperature "$method_temperature"
        --byot_branch_kd_teacher_temperature "$method_temperature"
        --byot_branch_kd_student_temperature "$method_temperature"
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_proxy_temperature 1.0
        --logdir "$LOG_ROOT" --log_file_name "${rel_dir}/${name}"
    )

    echo "[GPU ${gpu}] start: ${partition} | alpha=${alpha} | feature_beta=${FEATURE_BETA}"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run][GPU %s] ' "$gpu"
        printf '%q ' "${command[@]}"
        printf '\n'
        return 0
    fi

    if ! "${command[@]}" > "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${partition} | alpha=${alpha}" >&2
        tail -40 "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] incomplete: ${name}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${partition} | alpha=${alpha}"
}

JOBS=()
for partition in "${PARTITIONS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        JOBS+=("${partition}|${alpha}")
    done
done

echo "========== TinyImageNet feature-interaction control =========="
echo "GPUs=${GPUS[*]} | jobs=${#JOBS[@]} | seed=${SEED}"
echo "partitions=${PARTITIONS[*]} | alphas=${ALPHAS[*]}"
echo "R=${ROUNDS} | E=${LOCAL_EPOCHS} | CE T=${CE_TEMPERATURE} (inactive) | KD T=${KD_TEMPERATURE}"
echo "feature_beta=${FEATURE_BETA} | min_require_size=${MIN_REQUIRE_SIZE}"
echo "paired_resnet_init=${PAIRED_RESNET_INIT} | paired_execution_rng=1"
echo "log_root=${LOG_ROOT}"

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
    (
        failed=0
        for ((j=i; j<${#JOBS[@]}; j+=NUM_GPUS)); do
            IFS='|' read -r partition alpha <<< "${JOBS[$j]}"
            run_job "${GPUS[$i]}" "$partition" "$alpha" || failed=1
        done
        exit "$failed"
    ) &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

if (( status != 0 )); then
    echo "One or more feature-interaction jobs failed." >&2
    exit "$status"
fi
echo "All TinyImageNet feature-interaction controls completed."
