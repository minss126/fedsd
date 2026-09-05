#!/usr/bin/env bash

# Feature-imitation ablation for the selected soft-b adaptive KD method.
#
# Dataset / partitions:
#   CIFAR-100, IID and Dirichlet beta=0.1
#
# Paired methods:
#   plain                : main CE only, no BYOT branches
#   feature_only         : main CE + feature imitation, no branch KD/CE
#   adaptive_no_feature  : soft-b adaptive branch KD, feature loss disabled
#   adaptive_full        : soft-b adaptive branch KD + feature imitation
#
# The corrected partition/training protocol is explicit in every run:
#   min_require_size=10, no redistribution fallback, keep incomplete batches.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if (( ${#GPUS[@]} == 0 )); then
    echo "Set GPUS_OVERRIDE to at least one GPU id." >&2
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
PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.1})
METHODS=(${METHODS_OVERRIDE:-plain feature_only adaptive_no_feature adaptive_full})

ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-10}"

FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.5}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
WARMUP_ROUNDS="$(awk -v rounds="$ROUNDS" -v ratio="$WARMUP_RATIO" \
    'BEGIN { printf "%d", int(rounds * ratio + 0.5) }')"

LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_cifar100_feature_effect_min10_seed0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

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
        && [[ -f "$pkl_file" ]]
}

append_adaptive_flags() {
    local -n target="$1"
    target+=(
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
    )
}

run_job() {
    local gpu_id="$1" partition="$2" method="$3"
    local run_name log_name log_dir log_file pkl_file
    local -a cmd partition_flags

    run_name="cifar100_${partition}_${method}_min${MIN_REQUIRE_SIZE}_seed${SEED}_r${ROUNDS}"
    log_name="runs/${method}/cifar100/${partition}/seed${SEED}/${run_name}"
    log_dir="${LOG_ROOT}/runs/${method}/cifar100/${partition}/seed${SEED}"
    log_file="${LOG_ROOT}/${log_name}.log"
    pkl_file="${LOG_ROOT}/${log_name}.pkl"
    mkdir -p "$log_dir"

    if [[ "$SKIP_EXISTING" == "1" ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu_id}] skip: ${method} | ${partition}"
        return 0
    fi

    mapfile -t partition_flags < <(partition_args "$partition")
    cmd=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data --in_channels 3 --num_classes 100
        "${partition_flags[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --client_keep_last_batch
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$SEED" --device "cuda:${gpu_id}"
        --logdir "$LOG_ROOT" --log_file_name "$log_name"
        --sequential_client_execution
        --paired_resnet_init --paired_execution_rng --preserve_byot_proxy_rng
    )

    case "$method" in
        plain)
            cmd+=(--model resnet18 --alg fedavg)
            ;;
        feature_only)
            cmd+=(
                --model resnet18_byot --alg fedbyot
                --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
                --byot_branch_objective feature_only
                --byot_alpha 0.0 --byot_beta "$FEATURE_BETA"
                --temperature "$KD_TEMPERATURE"
                --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
                --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
                --byot_proxy_temperature "$PROXY_TEMPERATURE"
            )
            ;;
        adaptive_no_feature|adaptive_full)
            cmd+=(
                --model resnet18_byot --alg fedbyot
                --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
                --byot_branch_objective kd_only
                --temperature "$KD_TEMPERATURE"
                --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
                --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
                --byot_proxy_temperature "$PROXY_TEMPERATURE"
            )
            if [[ "$method" == "adaptive_full" ]]; then
                cmd+=(--byot_beta "$FEATURE_BETA")
            else
                cmd+=(--byot_beta 0.0)
            fi
            append_adaptive_flags cmd
            ;;
        *)
            echo "Unknown method: $method" >&2
            return 1
            ;;
    esac

    echo "[GPU ${gpu_id}] start: ${method} | CIFAR-100 ${partition} | seed=${SEED}"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run][GPU %s] ' "$gpu_id"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    if ! "${cmd[@]}" > "${log_dir}/${run_name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu_id}] failed: ${method} | ${partition}" >&2
        tail -40 "${log_dir}/${run_name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu_id}] incomplete: ${method} | ${partition}" >&2
        return 1
    fi
    echo "[GPU ${gpu_id}] complete: ${method} | ${partition}"
}

JOBS=()
for partition in "${PARTITIONS[@]}"; do
    for method in "${METHODS[@]}"; do
        JOBS+=("${partition}|${method}")
    done
done

echo "========== CIFAR-100 feature-effect pilot =========="
echo "GPUs=${GPUS[*]} | seed=${SEED} | jobs=${#JOBS[@]}"
echo "partitions=${PARTITIONS[*]} | methods=${METHODS[*]}"
echo "min=${MIN_REQUIRE_SIZE} | keep_last_batch=true | fallback=disabled"
echo "rounds=${ROUNDS} | local_epochs=${LOCAL_EPOCHS} | warmup=${WARMUP_ROUNDS}"
echo "feature_beta=${FEATURE_BETA} | log_root=${LOG_ROOT}"

run_queue() {
    local gpu_index="$1" gpu_id="${GPUS[$1]}" job_index partition method failed=0
    for ((job_index = gpu_index; job_index < ${#JOBS[@]}; job_index += ${#GPUS[@]})); do
        IFS='|' read -r partition method <<< "${JOBS[$job_index]}"
        run_job "$gpu_id" "$partition" "$method" || failed=1
    done
    return "$failed"
}

pids=()
for ((index = 0; index < ${#GPUS[@]} && index < ${#JOBS[@]}; index++)); do
    run_queue "$index" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

if (( status != 0 )); then
    echo "At least one CIFAR-100 feature-effect run failed." >&2
    exit "$status"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete."
else
    echo "CIFAR-100 feature-effect pilot complete."
fi
