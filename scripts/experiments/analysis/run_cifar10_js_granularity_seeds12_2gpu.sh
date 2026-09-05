#!/usr/bin/env bash

# Two-seed replication of the full-feature JS-client vs JS-branch comparison.
# The settings match the seed-0 runs produced by run_kd_need_proxy_seed0_4gpu.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS=(${GPUS_OVERRIDE:-0 1})
if (( ${#GPUS[@]} != 2 )); then
    echo "This launcher requires exactly two GPU ids; received: ${GPUS[*]:-(none)}" >&2
    exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

SEEDS=(${SEEDS_OVERRIDE:-1 2})
METHODS=(${METHODS_OVERRIDE:-js_client js})
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"

FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.5}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
JS_GAIN="${JS_GAIN_OVERRIDE:-1.0}"
NEED_MIN_GATE="${NEED_MIN_GATE:-0.0}"
WARMUP_ROUNDS="$(awk -v rounds="$ROUNDS" -v ratio="$WARMUP_RATIO" \
    'BEGIN { printf "%d", int(rounds * ratio + 0.5) }')"

LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_cifar10_js_granularity_seeds12}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

has_completed_run() {
    local log_file="$1" pkl_file="$2"
    [[ -f "$log_file" ]] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file" \
        && [[ -f "$pkl_file" ]]
}

run_job() {
    local gpu_id="$1" seed="$2" method="$3"
    local run_name log_name log_dir log_file pkl_file proxy
    local -a cmd

    case "$method" in
        js_client) proxy="js_client" ;;
        js|js_branch) proxy="js" ;;
        *) echo "Unknown method: $method" >&2; return 1 ;;
    esac

    run_name="cifar10_beta_0.1_${method}_seed${seed}_r${ROUNDS}"
    log_name="runs/${method}/cifar10/beta_0.1/seed${seed}/${run_name}"
    log_dir="${LOG_ROOT}/runs/${method}/cifar10/beta_0.1/seed${seed}"
    log_file="${LOG_ROOT}/${log_name}.log"
    pkl_file="${LOG_ROOT}/${log_name}.pkl"
    mkdir -p "$log_dir"

    if [[ "$SKIP_EXISTING" == "1" ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu_id}] skip: ${method} | seed=${seed}"
        return 0
    fi

    cmd=(
        "$PYTHON_BIN" main.py
        --dataset cifar10 --datadir ./data --in_channels 3 --num_classes 10
        --partition noniid --beta 0.1
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$seed" --device "cuda:${gpu_id}"
        --logdir "$LOG_ROOT" --log_file_name "$log_name"
        --sequential_client_execution
        --paired_resnet_init --paired_execution_rng --preserve_byot_proxy_rng
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
        --byot_branch_need_proxy "$proxy"
        --byot_branch_need_gain "$JS_GAIN"
        --byot_branch_need_min_gate "$NEED_MIN_GATE"
        --byot_branch_need_temperature "$PROXY_TEMPERATURE"
    )

    echo "[GPU ${gpu_id}] start: ${method} | CIFAR-10 beta=0.1 | seed=${seed}"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run][GPU %s] ' "$gpu_id"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    if ! "${cmd[@]}" > "${log_dir}/${run_name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu_id}] failed: ${method} | seed=${seed}" >&2
        tail -40 "${log_dir}/${run_name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu_id}] incomplete: ${method} | seed=${seed}" >&2
        return 1
    fi
    echo "[GPU ${gpu_id}] complete: ${method} | seed=${seed}"
}

# Interleave methods so each same-seed pair starts together on separate GPUs.
JOBS=()
for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        JOBS+=("${seed}|${method}")
    done
done

echo "========== CIFAR-10 JS granularity seed replication =========="
echo "GPUs=${GPUS[*]} | seeds=${SEEDS[*]} | methods=${METHODS[*]} | jobs=${#JOBS[@]}"
echo "partition=beta_0.1 | rounds=${ROUNDS} | local_epochs=${LOCAL_EPOCHS} | warmup=${WARMUP_ROUNDS}"
echo "feature_beta=${FEATURE_BETA} | JS_gain=${JS_GAIN} | log_root=${LOG_ROOT}"
echo "estimated 2-GPU wall time: about 4.5-6 hours"

run_queue() {
    local gpu_index="$1" gpu_id="${GPUS[$1]}" job_index seed method failed=0
    for ((job_index = gpu_index; job_index < ${#JOBS[@]}; job_index += 2)); do
        IFS='|' read -r seed method <<< "${JOBS[$job_index]}"
        run_job "$gpu_id" "$seed" "$method" || failed=1
    done
    return "$failed"
}

pids=()
for gpu_index in 0 1; do
    run_queue "$gpu_index" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

if (( status != 0 )); then
    echo "At least one JS granularity replication run failed." >&2
    exit "$status"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete."
else
    echo "CIFAR-10 JS granularity seed replication complete."
fi
