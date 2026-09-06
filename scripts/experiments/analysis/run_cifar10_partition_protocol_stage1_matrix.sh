#!/usr/bin/env bash

# Shared runner for the first-stage CIFAR-10 partition-protocol comparison.
#
# Required environment variables:
#   PROTOCOL_OVERRIDE=min10|balanced500
#   PARTITION_SEED_SPECS_OVERRIDE="<beta>:<seed> ..."
#
# The public 2-GPU and 4-GPU launchers set these variables for the planned
# split.  This file remains directly reusable for later seed expansion.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PROTOCOL="${PROTOCOL_OVERRIDE:?Set PROTOCOL_OVERRIDE to min10 or balanced500}"
case "$PROTOCOL" in
    min10) PARTITION="noniid"; PROTOCOL_LABEL="min10" ;;
    balanced500) PARTITION="noniid_balanced"; PROTOCOL_LABEL="balanced500" ;;
    *) echo "Unknown PROTOCOL_OVERRIDE: $PROTOCOL" >&2; exit 1 ;;
esac

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3}"
read -r -a SPECS <<< "${PARTITION_SEED_SPECS_OVERRIDE:?Set PARTITION_SEED_SPECS_OVERRIDE}"
read -r -a METHODS <<< "${METHODS_OVERRIDE:-plain current_adaptive js_client js_branch}"
(( ${#GPUS[@]} > 0 )) || { echo "Set GPUS_OVERRIDE to at least one GPU id." >&2; exit 1; }
(( ${#SPECS[@]} > 0 )) || { echo "No beta:seed specifications were supplied." >&2; exit 1; }

if [[ -n "${PYTHON_BIN:-}" ]]; then :
elif [[ -x venv/bin/python ]]; then PYTHON_BIN="venv/bin/python"
else PYTHON_BIN="python3"
fi

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
JS_GAIN="${JS_GAIN_OVERRIDE:-1.0}"
WARMUP_ROUNDS="$(awk -v rounds="$ROUNDS" -v ratio="$WARMUP_RATIO" \
    'BEGIN { printf "%d", int(rounds * ratio + 0.5) }')"

LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_cifar10_${PROTOCOL_LABEL}_partition_stage1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

has_completed_run() {
    local log_file="$1" pkl_file="$2"
    [[ -f "$log_file" ]] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file" \
        && [[ -f "$pkl_file" ]]
}

append_adaptive_flags() {
    local -n target="$1"
    target+=(
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
    )
}

run_job() {
    local gpu_id="$1" beta="$2" seed="$3" method="$4"
    local partition_label="beta_${beta}"
    local run_name="cifar10_${partition_label}_${PROTOCOL_LABEL}_${method}_seed${seed}_r${ROUNDS}"
    local log_name="runs/${method}/cifar10/${partition_label}_${PROTOCOL_LABEL}/seed${seed}/${run_name}"
    local log_dir="${LOG_ROOT}/runs/${method}/cifar10/${partition_label}_${PROTOCOL_LABEL}/seed${seed}"
    local log_file="${LOG_ROOT}/${log_name}.log"
    local pkl_file="${LOG_ROOT}/${log_name}.pkl"
    local proxy=""
    local -a cmd

    mkdir -p "$log_dir"
    if [[ "$SKIP_EXISTING" == "1" ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu_id}] skip: ${PROTOCOL_LABEL} | beta=${beta} | seed=${seed} | ${method}"
        return 0
    fi

    cmd=(
        "$PYTHON_BIN" main.py
        --dataset cifar10 --datadir ./data --in_channels 3 --num_classes 10
        --partition "$PARTITION" --beta "$beta" --min_require_size "$MIN_REQUIRE_SIZE"
        --client_keep_last_batch
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$seed" --device "cuda:${gpu_id}"
        --logdir "$LOG_ROOT" --log_file_name "$log_name"
        --sequential_client_execution
        --paired_resnet_init --paired_execution_rng --preserve_byot_proxy_rng
    )

    case "$method" in
        plain)
            cmd+=(--model resnet18 --alg fedavg)
            ;;
        current_adaptive|js_client|js_branch)
            append_adaptive_flags cmd
            case "$method" in
                js_client) proxy="js_client" ;;
                js_branch) proxy="js" ;;
            esac
            if [[ -n "$proxy" ]]; then
                cmd+=(
                    --byot_branch_need_proxy "$proxy"
                    --byot_branch_need_gain "$JS_GAIN"
                    --byot_branch_need_min_gate 0.0
                    --byot_branch_need_temperature "$PROXY_TEMPERATURE"
                )
            fi
            ;;
        *) echo "Unknown method: $method" >&2; return 1 ;;
    esac

    echo "[GPU ${gpu_id}] start: ${PROTOCOL_LABEL} | beta=${beta} | seed=${seed} | ${method}"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run][GPU %s] ' "$gpu_id"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    if ! "${cmd[@]}" > "${log_dir}/${run_name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu_id}] failed: ${PROTOCOL_LABEL} | beta=${beta} | seed=${seed} | ${method}" >&2
        tail -40 "${log_dir}/${run_name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu_id}] incomplete: ${run_name}" >&2
        return 1
    fi
    echo "[GPU ${gpu_id}] complete: ${PROTOCOL_LABEL} | beta=${beta} | seed=${seed} | ${method}"
}

# Rotate method order for each beta:seed specification.  With the planned
# 2-GPU/4-GPU launchers this gives every worker a comparable mix of the faster
# Plain job and the three BYOT jobs instead of leaving a long BYOT-only tail.
JOBS=()
for ((spec_index = 0; spec_index < ${#SPECS[@]}; spec_index++)); do
    IFS=':' read -r beta seed <<< "${SPECS[$spec_index]}"
    [[ -n "$beta" && -n "$seed" ]] || { echo "Invalid beta:seed spec: ${SPECS[$spec_index]}" >&2; exit 1; }
    for ((method_index = 0; method_index < ${#METHODS[@]}; method_index++)); do
        rotated_index=$(( (method_index + spec_index) % ${#METHODS[@]} ))
        JOBS+=("${beta}|${seed}|${METHODS[$rotated_index]}")
    done
done

echo "========== CIFAR-10 partition protocol stage 1: ${PROTOCOL_LABEL} =========="
echo "GPUs=${GPUS[*]} | specs=${SPECS[*]} | methods=${METHODS[*]} | jobs=${#JOBS[@]}"
echo "R=${ROUNDS} | E=${LOCAL_EPOCHS} | warmup=${WARMUP_ROUNDS} | min=${MIN_REQUIRE_SIZE} | keep_last=true"
echo "log_root=${LOG_ROOT} | skip_existing=${SKIP_EXISTING}"

run_queue() {
    local gpu_index="$1" gpu_id="${GPUS[$1]}" job_index beta seed method failed=0
    for ((job_index = gpu_index; job_index < ${#JOBS[@]}; job_index += ${#GPUS[@]})); do
        IFS='|' read -r beta seed method <<< "${JOBS[$job_index]}"
        run_job "$gpu_id" "$beta" "$seed" "$method" || failed=1
    done
    return "$failed"
}

pids=()
for ((index = 0; index < ${#GPUS[@]} && index < ${#JOBS[@]}; index++)); do
    run_queue "$index" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "At least one ${PROTOCOL_LABEL} stage-1 run failed." >&2; exit "$status"; }
[[ "$DRY_RUN" == "1" ]] && echo "Dry run complete." || echo "${PROTOCOL_LABEL} stage-1 matrix complete."

