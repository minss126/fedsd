#!/usr/bin/env bash

# Add JS-client/JS-branch feature on/off pairs to the completed CIFAR-100
# feature-effect experiment. Results share its LOG_ROOT for joint analysis.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
(( ${#GPUS[@]} > 0 )) || { echo "Set GPUS_OVERRIDE to at least one GPU id." >&2; exit 1; }
if [[ -n "${PYTHON_BIN:-}" ]]; then :
elif [[ -x venv/bin/python ]]; then PYTHON_BIN="venv/bin/python"
else PYTHON_BIN="python3"
fi

SEED="${SEED:-0}"
PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.1})
METHODS=(${METHODS_OVERRIDE:-js_client_no_feature js_client_full js_branch_no_feature js_branch_full})
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
WARMUP_ROUNDS="$(awk -v rounds="$ROUNDS" -v ratio="$WARMUP_RATIO" 'BEGIN {printf "%d",int(rounds*ratio+0.5)}')"
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
has_completed_run() { [[ -f "$1" ]] && grep -q "Round $((ROUNDS-1)) result" "$1" && [[ -f "$2" ]]; }

run_job() {
    local gpu="$1" partition="$2" method="$3" proxy feature_beta
    case "$method" in
        js_client_no_feature) proxy=js_client; feature_beta=0.0 ;;
        js_client_full) proxy=js_client; feature_beta="$FEATURE_BETA" ;;
        js_branch_no_feature) proxy=js; feature_beta=0.0 ;;
        js_branch_full) proxy=js; feature_beta="$FEATURE_BETA" ;;
        *) echo "Unknown method: $method" >&2; return 1 ;;
    esac
    local name="cifar100_${partition}_${method}_min${MIN_REQUIRE_SIZE}_seed${SEED}_r${ROUNDS}"
    local log_name="runs/${method}/cifar100/${partition}/seed${SEED}/${name}"
    local dir="${LOG_ROOT}/runs/${method}/cifar100/${partition}/seed${SEED}"
    local log="${LOG_ROOT}/${log_name}.log" pkl="${LOG_ROOT}/${log_name}.pkl"
    local -a part cmd
    mkdir -p "$dir"
    if [[ "$SKIP_EXISTING" == 1 ]] && has_completed_run "$log" "$pkl"; then
        echo "[GPU ${gpu}] skip: ${method} | ${partition}"; return 0
    fi
    mapfile -t part < <(partition_args "$partition")
    cmd=(
        "$PYTHON_BIN" main.py --dataset cifar100 --datadir ./data --in_channels 3 --num_classes 100
        "${part[@]}" --min_require_size "$MIN_REQUIRE_SIZE" --client_keep_last_batch
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$SEED" --device "cuda:${gpu}"
        --logdir "$LOG_ROOT" --log_file_name "$log_name"
        --sequential_client_execution --paired_resnet_init --paired_execution_rng --preserve_byot_proxy_rng
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta "$feature_beta"
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
        --byot_branch_need_proxy "$proxy" --byot_branch_need_gain "$JS_GAIN"
        --byot_branch_need_min_gate 0.0 --byot_branch_need_temperature "$PROXY_TEMPERATURE"
    )
    echo "[GPU ${gpu}] start: ${method} | ${partition} | feature_beta=${feature_beta}"
    if [[ "$DRY_RUN" == 1 ]]; then printf '[dry-run] '; printf '%q ' "${cmd[@]}"; printf '\n'; return 0; fi
    if ! "${cmd[@]}" > "${dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${method} | ${partition}" >&2
        tail -40 "${dir}/${name}_terminal.log" >&2 || true; return 1
    fi
    has_completed_run "$log" "$pkl" || { echo "Incomplete: $name" >&2; return 1; }
    echo "[GPU ${gpu}] complete: ${method} | ${partition}"
}

JOBS=(); for partition in "${PARTITIONS[@]}"; do for method in "${METHODS[@]}"; do JOBS+=("$partition|$method"); done; done
echo "========== CIFAR-100 JS feature-effect extension =========="
echo "GPUs=${GPUS[*]} | partitions=${PARTITIONS[*]} | methods=${METHODS[*]} | jobs=${#JOBS[@]}"
echo "seed=${SEED} | min=${MIN_REQUIRE_SIZE} | keep_last=true | R=${ROUNDS} | E=${LOCAL_EPOCHS}"
run_queue() {
    local gi="$1" gpu="${GPUS[$1]}" ji partition method failed=0
    for ((ji=gi;ji<${#JOBS[@]};ji+=${#GPUS[@]})); do
        IFS='|' read -r partition method <<< "${JOBS[$ji]}"; run_job "$gpu" "$partition" "$method" || failed=1
    done; return "$failed"
}
pids=(); for ((i=0;i<${#GPUS[@]} && i<${#JOBS[@]};i++)); do run_queue "$i" & pids+=("$!"); done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done
((status==0)) || exit "$status"
[[ "$DRY_RUN" == 1 ]] && echo "Dry run complete." || echo "CIFAR-100 JS feature extension complete."

