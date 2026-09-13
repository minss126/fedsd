#!/usr/bin/env bash

# Revalidate the final feature-free, canonical-execution protocol before the
# full seed/extension reruns.
#
# This runner contains two independent screens:
#   adaptive: JS-client one-factor screen around
#             lambda_max=1, soft_tau=.85 (T_KD=1, JS gain=1)
#   fixed:    constant lambda in {.1, .3, .5} (T_KD=1)
#
# Protocol invariants shared by every job:
#   CIFAR-100 / ResNet18-BYOT / FedAvg / R=500 / E=5 / participation=.1
#   branch objective=KD-only / feature loss=0 / proxy temperature=1
#   min_require_size=64 / seed=0 by default
#
# Deliberately absent from every command:
#   --paired_resnet_init
#   --paired_execution_rng
#   --preserve_byot_proxy_rng

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
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
JS_GAIN="${JS_GAIN:-1.0}"

ADAPTIVE_LOG_ROOT="${ADAPTIVE_LOG_ROOT:-logs/lambda/adaptive/logs_js_client_canonical_no_feature_retune}"
FIXED_LOG_ROOT="${FIXED_LOG_ROOT:-logs/lambda/analysis/logs_fixed_lambda_canonical_no_feature_t1_recheck}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$MIN_REQUIRE_SIZE" != 64 ]]; then
    echo "Final protocol requires MIN_REQUIRE_SIZE=64; got ${MIN_REQUIRE_SIZE}." >&2
    exit 1
fi
if [[ "$KD_TEMPERATURE" != 1.0 && "$KD_TEMPERATURE" != 1 ]]; then
    echo "This revalidation fixes KD_TEMPERATURE=1; got ${KD_TEMPERATURE}." >&2
    exit 1
fi
if [[ "$PROXY_TEMPERATURE" != 1.0 && "$PROXY_TEMPERATURE" != 1 ]]; then
    echo "This revalidation fixes PROXY_TEMPERATURE=1; got ${PROXY_TEMPERATURE}." >&2
    exit 1
fi
if [[ "$JS_GAIN" != 1.0 && "$JS_GAIN" != 1 ]]; then
    echo "JS_GAIN is fixed to 1 to avoid redundant scale tuning; got ${JS_GAIN}." >&2
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
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
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

# job format: kind|partition|value
JOBS=()
add_adaptive_jobs() {
    local partition="$1" value
    for value in center lmax0p5 lmax2p0 tau0p80 tau0p90; do
        JOBS+=("adaptive|${partition}|${value}")
    done
}
add_fixed_jobs() {
    local partition="$1" value
    for value in 0.1 0.3 0.5; do
        JOBS+=("fixed|${partition}|${value}")
    done
}

case "$RUN_SET" in
    server4)
        # Keep each partition together and balance measured wall time with the
        # two-GPU allocation (IID jobs are substantially slower than beta=.1).
        add_adaptive_jobs iid
        add_fixed_jobs iid
        add_fixed_jobs beta_0.5
        ;;
    server2)
        # Put the slower beta=.3 fixed block first so greedy queue assignment
        # balances it across both devices. Keep beta=.1 entirely here.
        add_fixed_jobs beta_0.3
        add_adaptive_jobs beta_0.1
        add_fixed_jobs beta_0.1
        ;;
    *)
        echo "Unknown RUN_SET=${RUN_SET}; expected server4 or server2." >&2
        exit 1
        ;;
esac

configure_adaptive_value() {
    local value="$1"
    JOB_LAMBDA_MAX=1.0
    JOB_SOFT_TAU=0.85
    case "$value" in
        center) ;;
        lmax0p5) JOB_LAMBDA_MAX=0.5 ;;
        lmax2p0) JOB_LAMBDA_MAX=2.0 ;;
        tau0p80) JOB_SOFT_TAU=0.80 ;;
        tau0p90) JOB_SOFT_TAU=0.90 ;;
        *) echo "Unknown adaptive value: $value" >&2; return 1 ;;
    esac
}

run_job() {
    local gpu="$1" job="$2" kind partition value name rel_dir log_root log_file pkl_file
    local -a part_flags cmd common
    IFS='|' read -r kind partition value <<< "$job"
    mapfile -t part_flags < <(partition_args "$partition")

    common=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data --in_channels 3 --num_classes 100
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
        --byot_teacher_source local
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
    )

    if [[ "$kind" == adaptive ]]; then
        configure_adaptive_value "$value"
        name="js_client_${value}_lmax$(value_tag "$JOB_LAMBDA_MAX")_tau$(value_tag "$JOB_SOFT_TAU")_tkd1p00_canonical_nofeat_seed${SEED}_r${ROUNDS}"
        rel_dir="cifar100/${partition}/seed${SEED}/${value}"
        log_root="$ADAPTIVE_LOG_ROOT"
        cmd=(
            "${common[@]}"
            --byot_alpha "$JOB_LAMBDA_MAX" --alpha_min_scale 0.0
            --byot_round_lambda_schedule linear
            --byot_round_lambda_min 0.0 --byot_round_lambda_warmup "$WARMUP_ROUNDS"
            --byot_client_proxy teacher_label_prob
            --byot_client_alpha_min 0.0 --byot_client_alpha_max 1.0
            --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
            --byot_client_skew_proxy prediction_entropy
            --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.0
            --byot_client_skew_correction_mode soft_relax
            --byot_client_skew_soft_tau "$JOB_SOFT_TAU"
            --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
            --byot_branch_need_proxy js_client
            --byot_branch_need_gain "$JS_GAIN" --byot_branch_need_min_gate 0.0
            --byot_branch_need_temperature "$PROXY_TEMPERATURE"
        )
    elif [[ "$kind" == fixed ]]; then
        name="fixed_lambda$(value_tag "$value")_tkd1p00_canonical_nofeat_seed${SEED}_r${ROUNDS}"
        rel_dir="cifar100/${partition}/seed${SEED}/lambda$(value_tag "$value")"
        log_root="$FIXED_LOG_ROOT"
        cmd=(
            "${common[@]}"
            --byot_alpha "$value"
            --byot_branch_need_proxy none
        )
    else
        echo "Unknown job kind: $kind" >&2
        return 1
    fi

    log_file="${log_root}/${rel_dir}/${name}.log"
    pkl_file="${log_root}/${rel_dir}/${name}.pkl"
    mkdir -p "${log_root}/${rel_dir}"
    cmd+=(--logdir "$log_root" --log_file_name "${rel_dir}/${name}")

    if [[ "$SKIP_EXISTING" == 1 ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] skip: ${kind} | ${partition} | ${value}"
        return 0
    fi

    echo "[GPU ${gpu}] start: ${kind} | ${partition} | ${value}"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run][GPU %s][%s] ' "$gpu" "$kind"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    if ! "${cmd[@]}" > "${log_root}/${rel_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${kind} | ${partition} | ${value}" >&2
        tail -40 "${log_root}/${rel_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] incomplete: ${kind} | ${partition} | ${value}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${kind} | ${partition} | ${value}"
}

echo "========== JS-client retune + fixed-lambda recheck =========="
echo "run_set=${RUN_SET} | GPUs=${GPUS[*]} | jobs=${#JOBS[@]} | seed=${SEED}"
echo "CIFAR-100 / ResNet18-BYOT / FedAvg / R=${ROUNDS} / E=${LOCAL_EPOCHS} / participation=${SAMPLE_FRACTION}"
echo "KD-only | T_KD=1 | T_proxy=1 | feature_beta=0 | min_require_size=64"
echo "warm-up=${WARMUP_ROUNDS} (adaptive only) | JS-client gain=1"
echo "canonical execution: paired/preserved RNG controls are absent"
echo "adaptive_log_root=${ADAPTIVE_LOG_ROOT}"
echo "fixed_log_root=${FIXED_LOG_ROOT}"

# Balance by measured CIFAR-100 wall time rather than job count.  These values
# affect scheduling only; they are never forwarded to main.py.
job_weight() {
    local job="$1" kind partition _value
    IFS='|' read -r kind partition _value <<< "$job"
    case "${kind}|${partition}" in
        adaptive\|iid) printf '155' ;;
        adaptive\|beta_0.1) printf '62' ;;
        fixed\|iid) printf '113' ;;
        fixed\|beta_0.5) printf '114' ;;
        fixed\|beta_0.3) printf '117' ;;
        fixed\|beta_0.1) printf '49' ;;
        *) printf '120' ;;
    esac
}

declare -a QUEUES LOADS
for ((i=0; i<${#GPUS[@]}; i++)); do
    QUEUES[$i]=''
    LOADS[$i]=0
done
for job in "${JOBS[@]}"; do
    target=0
    for ((i=1; i<${#GPUS[@]}; i++)); do
        if (( LOADS[i] < LOADS[target] )); then
            target=$i
        fi
    done
    QUEUES[$target]+="${job}"$'\n'
    weight="$(job_weight "$job")"
    LOADS[$target]=$((LOADS[$target] + weight))
done
echo "estimated_queue_minutes=${LOADS[*]}"

run_queue() {
    local gpu_index="$1"
    local failed=0
    local gpu="${GPUS[$gpu_index]}"
    local job
    while IFS= read -r job; do
        [[ -n "$job" ]] || continue
        run_job "$gpu" "$job" || failed=1
    done <<< "${QUEUES[$gpu_index]}"
    return "$failed"
}

if [[ "$DRY_RUN" == 1 ]]; then
    # Exercise the same run_queue/run_job path as a real launch, but do it
    # serially so concurrent dry-run output cannot interleave and hide jobs.
    status=0
    for ((i=0; i<${#GPUS[@]} && i<${#JOBS[@]}; i++)); do
        run_queue "$i" || status=1
    done
    if (( status != 0 )); then
        echo "Dry run failed while validating the real queue path." >&2
        exit "$status"
    fi
    echo "Dry run complete (${#JOBS[@]} jobs through the real queue path)."
    exit 0
fi

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

echo "JS-client retune + fixed-lambda recheck complete (${#JOBS[@]} jobs)."
