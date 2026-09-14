#!/usr/bin/env bash

# CIFAR-10 beta=.1 sanity checks before the full no-feature rerun.
#
# Four-GPU server jobs:
#   1) seed0 fixed lambda=.03, no warm-up
#   2) seed0 fixed lambda=.10, no warm-up
#   3) seed0 fixed lambda=.30, 50% warm-up (diagnostic only)
#
# With RUN_SET=server2, the same runner instead launches:
#   4) seed1 adaptive without a JS need gate
#   5) seed1 adaptive with the client-wise JS need gate
#
# The canonical protocol deliberately excludes paired_resnet_init,
# paired_execution_rng, and preserve_byot_proxy_rng.

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

LOG_ROOT="${LOG_ROOT:-logs/lambda/final/logs_cifar10_fixed_jsclient_sanity_canonical_no_feature}"
RUN_SET="${RUN_SET:-server4}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-$((ROUNDS / 2))}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

[[ "$ROUNDS" == 500 ]] || {
    echo "This sanity check requires ROUNDS=500; got ${ROUNDS}." >&2
    exit 1
}
[[ "$LOCAL_EPOCHS" == 5 ]] || {
    echo "This sanity check requires LOCAL_EPOCHS=5; got ${LOCAL_EPOCHS}." >&2
    exit 1
}
[[ "$WARMUP_ROUNDS" == 250 ]] || {
    echo "This sanity check requires WARMUP_ROUNDS=250; got ${WARMUP_ROUNDS}." >&2
    exit 1
}
[[ "$MIN_REQUIRE_SIZE" == 64 ]] || {
    echo "This sanity check requires MIN_REQUIRE_SIZE=64; got ${MIN_REQUIRE_SIZE}." >&2
    exit 1
}

has_completed_run() {
    local log_file="$1" pkl_file="$2"
    [[ -f "$log_file" ]] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file" \
        && [[ -s "$pkl_file" ]]
}

# job format: kind|seed|lambda|warmup|need_proxy|estimated_minutes
case "$RUN_SET" in
    server4)
        JOBS=(
            "fixed_lambda0p03|0|0.03|0|none|130"
            "fixed_lambda0p10|0|0.10|0|none|130"
            "fixed_lambda0p30_warmup|0|0.30|250|none|135"
        )
        ;;
    server2)
        JOBS=(
            "adaptive_no_js|1|1.0|250|none|170"
            "adaptive_js_client|1|1.0|250|js_client|180"
        )
        ;;
    all)
        JOBS=(
            "adaptive_js_client|1|1.0|250|js_client|180"
            "adaptive_no_js|1|1.0|250|none|170"
            "fixed_lambda0p30_warmup|0|0.30|250|none|135"
            "fixed_lambda0p03|0|0.03|0|none|130"
            "fixed_lambda0p10|0|0.10|0|none|130"
        )
        ;;
    *)
        echo "Unknown RUN_SET=${RUN_SET}; expected server4, server2, or all." >&2
        exit 1
        ;;
esac

run_job() {
    local gpu="$1" job="$2"
    local kind seed lambda warmup need_proxy estimated name rel_dir log_file pkl_file
    local -a cmd schedule_args adaptive_args
    IFS='|' read -r kind seed lambda warmup need_proxy estimated <<< "$job"

    name="cifar10_beta_0.1_${kind}_tkd1p00_canonical_nofeat_seed${seed}_r${ROUNDS}"
    if [[ "$kind" == fixed_* ]]; then
        rel_dir="fixed/seed${seed}/${kind}"
    else
        rel_dir="adaptive/seed${seed}/${kind}"
    fi
    log_file="${LOG_ROOT}/${rel_dir}/${name}.log"
    pkl_file="${LOG_ROOT}/${rel_dir}/${name}.pkl"
    mkdir -p "${LOG_ROOT}/${rel_dir}"

    if [[ "$SKIP_EXISTING" == 1 ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] skip: ${kind} | seed=${seed}"
        return 0
    fi

    if (( warmup > 0 )); then
        schedule_args=(
            --byot_round_lambda_schedule linear
            --byot_round_lambda_min 0.0
            --byot_round_lambda_warmup "$warmup"
        )
    else
        schedule_args=(
            --byot_round_lambda_schedule none
            --byot_round_lambda_min 0.0
            --byot_round_lambda_warmup 0
        )
    fi

    adaptive_args=()
    if [[ "$kind" == adaptive_* ]]; then
        adaptive_args=(
            --byot_client_proxy teacher_label_prob
            --byot_client_alpha_min 0.0
            --byot_client_alpha_max 1.0
            --byot_client_alpha_mode multiply
            --byot_client_reliability_power 1.0
            --byot_client_skew_proxy prediction_entropy
            --byot_client_skew_power 2.0
            --byot_client_skew_min_scale 0.0
            --byot_client_skew_correction_mode soft_relax
            --byot_client_skew_soft_tau 0.85
            --byot_client_skew_soft_temperature 0.05
            --byot_branch_need_proxy "$need_proxy"
            --byot_branch_need_gain 1.0
            --byot_branch_need_min_gate 0.0
            --byot_branch_need_temperature 1.0
        )
    else
        adaptive_args=(
            --byot_client_proxy none
            --byot_client_skew_proxy none
            --byot_branch_need_proxy none
        )
    fi

    cmd=(
        "$PYTHON_BIN" main.py
        --dataset cifar10 --datadir ./data --in_channels 3 --num_classes 10
        --partition noniid --beta 0.1 --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients 100 --sample_fraction 0.1
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS"
        --optimizer sgd --lr 0.1 --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size 64 --test_batch_size 512 --num_workers 0
        --seed "$seed" --device "cuda:${gpu}" --sequential_client_execution
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta 0.0
        --byot_teacher_source local --byot_alpha "$lambda" --alpha_min_scale 0.0
        --temperature 1.0
        --byot_branch_kd_teacher_temperature 1.0
        --byot_branch_kd_student_temperature 1.0
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_proxy_temperature 1.0
        "${schedule_args[@]}"
        "${adaptive_args[@]}"
        --logdir "$LOG_ROOT" --log_file_name "${rel_dir}/${name}"
    )

    echo "[GPU ${gpu}] start: ${kind} | seed=${seed} | lambda=${lambda} | warm-up=${warmup}"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run][GPU %s] ' "$gpu"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    if ! "${cmd[@]}" > "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${kind} | seed=${seed}" >&2
        tail -40 "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] incomplete: ${kind} | seed=${seed}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${kind} | seed=${seed}"
}

echo "========== CIFAR-10 fixed/JS-client sanity checks =========="
echo "run_set=${RUN_SET} | GPUs=${GPUS[*]} | jobs=${#JOBS[@]}"
echo "CIFAR-10 / beta=.1 / ResNet18-BYOT / FedAvg / R=500 / E=5 / participation=.1"
echo "KD-only | T_KD=1 | T_proxy=1 | feature_beta=0 | min_require_size=64"
echo "fixed={lambda .03, lambda .10, diagnostic lambda .30 + 250-round warm-up}"
echo "adaptive seed1={No-JS, JS-client} | lambda_max=1 | tau=.85 | warm-up=250"
echo "canonical execution: paired/preserved RNG controls are absent"
echo "log_root=${LOG_ROOT}"

# Longest-processing-time queue assignment keeps the four-GPU wall time low
# while still supporting any GPUS_OVERRIDE length.
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
    IFS='|' read -r _ _ _ _ _ estimated <<< "$job"
    LOADS[$target]=$((LOADS[$target] + estimated))
done
echo "estimated_queue_minutes=${LOADS[*]}"

run_queue() {
    local gpu_index="$1" gpu job failed=0
    gpu="${GPUS[$gpu_index]}"
    while IFS= read -r job; do
        [[ -n "$job" ]] || continue
        run_job "$gpu" "$job" || failed=1
    done <<< "${QUEUES[$gpu_index]}"
    return "$failed"
}

if [[ "$DRY_RUN" == 1 ]]; then
    for ((i=0; i<${#GPUS[@]}; i++)); do
        run_queue "$i"
    done
    echo "Dry run complete."
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

echo "CIFAR-10 fixed/JS-client sanity checks complete (${#JOBS[@]} jobs)."
