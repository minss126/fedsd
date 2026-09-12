#!/usr/bin/env bash

# TinyImageNet no-feature screen for:
#   1) CE/KD blend alpha: L_branch=(1-alpha) CE + alpha KD
#   2) fixed lambda=.3 KD
#   3) finalized local-teacher JS-branch adaptive KD
#
# Alpha=0 is temperature-independent and is run once. Alpha=1,
# fixed lambda=.3, and the adaptive method are evaluated at both KD
# temperatures. Default matrix:
#   partitions={IID,beta=.1}, alpha={0,1}, fixed=.3, T_KD={.5,1} (14 jobs).
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
PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.1})
ALPHAS=(${ALPHAS_OVERRIDE:-0.0 1.0})
KD_TEMPERATURES=(${KD_TEMPERATURES_OVERRIDE:-0.5 1.0})

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-$((ROUNDS / 2))}"
LR="${LR:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"

PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
FIXED_LAMBDA="${FIXED_LAMBDA:-0.3}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
JS_GAIN="${JS_GAIN:-1.0}"

LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_tinyimagenet_alpha_adaptive_temperature_no_feature}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "TinyImageNet directory is missing: ${DATA_DIR}" >&2
    exit 1
fi
if [[ "$MIN_REQUIRE_SIZE" != 64 ]]; then
    echo "The final protocol requires MIN_REQUIRE_SIZE=64; got ${MIN_REQUIRE_SIZE}." >&2
    exit 1
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

# job: partition|method|alpha|temperature
# alpha=0 is stored with T=1 only and reused in both temperature tables.
JOBS=()
for partition in "${PARTITIONS[@]}"; do
    for temperature in "${KD_TEMPERATURES[@]}"; do
        JOBS+=("${partition}|adaptive|na|${temperature}")
        JOBS+=("${partition}|fixed|${FIXED_LAMBDA}|${temperature}")
    done
    for alpha in "${ALPHAS[@]}"; do
        if is_zero "$alpha"; then
            JOBS+=("${partition}|blend|${alpha}|1.0")
        else
            for temperature in "${KD_TEMPERATURES[@]}"; do
                JOBS+=("${partition}|blend|${alpha}|${temperature}")
            done
        fi
    done
done

job_weight() {
    local job="$1" partition method alpha temperature
    IFS='|' read -r partition method alpha temperature <<< "$job"
    # Adaptive performs extra client-level proxy passes.
    [[ "$method" == adaptive ]] && printf 3 || printf 1
}

run_job() {
    local gpu="$1" job="$2" partition method alpha temperature method_tag name rel_dir
    local log_file pkl_file
    local -a partition_flags command method_flags
    IFS='|' read -r partition method alpha temperature <<< "$job"

    if [[ "$method" == blend ]]; then
        method_tag="alpha$(value_tag "$alpha")"
    elif [[ "$method" == fixed ]]; then
        method_tag="fixed_lambda$(value_tag "$alpha")"
    else
        method_tag="adaptive"
    fi
    name="tinyimagenet_${partition}_${method_tag}_tkd$(value_tag "$temperature")_nofeat_seed${SEED}_r${ROUNDS}"
    rel_dir="${partition}/seed${SEED}"
    log_file="${LOG_ROOT}/${rel_dir}/${name}.log"
    pkl_file="${LOG_ROOT}/${rel_dir}/${name}.pkl"
    mkdir -p "${LOG_ROOT}/${rel_dir}"

    if [[ "$SKIP_EXISTING" == 1 ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] skip: ${partition} | ${method_tag} | T_KD=${temperature}"
        return 0
    fi

    method_flags=()
    if [[ "$method" == blend ]]; then
        method_flags=(
            --byot_branch_objective blend
            --byot_alpha "$alpha"
        )
    elif [[ "$method" == fixed ]]; then
        # Constant KD coefficient from round 0: no adaptive components.
        method_flags=(
            --byot_branch_objective kd_only
            --byot_alpha "$alpha"
        )
    else
        method_flags=(
            --byot_branch_objective kd_only
            --byot_alpha "$LAMBDA_MAX" --alpha_min_scale 0.0
            --byot_round_lambda_schedule linear
            --byot_round_lambda_min 0.0
            --byot_round_lambda_warmup "$WARMUP_ROUNDS"
            --byot_client_proxy teacher_label_prob
            --byot_client_alpha_min 0.0 --byot_client_alpha_max 1.0
            --byot_client_alpha_mode multiply
            --byot_client_reliability_power 1.0
            --byot_client_skew_proxy prediction_entropy
            --byot_client_skew_power "$SKEW_POWER"
            --byot_client_skew_min_scale 0.0
            --byot_client_skew_correction_mode soft_relax
            --byot_client_skew_soft_tau "$SOFT_TAU"
            --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
            --byot_branch_need_proxy js
            --byot_branch_need_gain "$JS_GAIN"
            --byot_branch_need_min_gate 0.0
            --byot_branch_need_temperature "$PROXY_TEMPERATURE"
        )
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
        --sequential_client_execution --paired_execution_rng --paired_resnet_init
        --preserve_byot_proxy_rng
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_beta 0.0 --byot_teacher_source local
        --temperature "$temperature"
        --byot_branch_kd_teacher_temperature "$temperature"
        --byot_branch_kd_student_temperature "$temperature"
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
        "${method_flags[@]}"
        --logdir "$LOG_ROOT" --log_file_name "${rel_dir}/${name}"
    )

    echo "[GPU ${gpu}] start: ${partition} | ${method_tag} | T_KD=${temperature}"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run][GPU %s] ' "$gpu"
        printf '%q ' "${command[@]}"
        printf '\n'
        return 0
    fi

    if ! "${command[@]}" > "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${partition} | ${method_tag} | T_KD=${temperature}" >&2
        tail -40 "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] incomplete: ${name}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${partition} | ${method_tag} | T_KD=${temperature}"
}

# Greedy weighted placement keeps the expensive adaptive jobs balanced.
declare -a QUEUES LOADS
for ((i=0; i<NUM_GPUS; i++)); do QUEUES[$i]=''; LOADS[$i]=0; done
for job in "${JOBS[@]}"; do
    target=0
    for ((i=1; i<NUM_GPUS; i++)); do
        (( LOADS[i] < LOADS[target] )) && target=$i
    done
    weight="$(job_weight "$job")"
    QUEUES[$target]+="$job"$'\n'
    LOADS[$target]=$((LOADS[$target] + weight))
done

echo "========== TinyImageNet alpha/adaptive temperature screen =========="
echo "GPUs=${GPUS[*]} | jobs=${#JOBS[@]} | seed=${SEED}"
echo "partitions=${PARTITIONS[*]} | alphas=${ALPHAS[*]} | T_KD=${KD_TEMPERATURES[*]}"
echo "R=${ROUNDS} | E=${LOCAL_EPOCHS} | adaptive warm-up=${WARMUP_ROUNDS}"
echo "feature_beta=0 | min_require_size=64 | proxy_temperature=${PROXY_TEMPERATURE}"
echo "fixed_lambda=${FIXED_LAMBDA} | adaptive: lambda_max=${LAMBDA_MAX}, tau=${SOFT_TAU}, JS gain=${JS_GAIN}"
echo "log_root=${LOG_ROOT}"
for ((i=0; i<NUM_GPUS; i++)); do
    count=$(printf '%s' "${QUEUES[$i]}" | sed '/^$/d' | wc -l)
    echo "GPU ${GPUS[$i]}: ${count} jobs (relative load ${LOADS[$i]})"
done

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
    [[ -n "${QUEUES[$i]}" ]] || continue
    (
        failed=0
        while IFS= read -r job; do
            [[ -n "$job" ]] || continue
            run_job "${GPUS[$i]}" "$job" || failed=1
        done <<< "${QUEUES[$i]}"
        exit "$failed"
    ) &
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

echo "TinyImageNet alpha/adaptive temperature screen complete (${#JOBS[@]} jobs)."
