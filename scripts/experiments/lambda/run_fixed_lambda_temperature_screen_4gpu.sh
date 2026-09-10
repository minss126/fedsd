#!/usr/bin/env bash

# Fixed-lambda KD-temperature screen for the finalized no-feature protocol.
#
# Only T_KD changes. Lambda is constant from round 0, so this launcher does
# not pass any warm-up, reliability, skew-correction, or branch-gating flags.
# Default matrix: CIFAR-100 x {IID, beta=.5, beta=.3, beta=.1} x T={.5, 1}.
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

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"
# FIXED_LAMBDA is retained as a backward-compatible single-value override.
# FIXED_LAMBDAS_OVERRIDE takes precedence and accepts a space-separated grid.
FIXED_LAMBDA="${FIXED_LAMBDA:-0.3}"
FIXED_LAMBDAS=(${FIXED_LAMBDAS_OVERRIDE:-$FIXED_LAMBDA})
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
ENVS=(${ENVS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})
KD_TEMPERATURES=(${KD_TEMPERATURES_OVERRIDE:-0.5 1.0})
LOG_ROOT="${LOG_ROOT:-logs/lambda/analysis/logs_fixed_lambda0p3_temperature_no_feature}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$MIN_REQUIRE_SIZE" != 64 ]]; then
    echo "The final comparison protocol requires MIN_REQUIRE_SIZE=64; got ${MIN_REQUIRE_SIZE}." >&2
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

# job: partition|lambda|temperature
JOBS=()
for env_name in "${ENVS[@]}"; do
    for lambda in "${FIXED_LAMBDAS[@]}"; do
        for temperature in "${KD_TEMPERATURES[@]}"; do
            JOBS+=("${env_name}|${lambda}|${temperature}")
        done
    done
done

run_job() {
    local gpu="$1" job="$2" env_name lambda temperature name rel_dir log_file pkl_file
    local -a partition_flags command
    IFS='|' read -r env_name lambda temperature <<< "$job"

    name="fixed_lambda$(value_tag "$lambda")_tkd$(value_tag "$temperature")_nofeat_seed${SEED}_r${ROUNDS}"
    rel_dir="cifar100/${env_name}/seed${SEED}"
    log_file="${LOG_ROOT}/${rel_dir}/${name}.log"
    pkl_file="${LOG_ROOT}/${rel_dir}/${name}.pkl"
    mkdir -p "${LOG_ROOT}/${rel_dir}"

    if [[ "$SKIP_EXISTING" == 1 ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] skip: ${env_name} | T_KD=${temperature}"
        return 0
    fi

    mapfile -t partition_flags < <(partition_args "$env_name")
    command=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data --in_channels 3 --num_classes 100
        "${partition_flags[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients 100 --sample_fraction 0.1
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS"
        --lr "$LR" --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --seed "$SEED" --device "cuda:${gpu}"
        --sequential_client_execution --paired_execution_rng --paired_resnet_init
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta 0.0
        --byot_teacher_source local --byot_alpha "$lambda"
        --temperature "$temperature"
        --byot_branch_kd_teacher_temperature "$temperature"
        --byot_branch_kd_student_temperature "$temperature"
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
        --logdir "$LOG_ROOT" --log_file_name "${rel_dir}/${name}"
    )

    echo "[GPU ${gpu}] start: ${env_name} | fixed lambda=${lambda} | T_KD=${temperature}"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run][GPU %s] ' "$gpu"
        printf '%q ' "${command[@]}"
        printf '\n'
        return 0
    fi

    if ! "${command[@]}" > "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${env_name} | T_KD=${temperature}" >&2
        tail -40 "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] incomplete: ${env_name} | T_KD=${temperature}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${env_name} | T_KD=${temperature}"
}

declare -a QUEUES LOADS
for ((i=0; i<NUM_GPUS; i++)); do QUEUES[$i]=''; LOADS[$i]=0; done
for job in "${JOBS[@]}"; do
    target=0
    for ((i=1; i<NUM_GPUS; i++)); do
        (( LOADS[i] < LOADS[target] )) && target=$i
    done
    QUEUES[$target]+="$job"$'\n'
    LOADS[$target]=$((LOADS[$target] + 1))
done

echo "========== Fixed-lambda KD-temperature screen =========="
echo "GPUs=${GPUS[*]} | jobs=${#JOBS[@]} | seed=${SEED}"
echo "CIFAR-100 / ResNet18-BYOT / FedAvg / R=${ROUNDS} / E=${LOCAL_EPOCHS}"
echo "partitions=${ENVS[*]} | T_KD=${KD_TEMPERATURES[*]}"
echo "fixed_lambdas=${FIXED_LAMBDAS[*]} | feature_beta=0 | warm-up=none | min_require_size=64"
echo "log_root=${LOG_ROOT}"

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

echo "Fixed-lambda temperature screen complete (${#JOBS[@]} jobs)."
