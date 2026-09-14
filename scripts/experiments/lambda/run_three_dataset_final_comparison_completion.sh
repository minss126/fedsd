#!/usr/bin/env bash

# Complete the final Plain / fixed-lambda / adaptive comparison on
# CIFAR-10, CIFAR-100, and TinyImageNet.
#
# Final protocol:
#   partitions: IID and Dirichlet beta=.1
#   methods:
#     plain    : ResNet18 + FedAvg
#     fixed    : ResNet18-BYOT + KD-only, lambda=.3, no warm-up
#     adaptive : ResNet18-BYOT + KD-only, final JS-client rule
#                (lambda_max=1, tau=.85, 50% round warm-up)
#   feature imitation is disabled for fixed and adaptive.
#   paired/preserved RNG-control flags are deliberately absent.
#
# Work split, based on measured runtime:
#   server4: seed-0 missing TinyImageNet fixed runs (2 jobs by default)
#   server2: seed-0 missing CIFAR-10 fixed runs (2 jobs by default)
#
# Seed 1/2 validation remains available as an explicit opt-in by setting
# INCLUDE_VALIDATION=1, but it is disabled for the current seed-0 comparison.
#
# Completed jobs are skipped. A killed/incomplete job is rerun from its start.

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

DATA_ROOT="${DATA_ROOT:-./data}"
TINYIMAGENET_DATADIR="${TINYIMAGENET_DATADIR:-${DATA_ROOT}/tiny-imagenet-200}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/final/logs_three_dataset_plain_fixed_adaptive}"

LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"

FIXED_LAMBDA="${FIXED_LAMBDA:-0.3}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
ADAPTIVE_LAMBDA_MAX="${ADAPTIVE_LAMBDA_MAX:-1.0}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
JS_GAIN="${JS_GAIN:-1.0}"

INCLUDE_SEED0_FIXED="${INCLUDE_SEED0_FIXED:-1}"
INCLUDE_VALIDATION="${INCLUDE_VALIDATION:-0}"
read -r -a VALIDATION_SEEDS <<< "${VALIDATION_SEEDS_OVERRIDE:-1 2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

[[ "$MIN_REQUIRE_SIZE" == 64 ]] || {
    echo "Final protocol requires MIN_REQUIRE_SIZE=64; got ${MIN_REQUIRE_SIZE}." >&2
    exit 1
}
[[ "$FIXED_LAMBDA" == 0.3 || "$FIXED_LAMBDA" == .3 ]] || {
    echo "Final fixed comparison requires FIXED_LAMBDA=0.3; got ${FIXED_LAMBDA}." >&2
    exit 1
}
[[ "$KD_TEMPERATURE" == 1 || "$KD_TEMPERATURE" == 1.0 ]] || {
    echo "Final protocol fixes KD_TEMPERATURE=1; got ${KD_TEMPERATURE}." >&2
    exit 1
}
[[ "$PROXY_TEMPERATURE" == 1 || "$PROXY_TEMPERATURE" == 1.0 ]] || {
    echo "Final protocol fixes PROXY_TEMPERATURE=1; got ${PROXY_TEMPERATURE}." >&2
    exit 1
}
[[ "$ADAPTIVE_LAMBDA_MAX" == 1 || "$ADAPTIVE_LAMBDA_MAX" == 1.0 ]] || {
    echo "Final protocol fixes ADAPTIVE_LAMBDA_MAX=1; got ${ADAPTIVE_LAMBDA_MAX}." >&2
    exit 1
}
[[ "$SOFT_TAU" == 0.85 || "$SOFT_TAU" == .85 ]] || {
    echo "Final protocol fixes SOFT_TAU=.85; got ${SOFT_TAU}." >&2
    exit 1
}
[[ "$JS_GAIN" == 1 || "$JS_GAIN" == 1.0 ]] || {
    echo "Final protocol fixes JS_GAIN=1; got ${JS_GAIN}." >&2
    exit 1
}

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

configure_dataset() {
    case "$1" in
        cifar10)
            JOB_DATADIR="$DATA_ROOT"
            JOB_NUM_CLASSES=10
            JOB_ROUNDS=500
            JOB_LR=0.1
            JOB_NUM_WORKERS=0
            ;;
        cifar100)
            JOB_DATADIR="$DATA_ROOT"
            JOB_NUM_CLASSES=100
            JOB_ROUNDS=500
            JOB_LR=0.1
            JOB_NUM_WORKERS=0
            ;;
        tinyimagenet)
            JOB_DATADIR="$TINYIMAGENET_DATADIR"
            JOB_NUM_CLASSES=200
            JOB_ROUNDS=100
            JOB_LR=0.01
            JOB_NUM_WORKERS=2
            [[ -d "${JOB_DATADIR}/train" && -d "${JOB_DATADIR}/val" ]] || {
                echo "TinyImageNet directories are missing: ${JOB_DATADIR}/{train,val}" >&2
                return 1
            }
            ;;
        *) echo "Unknown dataset: $1" >&2; return 1 ;;
    esac
    JOB_WARMUP_ROUNDS=$((JOB_ROUNDS / 2))
}

has_completed_run() {
    local log_file="$1" pkl_file="$2" rounds="$3"
    [[ -f "$log_file" ]] \
        && grep -q "Round $((rounds - 1)) result" "$log_file" \
        && [[ -s "$pkl_file" ]]
}

# job format: dataset|partition|method|seed
JOBS=()

add_seed0_fixed_jobs() {
    local dataset="$1"
    JOBS+=(
        "${dataset}|iid|fixed|0"
        "${dataset}|beta_0.1|fixed|0"
    )
}

add_validation_jobs() {
    local dataset="$1" seed partition method
    [[ "$INCLUDE_VALIDATION" == 1 ]] || return 0
    for seed in "${VALIDATION_SEEDS[@]}"; do
        [[ -n "$seed" ]] || continue
        for partition in iid beta_0.1; do
            for method in plain fixed adaptive; do
                JOBS+=("${dataset}|${partition}|${method}|${seed}")
            done
        done
    done
}

case "$RUN_SET" in
    server4)
        if [[ "$INCLUDE_SEED0_FIXED" == 1 ]]; then
            add_seed0_fixed_jobs tinyimagenet
        fi
        add_validation_jobs tinyimagenet
        add_validation_jobs cifar100
        ;;
    server2)
        if [[ "$INCLUDE_SEED0_FIXED" == 1 ]]; then
            add_seed0_fixed_jobs cifar10
        fi
        add_validation_jobs cifar10
        ;;
    *)
        echo "Unknown RUN_SET=${RUN_SET}; expected server4 or server2." >&2
        exit 1
        ;;
esac

run_job() {
    local gpu="$1" job="$2"
    local dataset partition method seed name rel_dir log_file pkl_file
    local -a part_flags common byot_common cmd
    IFS='|' read -r dataset partition method seed <<< "$job"
    configure_dataset "$dataset"
    mapfile -t part_flags < <(partition_args "$partition")

    case "$method" in
        plain)
            name="${dataset}_${partition}_plain_canonical_seed${seed}_r${JOB_ROUNDS}"
            ;;
        fixed)
            name="${dataset}_${partition}_fixed_lambda0p30_tkd1p00_canonical_nofeat_seed${seed}_r${JOB_ROUNDS}"
            ;;
        adaptive)
            name="${dataset}_${partition}_js_client_lmax1p00_tau0p85_tkd1p00_canonical_nofeat_seed${seed}_r${JOB_ROUNDS}"
            ;;
        *) echo "Unknown method: $method" >&2; return 1 ;;
    esac

    rel_dir="${dataset}/${partition}/seed${seed}/${method}"
    log_file="${LOG_ROOT}/${rel_dir}/${name}.log"
    pkl_file="${LOG_ROOT}/${rel_dir}/${name}.pkl"
    mkdir -p "${LOG_ROOT}/${rel_dir}"

    if [[ "$SKIP_EXISTING" == 1 ]] \
        && has_completed_run "$log_file" "$pkl_file" "$JOB_ROUNDS"; then
        echo "[GPU ${gpu}] skip: ${dataset} | ${partition} | ${method} | seed=${seed}"
        return 0
    fi

    common=(
        "$PYTHON_BIN" main.py
        --dataset "$dataset" --datadir "$JOB_DATADIR"
        --in_channels 3 --num_classes "$JOB_NUM_CLASSES"
        "${part_flags[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --round "$JOB_ROUNDS" --epochs "$LOCAL_EPOCHS"
        --optimizer sgd --lr "$JOB_LR" --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$JOB_NUM_WORKERS" --seed "$seed" --device "cuda:${gpu}"
        --sequential_client_execution
    )

    byot_common=(
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

    case "$method" in
        plain)
            cmd=("${common[@]}" --model resnet18 --alg fedavg)
            ;;
        fixed)
            cmd=(
                "${common[@]}" "${byot_common[@]}"
                --byot_alpha "$FIXED_LAMBDA"
                --byot_round_lambda_schedule none
                --byot_client_proxy none
                --byot_client_skew_proxy none
                --byot_branch_need_proxy none
            )
            ;;
        adaptive)
            cmd=(
                "${common[@]}" "${byot_common[@]}"
                --byot_alpha "$ADAPTIVE_LAMBDA_MAX" --alpha_min_scale 0.0
                --byot_round_lambda_schedule linear
                --byot_round_lambda_min 0.0
                --byot_round_lambda_warmup "$JOB_WARMUP_ROUNDS"
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
                --byot_branch_need_proxy js_client
                --byot_branch_need_gain "$JS_GAIN"
                --byot_branch_need_min_gate 0.0
                --byot_branch_need_temperature "$PROXY_TEMPERATURE"
            )
            ;;
    esac

    cmd+=(--logdir "$LOG_ROOT" --log_file_name "${rel_dir}/${name}")

    echo "[GPU ${gpu}] start: ${dataset} | ${partition} | ${method} | seed=${seed} | R=${JOB_ROUNDS}"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run][GPU %s] ' "$gpu"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    if ! "${cmd[@]}" > "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${dataset} | ${partition} | ${method} | seed=${seed}" >&2
        tail -40 "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file" "$JOB_ROUNDS"; then
        echo "[GPU ${gpu}] incomplete: ${dataset} | ${partition} | ${method} | seed=${seed}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${dataset} | ${partition} | ${method} | seed=${seed}"
}

# Approximate minutes from completed runs. Used only for queue balancing.
job_weight() {
    local job="$1" dataset partition method _seed
    IFS='|' read -r dataset partition method _seed <<< "$job"
    case "${dataset}|${partition}|${method}" in
        cifar100\|iid\|plain) printf '90' ;;
        cifar100\|iid\|fixed) printf '115' ;;
        cifar100\|iid\|adaptive) printf '155' ;;
        cifar100\|beta_0.1\|plain) printf '90' ;;
        cifar100\|beta_0.1\|fixed) printf '50' ;;
        cifar100\|beta_0.1\|adaptive) printf '65' ;;
        cifar10\|iid\|plain) printf '160' ;;
        cifar10\|iid\|fixed) printf '110' ;;
        cifar10\|iid\|adaptive) printf '150' ;;
        cifar10\|beta_0.1\|plain) printf '165' ;;
        cifar10\|beta_0.1\|fixed) printf '110' ;;
        cifar10\|beta_0.1\|adaptive) printf '155' ;;
        tinyimagenet\|iid\|plain) printf '130' ;;
        tinyimagenet\|iid\|fixed) printf '180' ;;
        tinyimagenet\|iid\|adaptive) printf '255' ;;
        tinyimagenet\|beta_0.1\|plain) printf '130' ;;
        tinyimagenet\|beta_0.1\|fixed) printf '205' ;;
        tinyimagenet\|beta_0.1\|adaptive) printf '155' ;;
        *) printf '180' ;;
    esac
}

echo "========== Three-dataset final comparison completion =========="
echo "run_set=${RUN_SET} | GPUs=${GPUS[*]} | jobs=${#JOBS[@]}"
if [[ "$INCLUDE_VALIDATION" == 1 ]]; then
    effective_validation_seeds="${VALIDATION_SEEDS[*]:-none}"
else
    effective_validation_seeds="none"
fi
echo "validation_seeds=${effective_validation_seeds} | include_validation=${INCLUDE_VALIDATION} | include_seed0_fixed=${INCLUDE_SEED0_FIXED}"
echo "partitions={IID,beta=.1} | methods={Plain,Fixed lambda=.3,Adaptive JS-client}"
echo "CIFAR-10/100: R=500 | TinyImageNet: R=100 | E=5 | participation=.1"
echo "KD-only | T_KD=1 | T_proxy=1 | feature_beta=0 | min_require_size=64"
echo "adaptive: lambda_max=1 | warm-up=50% | tau=.85 | skew_power=2 | JS gain=1"
echo "fixed: lambda=.3 | no warm-up or adaptive proxy"
echo "canonical execution: paired/preserved RNG controls are absent"
echo "log_root=${LOG_ROOT}"

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
    local gpu="${GPUS[$gpu_index]}"
    local failed=0
    local job
    while IFS= read -r job; do
        [[ -n "$job" ]] || continue
        run_job "$gpu" "$job" || failed=1
    done <<< "${QUEUES[$gpu_index]}"
    return "$failed"
}

if [[ "$DRY_RUN" == 1 ]]; then
    status=0
    for ((i=0; i<${#GPUS[@]} && i<${#JOBS[@]}; i++)); do
        run_queue "$i" || status=1
    done
    (( status == 0 )) || exit "$status"
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

echo "Three-dataset final comparison completion finished (${#JOBS[@]} jobs)."
