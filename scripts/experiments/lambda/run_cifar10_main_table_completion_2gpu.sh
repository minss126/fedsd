#!/usr/bin/env bash

# CIFAR-10 completion for the ResNet18 publication table.
#
# Logical matrix:
#   mechanisms : FedAvg, FedProx, MOON
#   partitions : IID, beta=.1
#   methods    : Plain, fixed lambda=.3, final JS-client Adaptive
#   seeds      : 0,1,2
#
# Seven exact cells already retained in the primary repository are reused:
#   FedAvg seed 0, all three methods x two partitions                 (6)
#   FedAvg beta=.1 Adaptive seed 1                                    (1)
# Therefore the default full queue contains 47 new cells.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1}"
(( ${#GPUS[@]} > 0 )) || { echo "GPUS_OVERRIDE is empty." >&2; exit 2; }
NUM_GPUS=${#GPUS[@]}

if [[ -n "${PYTHON_BIN:-}" ]]; then :
elif [[ -x venv/bin/python ]]; then PYTHON_BIN=venv/bin/python
else PYTHON_BIN=python3
fi

DATA_ROOT="${DATA_ROOT:-./data}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-0 1 2}"
ROUNDS=500
WARMUP=250
LOCAL_EPOCHS=5
NUM_CLIENTS=100
SAMPLE_FRACTION=0.1
MIN_REQUIRE_SIZE=64
PARTITION_MAX_ATTEMPTS="${PARTITION_MAX_ATTEMPTS:-10000}"
BATCH_SIZE=64
TEST_BATCH_SIZE=512
NUM_WORKERS=0
LR=0.1

FIXED_LAMBDA=0.3
KD_TEMPERATURE=1.0
PROXY_TEMPERATURE=1.0
LAMBDA_MAX=1.0
SKEW_POWER=2.0
SOFT_TAU=0.85
SOFT_TEMPERATURE=0.05
JS_GAIN=1.0
FEDPROX_MU=0.01
MOON_MU=0.01
MOON_TEMPERATURE=0.5

RUN_SMOKE_FIRST="${RUN_SMOKE_FIRST:-1}"
SMOKE_ROUNDS="${SMOKE_ROUNDS:-2}"
SMOKE_LOG_ROOT="${SMOKE_LOG_ROOT:-logs/lambda/smoke/logs_cifar10_main_table_completion}"
FULL_LOG_ROOT="${FULL_LOG_ROOT:-logs/lambda/final/logs_cifar10_main_table_completion}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REUSE_PRIOR="${REUSE_PRIOR:-1}"
DRY_RUN="${DRY_RUN:-0}"

for seed in "${SEEDS[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
done
[[ "$SMOKE_ROUNDS" =~ ^[1-9][0-9]*$ ]] || { echo "SMOKE_ROUNDS must be positive." >&2; exit 2; }

partition_flags() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) return 1 ;;
    esac
}

pkl_complete() {
    local path="$1" expected="$2"
    [[ -s "$path" ]] || return 1
    "$PYTHON_BIN" -c '
import pickle, sys
try:
    with open(sys.argv[1], "rb") as f: x = pickle.load(f)
    n = int(sys.argv[2])
    ok = any(isinstance(x.get(k), (list, tuple)) and len(x[k]) >= n
             for k in ("acc_global", "branch_acc", "test_loss"))
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
' "$path" "$expected"
}

prior_reusable() {
    local mechanism="$1" partition="$2" method="$3" seed="$4"
    [[ "$REUSE_PRIOR" == 1 && "$mechanism" == fedavg ]] || return 1
    [[ "$seed" == 0 ]] && return 0
    [[ "$seed" == 1 && "$partition" == beta_0.1 && "$method" == adaptive ]]
}

declare -a SMOKE_JOBS=() FULL_JOBS=()
for mechanism in fedavg fedprox moon; do
    for partition in iid beta_0.1; do
        for method in plain fixed adaptive; do
            SMOKE_JOBS+=("$mechanism|$partition|$method|2")
            for seed in "${SEEDS[@]}"; do
                prior_reusable "$mechanism" "$partition" "$method" "$seed" || \
                    FULL_JOBS+=("$mechanism|$partition|$method|$seed")
            done
        done
    done
done

run_job() {
    local gpu="$1" log_root="$2" job="$3" mechanism partition method seed tag name rel pkl terminal rounds warmup
    local -a PART CMD
    IFS='|' read -r mechanism partition method seed <<< "$job"
    mapfile -t PART < <(partition_flags "$partition")
    rounds="${ROUNDS_OVERRIDE:-$ROUNDS}"
    warmup=$((rounds / 2))
    case "$method" in
        plain) tag=plain ;;
        fixed) tag=fixed_lambda0p30_tkd1p00_nofeat ;;
        adaptive) tag=js_client_lmax1p00_tau0p85_tkd1p00_nofeat ;;
        *) return 1 ;;
    esac
    name="cifar10_resnet18_${mechanism}_${partition}_${tag}_canonical_seed${seed}_r${rounds}"
    rel="resnet18/cifar10/${mechanism}/${partition}/seed${seed}/${method}"
    pkl="${log_root}/${rel}/${name}.pkl"
    terminal="${log_root}/${rel}/${name}_terminal.log"
    if [[ "$SKIP_EXISTING" == 1 ]] && pkl_complete "$pkl" "$rounds"; then
        echo "[GPU $gpu] skip: $job"
        return 0
    fi
    mkdir -p "${log_root}/${rel}"

    CMD=(
        "$PYTHON_BIN" main.py
        --dataset cifar10 --datadir "$DATA_ROOT" --in_channels 3 --num_classes 10
        "${PART[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --partition_max_attempts "$PARTITION_MAX_ATTEMPTS"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --round "$rounds" --epochs "$LOCAL_EPOCHS"
        --optimizer sgd --lr "$LR" --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --seed "$seed"
        --device "cuda:${gpu}" --sequential_client_execution
        --logdir "$log_root" --log_file_name "${rel}/${name}"
    )

    if [[ "$method" == plain ]]; then
        CMD+=(--model resnet18)
        case "$mechanism" in
            fedavg) CMD+=(--alg fedavg) ;;
            fedprox) CMD+=(--alg fedprox --mu "$FEDPROX_MU") ;;
            moon) CMD+=(--alg moon --mu "$MOON_MU" --temperature "$MOON_TEMPERATURE") ;;
        esac
    else
        CMD+=(
            --model resnet18_byot --alg fedbyot
            --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
            --byot_branch_objective kd_only --byot_beta 0.0
            --byot_teacher_source local --alpha_min_scale 0.0
            --temperature "$KD_TEMPERATURE"
            --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_loss_scale_mode native_t2
            --byot_proxy_temperature "$PROXY_TEMPERATURE"
        )
        case "$method" in
            fixed)
                CMD+=(
                    --byot_alpha "$FIXED_LAMBDA"
                    --byot_round_lambda_schedule none
                    --byot_client_proxy none --byot_client_skew_proxy none
                    --byot_branch_need_proxy none
                ) ;;
            adaptive)
                CMD+=(
                    --byot_alpha "$LAMBDA_MAX"
                    --byot_round_lambda_schedule linear
                    --byot_round_lambda_min 0.0 --byot_round_lambda_warmup "$warmup"
                    --byot_client_proxy teacher_label_prob
                    --byot_client_alpha_min 0.0 --byot_client_alpha_max 1.0
                    --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
                    --byot_client_skew_proxy prediction_entropy
                    --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.0
                    --byot_client_skew_correction_mode soft_relax
                    --byot_client_skew_soft_tau "$SOFT_TAU"
                    --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
                    --byot_branch_need_proxy js_client --byot_branch_need_gain "$JS_GAIN"
                    --byot_branch_need_min_gate 0.0
                    --byot_branch_need_temperature "$PROXY_TEMPERATURE"
                ) ;;
        esac
        case "$mechanism" in
            fedavg) ;;
            fedprox) CMD+=(--use_fedprox --mu "$FEDPROX_MU") ;;
            moon) CMD+=(--use_moon --mu "$MOON_MU" --temperature "$MOON_TEMPERATURE") ;;
        esac
    fi

    echo "[GPU $gpu] start: $job | R=$rounds warm=$warmup"
    if [[ "$DRY_RUN" == 1 ]]; then printf '[dry-run] '; printf '%q ' "${CMD[@]}"; printf '\n'; return 0; fi
    if ! "${CMD[@]}" > "$terminal" 2>&1; then
        echo "[GPU $gpu] failed: $job; see $terminal" >&2
        tail -40 "$terminal" >&2 || true
        return 1
    fi
    pkl_complete "$pkl" "$rounds" || { echo "Incomplete output: $pkl" >&2; return 1; }
    echo "[GPU $gpu] complete: $job"
}

job_weight() {
    local mechanism partition method seed base factor
    IFS='|' read -r mechanism partition method seed <<< "$1"
    case "$method" in plain) base=4200 ;; fixed) base=5600 ;; adaptive) base=7200 ;; esac
    case "$mechanism" in fedavg) factor=100 ;; fedprox) factor=110 ;; moon) factor=125 ;; esac
    printf '%s\n' $((base * factor / 100))
}

run_matrix() {
    local log_root="$1" source="$2" job mechanism partition method seed tag name rel pkl rounds
    local -a SOURCE_JOBS PENDING SORTED LOADS GPU_QUEUES PIDS
    if [[ "$source" == smoke ]]; then SOURCE_JOBS=("${SMOKE_JOBS[@]}"); else SOURCE_JOBS=("${FULL_JOBS[@]}"); fi
    rounds="${ROUNDS_OVERRIDE:-$ROUNDS}"
    for job in "${SOURCE_JOBS[@]}"; do
        IFS='|' read -r mechanism partition method seed <<< "$job"
        case "$method" in plain) tag=plain ;; fixed) tag=fixed_lambda0p30_tkd1p00_nofeat ;; adaptive) tag=js_client_lmax1p00_tau0p85_tkd1p00_nofeat ;; esac
        name="cifar10_resnet18_${mechanism}_${partition}_${tag}_canonical_seed${seed}_r${rounds}"
        rel="resnet18/cifar10/${mechanism}/${partition}/seed${seed}/${method}"
        pkl="${log_root}/${rel}/${name}.pkl"
        if [[ "$SKIP_EXISTING" == 1 ]] && pkl_complete "$pkl" "$rounds"; then continue; fi
        PENDING+=("$job")
    done
    echo "queue=$source | pending=${#PENDING[@]} | log_root=$log_root"
    (( ${#PENDING[@]} > 0 )) || return 0
    mapfile -t SORTED < <(for job in "${PENDING[@]}"; do printf '%s\t%s\n' "$(job_weight "$job")" "$job"; done | sort -rn | cut -f2-)

    local i target weight status=0
    for ((i=0;i<NUM_GPUS;i++)); do LOADS[$i]=0; GPU_QUEUES[$i]=''; done
    for job in "${SORTED[@]}"; do
        target=0
        for ((i=1;i<NUM_GPUS;i++)); do (( LOADS[i] < LOADS[target] )) && target=$i; done
        GPU_QUEUES[$target]+="$job"$'\n'; weight="$(job_weight "$job")"; LOADS[$target]=$((LOADS[target]+weight))
    done
    for ((i=0;i<NUM_GPUS;i++)); do
        ( while IFS= read -r job; do [[ -z "$job" ]] || run_job "${GPUS[$i]}" "$log_root" "$job"; done <<< "${GPU_QUEUES[$i]}" ) &
        PIDS+=("$!")
        echo "GPU ${GPUS[$i]} estimated_seconds=${LOADS[$i]}"
    done
    for i in "${PIDS[@]}"; do wait "$i" || status=1; done
    (( status == 0 ))
}

if [[ "$DRY_RUN" != 1 ]]; then
    [[ -d "$DATA_ROOT/cifar-10-batches-py" ]] || { echo "Missing CIFAR-10: $DATA_ROOT/cifar-10-batches-py" >&2; exit 2; }
fi

echo "========== CIFAR-10 main-table completion =========="
echo "GPUs=${GPUS[*]} | full_jobs=${#FULL_JOBS[@]} | reused_prior=7"
echo "mechanisms=FedAvg,FedProx,MOON | methods=Plain,Fixed,Adaptive"
echo "partitions=IID,beta=.1 | seeds=${SEEDS[*]} | R=500 | E=5 | C=.1"
echo "fixed lambda=.3; adaptive=JS-client; KD-only; feature=0; T_KD=1; canonical RNG"

if [[ "$RUN_SMOKE_FIRST" == 1 ]]; then
    ROUNDS_OVERRIDE="$SMOKE_ROUNDS" run_matrix "$SMOKE_LOG_ROOT" smoke
    echo "All smoke cells passed; starting full queue."
fi
unset ROUNDS_OVERRIDE || true
run_matrix "$FULL_LOG_ROOT" full
echo "CIFAR-10 main-table completion finished."
