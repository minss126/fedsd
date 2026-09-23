#!/usr/bin/env bash

# TinyImageNet 300-round MobileNetV2 check under the finalized publication
# protocol.
#
# The canonical TinyImageNet run uses 100 communication rounds.  This launcher
# repeats the same seed-0 comparison at 300 rounds to check whether the unusual
# MobileNetV2 result was caused by insufficient convergence time.
#
# Matrix (default):
#   dataset   : TinyImageNet
#   partition : IID, beta=.1
#   method    : Plain, fixed lambda=.3, final JS-client Adaptive
#   seed      : 0
#   FL        : FedAvg
#
# The comparison is configuration matched across methods.  Adaptive alone has
# a 50% linear warm-up (100/200 rounds); fixed lambda has no warm-up or proxy.

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
TINYIMAGENET_DATADIR="${TINYIMAGENET_DATADIR:-${DATA_ROOT}/tiny-imagenet-200}"
IMAGENET100_DATADIR="${IMAGENET100_DATADIR:-}"
if [[ -z "$IMAGENET100_DATADIR" ]]; then
    for candidate in \
        "${DATA_ROOT}/imagenet100_resized_64_png" \
        "$HOME/data/imagenet100_resized_64_png" \
        "/data/imagenet100_resized_64_png"; do
        if [[ -d "$candidate/train" && -d "$candidate/val" ]]; then
            IMAGENET100_DATADIR="$candidate"
            break
        fi
    done
fi
IMAGENET100_DATADIR="${IMAGENET100_DATADIR:-${DATA_ROOT}/imagenet100_resized_64_png}"

read -r -a DATASETS <<< "${DATASETS_OVERRIDE:-tinyimagenet}"
read -r -a PARTITIONS <<< "${PARTITIONS_OVERRIDE:-iid beta_0.1}"
read -r -a METHODS <<< "${METHODS_OVERRIDE:-plain fixed adaptive}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-0}"

ROUNDS="${ROUNDS_OVERRIDE:-300}"
WARMUP_ROUNDS=$((ROUNDS / 2))
LOCAL_EPOCHS=5
NUM_CLIENTS=100
SAMPLE_FRACTION=0.1
MIN_REQUIRE_SIZE=64
BATCH_SIZE=64
TEST_BATCH_SIZE=512

FIXED_LAMBDA=0.3
LAMBDA_MAX=1.0
KD_TEMPERATURE=1.0
PROXY_TEMPERATURE=1.0
SKEW_POWER=2.0
SOFT_TAU=0.85
SOFT_TEMPERATURE=0.05
JS_GAIN=1.0

LOG_ROOT="${LOG_ROOT:-logs/lambda/final/logs_mobilenetv2_tiny_r300_canonical_seed0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
USE_WANDB="${USE_WANDB:-0}"

[[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ROUNDS_OVERRIDE must be a positive integer; got $ROUNDS." >&2
    exit 2
}
for seed in "${SEEDS[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
done

configure_dataset() {
    case "$1" in
        tinyimagenet)
            DATA_DIR="$TINYIMAGENET_DATADIR"; NUM_CLASSES=200; LR=0.01; NUM_WORKERS=2 ;;
        imagenet100_64)
            DATA_DIR="$IMAGENET100_DATADIR"; NUM_CLASSES=100; LR=0.01; NUM_WORKERS=2 ;;
        *)
            echo "This long-horizon launcher supports only tinyimagenet and imagenet100_64; got $1." >&2
            return 1 ;;
    esac
}

validate_dataset() {
    configure_dataset "$1"
    [[ -d "$DATA_DIR/train" && -d "$DATA_DIR/val" ]] || {
        echo "Missing $1 directories: $DATA_DIR/{train,val}" >&2
        return 1
    }
}

partition_flags() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

method_tag() {
    case "$1" in
        plain) printf '%s' plain ;;
        fixed) printf '%s' fixed_lambda0p30_tkd1p00_nofeat ;;
        adaptive) printf '%s' js_client_lmax1p00_tau0p85_tkd1p00_nofeat ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

pkl_complete() {
    local path="$1" expected="$2"
    [[ -s "$path" ]] || return 1
    "$PYTHON_BIN" -c '
import pickle, sys
try:
    with open(sys.argv[1], "rb") as stream:
        payload = pickle.load(stream)
    expected = int(sys.argv[2])
    ok = any(isinstance(payload.get(key), (list, tuple)) and
             len(payload[key]) >= expected
             for key in ("acc_global", "branch_acc", "test_loss"))
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
' "$path" "$expected"
}

job_paths() {
    local dataset="$1" partition="$2" method="$3" seed="$4" tag
    tag="$(method_tag "$method")"
    JOB_NAME="${dataset}_mobilenetv2_fedavg_${partition}_${tag}_canonical_seed${seed}_r${ROUNDS}"
    JOB_REL="mobilenetv2/${dataset}/fedavg/${partition}/seed${seed}/${method}"
    JOB_PKL="${LOG_ROOT}/${JOB_REL}/${JOB_NAME}.pkl"
    JOB_TERMINAL="${LOG_ROOT}/${JOB_REL}/${JOB_NAME}_terminal.log"
}

run_job() {
    local gpu="$1" job="$2"
    local dataset partition method seed
    local -a PART CMD
    IFS='|' read -r dataset partition method seed <<< "$job"
    configure_dataset "$dataset"
    mapfile -t PART < <(partition_flags "$partition")
    job_paths "$dataset" "$partition" "$method" "$seed"

    if [[ "$SKIP_EXISTING" == 1 ]] && pkl_complete "$JOB_PKL" "$ROUNDS"; then
        echo "[GPU $gpu] skip complete: $job"
        return 0
    fi
    mkdir -p "$(dirname "$JOB_PKL")"

    CMD=(
        "$PYTHON_BIN" main.py
        --dataset "$dataset" --datadir "$DATA_DIR"
        --in_channels 3 --num_classes "$NUM_CLASSES"
        "${PART[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS"
        --optimizer sgd --lr "$LR" --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --seed "$seed"
        --device "cuda:${gpu}" --sequential_client_execution
        --logdir "$LOG_ROOT" --log_file_name "${JOB_REL}/${JOB_NAME}"
    )

    if [[ "$method" == plain ]]; then
        CMD+=(--model mobilenet --alg fedavg)
    else
        CMD+=(
            --model mobilenet_byot --alg fedbyot
            --byot_active_branches 1,2,3
            --byot_branch_loss_reduction sum
            --byot_branch_objective kd_only
            --byot_beta 0.0
            --byot_teacher_source local
            --alpha_min_scale 0.0
            --temperature "$KD_TEMPERATURE"
            --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_loss_scale_mode native_t2
            --byot_proxy_temperature "$PROXY_TEMPERATURE"
        )
        if [[ "$method" == fixed ]]; then
            CMD+=(
                --byot_alpha "$FIXED_LAMBDA"
                --byot_round_lambda_schedule none
                --byot_client_proxy none
                --byot_client_skew_proxy none
                --byot_branch_need_proxy none
            )
        else
            CMD+=(
                --byot_alpha "$LAMBDA_MAX"
                --byot_round_lambda_schedule linear
                --byot_round_lambda_min 0.0
                --byot_round_lambda_warmup "$WARMUP_ROUNDS"
                --byot_client_proxy teacher_label_prob
                --byot_client_alpha_min 0.0
                --byot_client_alpha_max 1.0
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
        fi
    fi

    if [[ "$USE_WANDB" == 1 ]]; then
        CMD+=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
        [[ -n "${WANDB_ENTITY:-}" ]] && CMD+=(--wandb_entity "$WANDB_ENTITY")
    fi

    echo "[GPU $gpu] start: $job | R=$ROUNDS warm=$([[ "$method" == adaptive ]] && echo "$WARMUP_ROUNDS" || echo 0)"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run] '; printf '%q ' "${CMD[@]}"; printf '\n'
        return 0
    fi
    if ! "${CMD[@]}" > "$JOB_TERMINAL" 2>&1; then
        echo "[GPU $gpu] failed: $job; see $JOB_TERMINAL" >&2
        tail -40 "$JOB_TERMINAL" >&2 || true
        return 1
    fi
    pkl_complete "$JOB_PKL" "$ROUNDS" || {
        echo "Incomplete output: $JOB_PKL" >&2
        return 1
    }
    echo "[GPU $gpu] complete: $job"
}

# Estimated GPU seconds at 200 rounds, based on retained MobileNet logs, then
# scaled to the requested round count.  The values affect queue balancing only
# and never alter an experimental setting.
job_weight() {
    local dataset partition method seed base
    IFS='|' read -r dataset partition method seed <<< "$1"
    case "$dataset:$method" in
        tinyimagenet:plain) base=5600 ;;
        tinyimagenet:fixed) base=10000 ;;
        tinyimagenet:adaptive) base=10500 ;;
        imagenet100_64:plain) base=13000 ;;
        imagenet100_64:fixed) base=20000 ;;
        imagenet100_64:adaptive) base=12000 ;;
        *) return 1 ;;
    esac
    echo $((base * ROUNDS / 200))
}

declare -a ALL_JOBS=() PENDING=() SORTED=() QUEUES=() LOADS=() PIDS=()
for dataset in "${DATASETS[@]}"; do
    [[ "$DRY_RUN" == 1 ]] || validate_dataset "$dataset"
    configure_dataset "$dataset"
    for partition in "${PARTITIONS[@]}"; do
        partition_flags "$partition" >/dev/null
        for method in "${METHODS[@]}"; do
            method_tag "$method" >/dev/null
            for seed in "${SEEDS[@]}"; do
                ALL_JOBS+=("$dataset|$partition|$method|$seed")
            done
        done
    done
done

for job in "${ALL_JOBS[@]}"; do
    IFS='|' read -r dataset partition method seed <<< "$job"
    job_paths "$dataset" "$partition" "$method" "$seed"
    if [[ "$SKIP_EXISTING" == 1 ]] && pkl_complete "$JOB_PKL" "$ROUNDS"; then
        echo "[skip complete] $job"
    else
        PENDING+=("$job")
    fi
done

if (( ${#PENDING[@]} > 0 )); then
    mapfile -t SORTED < <(
        for job in "${PENDING[@]}"; do
            printf '%s\t%s\n' "$(job_weight "$job")" "$job"
        done | sort -rn | cut -f2-
    )
fi

for ((i=0; i<NUM_GPUS; i++)); do QUEUES[$i]=''; LOADS[$i]=0; done
for job in "${SORTED[@]}"; do
    target=0
    for ((i=1; i<NUM_GPUS; i++)); do
        (( LOADS[i] < LOADS[target] )) && target=$i
    done
    weight="$(job_weight "$job")"
    QUEUES[$target]+="$job"$'\n'
    LOADS[$target]=$((LOADS[target] + weight))
done

echo "========== MobileNetV2 TinyImageNet R300 validation =========="
echo "GPUs=${GPUS[*]} | jobs=${#ALL_JOBS[@]} | pending=${#PENDING[@]} | seeds=${SEEDS[*]}"
echo "datasets=${DATASETS[*]} | partitions=${PARTITIONS[*]} | methods=${METHODS[*]}"
echo "FedAvg | R=$ROUNDS | E=5 | C=.1 | batch=64 | test_batch=512"
echo "fixed=lambda .3, no warm-up | adaptive=JS-client, warm-up=$WARMUP_ROUNDS"
echo "KD-only | feature=0 | T_KD=1 | T_proxy=1 | canonical RNG"
echo "log_root=$LOG_ROOT | skip_existing=$SKIP_EXISTING | dry_run=$DRY_RUN"
for ((i=0; i<NUM_GPUS; i++)); do
    count="$(printf '%s' "${QUEUES[$i]}" | sed '/^$/d' | wc -l)"
    echo "GPU ${GPUS[$i]}: jobs=$count estimated_hours=$(awk -v s="${LOADS[$i]}" 'BEGIN {printf "%.1f", s/3600}')"
done

(( ${#PENDING[@]} > 0 )) || { echo "No incomplete jobs remain."; exit 0; }

worker() {
    local slot="$1" job failed=0
    while IFS= read -r job; do
        [[ -n "$job" ]] || continue
        run_job "${GPUS[$slot]}" "$job" || failed=1
    done <<< "${QUEUES[$slot]}"
    return "$failed"
}

for ((i=0; i<NUM_GPUS; i++)); do
    [[ -n "${QUEUES[$i]}" ]] || continue
    worker "$i" & PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "At least one MobileNetV2 job failed." >&2; exit 1; }
echo "MobileNetV2 long-horizon validation complete."
