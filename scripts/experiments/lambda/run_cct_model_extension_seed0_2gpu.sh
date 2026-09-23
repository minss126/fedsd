#!/usr/bin/env bash

# CCT-7 model-extension experiment on a two-GPU server (e.g. 5070-3).
#
# Full publication matrix:
#   dataset   : CIFAR-100, TinyImageNet, ImageNet100-64
#   partition : IID, beta=.1
#   method    : Plain, fixed lambda=.3, final JS-client Adaptive
#   seed / FL : seed 0 / FedAvg
#
# This matches the finalized ResNet50/MobileNetV2 model-extension protocol:
# CIFAR-100 uses 500 rounds and lr=.1; the two 64x64 datasets use 100 rounds
# and lr=.01.  All full runs use E=5, C=.1, batch=64, T_KD=T_proxy=1,
# KD-only branch supervision, and no feature-imitation loss.  Fixed lambda
# has no warm-up.  Adaptive uses a 50% linear warm-up and JS-client gating.
#
# CCT is a new execution path, so a small three-cell smoke matrix is run
# first.  It covers the plain CCT, fixed CCT-BYOT and adaptive CCT-BYOT paths,
# both 32x32 and 64x64 inputs, and all three class counts.  The full matrix is
# launched only if every smoke cell completes and writes a valid PKL.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1}"
(( ${#GPUS[@]} == 2 )) || {
    echo "Exactly two GPU ids are required; got: ${GPUS[*]:-<empty>}" >&2
    exit 2
}

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

LOG_ROOT="${LOG_ROOT:-logs/lambda/final/logs_cct_model_extension_seed0}"
SMOKE_LOG_ROOT="${SMOKE_LOG_ROOT:-logs/lambda/smoke/logs_cct_model_extension_seed0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_SMOKE_FIRST="${RUN_SMOKE_FIRST:-1}"
REUSE_SMOKE="${REUSE_SMOKE:-1}"
RUN_FULL_AFTER_SMOKE="${RUN_FULL_AFTER_SMOKE:-1}"
SMOKE_ROUNDS="${SMOKE_ROUNDS:-2}"
SMOKE_LOCAL_EPOCHS="${SMOKE_LOCAL_EPOCHS:-1}"
DRY_RUN="${DRY_RUN:-0}"
USE_WANDB="${USE_WANDB:-0}"

NUM_CLIENTS=100
SAMPLE_FRACTION=0.1
LOCAL_EPOCHS=5
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

read -r -a DATASETS <<< "${DATASETS_OVERRIDE:-cifar100 tinyimagenet imagenet100_64}"
read -r -a PARTITIONS <<< "${PARTITIONS_OVERRIDE:-iid beta_0.1}"
read -r -a METHODS <<< "${METHODS_OVERRIDE:-plain fixed adaptive}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-0}"

for value_name in SKIP_EXISTING RUN_SMOKE_FIRST REUSE_SMOKE RUN_FULL_AFTER_SMOKE DRY_RUN USE_WANDB; do
    value="${!value_name}"
    [[ "$value" == 0 || "$value" == 1 ]] || {
        echo "$value_name must be 0 or 1; got $value." >&2
        exit 2
    }
done
[[ "$SMOKE_ROUNDS" =~ ^[1-9][0-9]*$ ]] || {
    echo "SMOKE_ROUNDS must be a positive integer." >&2; exit 2;
}
[[ "$SMOKE_LOCAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
    echo "SMOKE_LOCAL_EPOCHS must be a positive integer." >&2; exit 2;
}
for seed in "${SEEDS[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
done

configure_dataset() {
    case "$1" in
        cifar100)
            DATA_DIR="$DATA_ROOT"; NUM_CLASSES=100; ROUNDS=500; LR=0.1; NUM_WORKERS=0 ;;
        tinyimagenet)
            DATA_DIR="$TINYIMAGENET_DATADIR"; NUM_CLASSES=200; ROUNDS=100; LR=0.01; NUM_WORKERS=2 ;;
        imagenet100_64)
            DATA_DIR="$IMAGENET100_DATADIR"; NUM_CLASSES=100; ROUNDS=100; LR=0.01; NUM_WORKERS=2 ;;
        *)
            echo "Unknown dataset: $1" >&2; return 1 ;;
    esac
    if [[ -n "${ROUNDS_OVERRIDE:-}" ]]; then
        ROUNDS="$ROUNDS_OVERRIDE"
    fi
    [[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || {
        echo "ROUNDS_OVERRIDE must be a positive integer." >&2; return 1;
    }
    WARMUP_ROUNDS=$((ROUNDS / 2))
}

validate_dataset() {
    configure_dataset "$1"
    case "$1" in
        cifar100)
            [[ -d "$DATA_DIR/cifar-100-python" ]] || {
                echo "Missing CIFAR-100: $DATA_DIR/cifar-100-python" >&2; return 1;
            } ;;
        tinyimagenet|imagenet100_64)
            [[ -d "$DATA_DIR/train" && -d "$DATA_DIR/val" ]] || {
                echo "Missing $1: $DATA_DIR/{train,val}" >&2; return 1;
            } ;;
    esac
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
    ok = any(
        isinstance(payload.get(key), (list, tuple)) and len(payload[key]) >= expected
        for key in ("acc_global", "branch_acc", "test_loss")
    )
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
' "$path" "$expected"
}

append_byot_args() {
    local method="$1" warmup="$2"
    CMD+=(
        --model cct_byot --alg fedbyot
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
    elif [[ "$method" == adaptive ]]; then
        CMD+=(
            --byot_alpha "$LAMBDA_MAX"
            --byot_round_lambda_schedule linear
            --byot_round_lambda_min 0.0
            --byot_round_lambda_warmup "$warmup"
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
    else
        echo "append_byot_args received unsupported method=$method" >&2
        return 1
    fi
}

job_paths() {
    local root="$1" dataset="$2" partition="$3" method="$4" seed="$5" rounds="$6"
    local tag
    tag="$(method_tag "$method")"
    JOB_NAME="${dataset}_cct_fedavg_${partition}_${tag}_canonical_seed${seed}_r${rounds}"
    JOB_REL="cct/${dataset}/fedavg/${partition}/seed${seed}/${method}"
    JOB_PKL="${root}/${JOB_REL}/${JOB_NAME}.pkl"
    JOB_TERMINAL="${root}/${JOB_REL}/${JOB_NAME}_terminal.log"
}

run_cell() {
    local gpu="$1" root="$2" dataset="$3" partition="$4" method="$5" seed="$6"
    local rounds="$7" epochs="$8" warmup="$9" label="${10}"
    local applied_warmup=0
    local -a PART CMD

    configure_dataset "$dataset"
    mapfile -t PART < <(partition_flags "$partition")
    job_paths "$root" "$dataset" "$partition" "$method" "$seed" "$rounds"

    if [[ "$SKIP_EXISTING" == 1 ]] && pkl_complete "$JOB_PKL" "$rounds"; then
        echo "[GPU $gpu] skip complete ($label): $dataset|$partition|$method|seed$seed"
        return 0
    fi
    mkdir -p "$(dirname "$JOB_PKL")"

    CMD=(
        "$PYTHON_BIN" main.py
        --dataset "$dataset" --datadir "$DATA_DIR"
        --in_channels 3 --num_classes "$NUM_CLASSES"
        "${PART[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --round "$rounds" --epochs "$epochs"
        --optimizer sgd --lr "$LR" --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --seed "$seed"
        --device "cuda:${gpu}"
        --sequential_client_execution --transient_client_models
        --logdir "$root" --log_file_name "${JOB_REL}/${JOB_NAME}"
    )

    if [[ "$method" == plain ]]; then
        CMD+=(--model cct --alg fedavg)
    else
        append_byot_args "$method" "$warmup"
        [[ "$method" == adaptive ]] && applied_warmup="$warmup"
    fi

    if [[ "$USE_WANDB" == 1 ]]; then
        CMD+=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
        [[ -n "${WANDB_ENTITY:-}" ]] && CMD+=(--wandb_entity "$WANDB_ENTITY")
    fi

    echo "[GPU $gpu] start ($label): $dataset|$partition|$method|seed$seed | R=$rounds E=$epochs warm=$applied_warmup"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run] '; printf '%q ' "${CMD[@]}"; printf '\n'
        return 0
    fi
    if ! "${CMD[@]}" > "$JOB_TERMINAL" 2>&1; then
        echo "[GPU $gpu] failed ($label): $dataset|$partition|$method|seed$seed" >&2
        echo "Inspect: $JOB_TERMINAL" >&2
        tail -40 "$JOB_TERMINAL" >&2 || true
        return 1
    fi
    if ! pkl_complete "$JOB_PKL" "$rounds"; then
        echo "[GPU $gpu] incomplete output: $JOB_PKL" >&2
        return 1
    fi
    echo "[GPU $gpu] complete ($label): $dataset|$partition|$method|seed$seed"
}

smoke_complete() {
    local dataset partition method seed=0
    for spec in \
        'cifar100|iid|plain' \
        'tinyimagenet|iid|fixed' \
        'imagenet100_64|iid|adaptive'; do
        IFS='|' read -r dataset partition method <<< "$spec"
        job_paths "$SMOKE_LOG_ROOT" "$dataset" "$partition" "$method" "$seed" "$SMOKE_ROUNDS"
        pkl_complete "$JOB_PKL" "$SMOKE_ROUNDS" || return 1
    done
}

run_smoke() {
    local smoke_warmup=$(( (SMOKE_ROUNDS + 1) / 2 ))
    local failed=0 p0 p1
    echo "========== CCT preflight smoke test =========="
    echo "GPU ${GPUS[0]}: CIFAR-100 plain -> TinyImageNet fixed"
    echo "GPU ${GPUS[1]}: ImageNet100-64 adaptive"
    echo "smoke only: R=$SMOKE_ROUNDS, E=$SMOKE_LOCAL_EPOCHS, batch=$BATCH_SIZE, test_batch=$TEST_BATCH_SIZE"
    (
        run_cell "${GPUS[0]}" "$SMOKE_LOG_ROOT" cifar100 iid plain 0 \
            "$SMOKE_ROUNDS" "$SMOKE_LOCAL_EPOCHS" 0 smoke
        run_cell "${GPUS[0]}" "$SMOKE_LOG_ROOT" tinyimagenet iid fixed 0 \
            "$SMOKE_ROUNDS" "$SMOKE_LOCAL_EPOCHS" 0 smoke
    ) & p0=$!
    run_cell "${GPUS[1]}" "$SMOKE_LOG_ROOT" imagenet100_64 iid adaptive 0 \
        "$SMOKE_ROUNDS" "$SMOKE_LOCAL_EPOCHS" "$smoke_warmup" smoke & p1=$!
    wait "$p0" || failed=1
    wait "$p1" || failed=1
    (( failed == 0 )) || return 1
    [[ "$DRY_RUN" == 1 ]] && return 0
    smoke_complete
}

# Validate model imports and both supported token-grid sizes before occupying a
# GPU.  This is intentionally a forward-only static preflight, not a training
# experiment.
preflight_model() {
    "$PYTHON_BIN" - <<'PY'
import torch
from models.cct import cct_7_3x2_32, cct_7_3x2_32_byot

for image_size, classes in ((32, 100), (64, 200), (64, 100)):
    x = torch.zeros(1, 3, image_size, image_size)
    plain = cct_7_3x2_32(img_size=image_size, num_classes=classes)
    byot = cct_7_3x2_32_byot(img_size=image_size, num_classes=classes)
    plain_output = plain(x)
    byot_output = byot(x)
    assert plain_output.shape == (1, classes)
    assert isinstance(byot_output, tuple) and len(byot_output) == 8
    assert all(value.shape == (1, classes) for value in byot_output[:4])
    assert all(value.shape == (1, 256) for value in byot_output[4:])
print("CCT static preflight passed for 32x32 and 64x64 inputs.")
PY
}

if [[ "$DRY_RUN" != 1 ]]; then
    for dataset in "${DATASETS[@]}"; do validate_dataset "$dataset"; done
    preflight_model
fi

for partition in "${PARTITIONS[@]}"; do partition_flags "$partition" >/dev/null; done
for method in "${METHODS[@]}"; do method_tag "$method" >/dev/null; done

if [[ "$RUN_SMOKE_FIRST" == 1 ]]; then
    if [[ "$REUSE_SMOKE" == 1 ]] && smoke_complete; then
        echo "Reusing the completed CCT smoke test."
    else
        run_smoke || {
            echo "CCT smoke test failed; the full matrix was not started." >&2
            exit 1
        }
        echo "CCT smoke test passed."
    fi
fi

if [[ "$RUN_FULL_AFTER_SMOKE" != 1 ]]; then
    echo "RUN_FULL_AFTER_SMOKE=$RUN_FULL_AFTER_SMOKE; full matrix skipped."
    exit 0
fi

declare -a ALL_JOBS=() PENDING=() SORTED=() QUEUES=() LOADS=() PIDS=()
for dataset in "${DATASETS[@]}"; do
    configure_dataset "$dataset"
    for partition in "${PARTITIONS[@]}"; do
        for method in "${METHODS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                ALL_JOBS+=("$dataset|$partition|$method|$seed")
            done
        done
    done
done

for job in "${ALL_JOBS[@]}"; do
    IFS='|' read -r dataset partition method seed <<< "$job"
    configure_dataset "$dataset"
    job_paths "$LOG_ROOT" "$dataset" "$partition" "$method" "$seed" "$ROUNDS"
    if [[ "$SKIP_EXISTING" == 1 ]] && pkl_complete "$JOB_PKL" "$ROUNDS"; then
        echo "[skip complete] $job"
    else
        PENDING+=("$job")
    fi
done

# Queue weights are scheduling hints only.  They do not modify any experiment
# argument.  Until full CCT timing logs exist, the 64x64 datasets are weighted
# conservatively because their token sequence is four times longer than at
# 32x32 and self-attention is quadratic in sequence length.
job_weight() {
    local dataset partition method seed base
    IFS='|' read -r dataset partition method seed <<< "$1"
    case "$dataset:$method" in
        cifar100:plain) base=500 ;;
        cifar100:fixed) base=700 ;;
        cifar100:adaptive) base=780 ;;
        tinyimagenet:plain) base=900 ;;
        tinyimagenet:fixed) base=1250 ;;
        tinyimagenet:adaptive) base=1400 ;;
        imagenet100_64:plain) base=1100 ;;
        imagenet100_64:fixed) base=1500 ;;
        imagenet100_64:adaptive) base=1680 ;;
        *) echo "No queue weight for $1" >&2; return 1 ;;
    esac
    echo "$base"
}

if (( ${#PENDING[@]} > 0 )); then
    mapfile -t SORTED < <(
        for job in "${PENDING[@]}"; do
            printf '%s\t%s\n' "$(job_weight "$job")" "$job"
        done | sort -rn | cut -f2-
    )
fi

for ((slot=0; slot<${#GPUS[@]}; slot++)); do
    QUEUES[$slot]=''
    LOADS[$slot]=0
done
for job in "${SORTED[@]}"; do
    target=0
    for ((slot=1; slot<${#GPUS[@]}; slot++)); do
        (( LOADS[slot] < LOADS[target] )) && target=$slot
    done
    weight="$(job_weight "$job")"
    QUEUES[$target]+="$job"$'\n'
    LOADS[$target]=$((LOADS[target] + weight))
done

echo "========== CCT model-extension matrix =========="
echo "GPUs=${GPUS[*]} | jobs=${#ALL_JOBS[@]} | pending=${#PENDING[@]} | seeds=${SEEDS[*]}"
echo "datasets=${DATASETS[*]} | partitions=${PARTITIONS[*]} | methods=${METHODS[*]}"
if [[ -n "${ROUNDS_OVERRIDE:-}" ]]; then
    echo "FedAvg | rounds_override=$ROUNDS_OVERRIDE | CIFAR lr=.1 | Tiny+Image lr=.01 | E=5 | C=.1"
else
    echo "FedAvg | CIFAR R=500/lr=.1 | Tiny+Image R=100/lr=.01 | E=5 | C=.1"
fi
echo "batch=$BATCH_SIZE | test_batch=$TEST_BATCH_SIZE | min_client_size=$MIN_REQUIRE_SIZE"
echo "fixed=lambda .3, no warm-up | adaptive=JS-client, 50% warm-up"
echo "KD-only | feature=0 | T_KD=1 | T_proxy=1 | canonical RNG"
echo "transient+sequential client execution (memory-only implementation detail)"
echo "log_root=$LOG_ROOT | skip_existing=$SKIP_EXISTING | dry_run=$DRY_RUN"
for ((slot=0; slot<${#GPUS[@]}; slot++)); do
    count="$(printf '%s' "${QUEUES[$slot]}" | sed '/^$/d' | wc -l)"
    echo "GPU ${GPUS[$slot]}: jobs=$count | relative_queue_weight=${LOADS[$slot]}"
done

(( ${#PENDING[@]} > 0 )) || { echo "No incomplete CCT jobs remain."; exit 0; }

worker() {
    local slot="$1" job failed=0 dataset partition method seed
    while IFS= read -r job; do
        [[ -n "$job" ]] || continue
        IFS='|' read -r dataset partition method seed <<< "$job"
        configure_dataset "$dataset"
        run_cell "${GPUS[$slot]}" "$LOG_ROOT" "$dataset" "$partition" "$method" "$seed" \
            "$ROUNDS" "$LOCAL_EPOCHS" "$WARMUP_ROUNDS" full || failed=1
    done <<< "${QUEUES[$slot]}"
    return "$failed"
}

for ((slot=0; slot<${#GPUS[@]}; slot++)); do
    [[ -n "${QUEUES[$slot]}" ]] || continue
    worker "$slot" & PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || {
    echo "At least one CCT job failed; completed PKLs remain reusable on restart." >&2
    exit 1
}
echo "CCT model-extension matrix complete."
