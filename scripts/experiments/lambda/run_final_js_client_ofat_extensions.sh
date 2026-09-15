#!/usr/bin/env bash

# Final one-factor-at-a-time (OFAT) validation matrix.
#
# Shared reference configuration:
#   CIFAR-100 / ResNet18(-BYOT) / FedAvg / E=5 / participation=0.1
#
# Exactly one axis changes in each scheduled family:
#   dataset       : CIFAR-10, TinyImageNet, ImageNet100-64
#   model         : MobileNetV2(-BYOT), on CIFAR-100
#   mechanism     : FedProx or MOON, on CIFAR-100 + ResNet18
#   local_epochs  : E=1 or E=10, on CIFAR-100 + ResNet18 + FedAvg
#   participation : C=0.05 or C=0.2, on CIFAR-100 + ResNet18 + FedAvg
#
# Every cell compares Plain, fixed lambda=0.3, and the final JS-client
# adaptive method over IID, beta=0.3, and beta=0.1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  GPUS_OVERRIDE="0 1 2 3" \
  AXES_OVERRIDE="dataset model local_epochs participation" \
  bash scripts/experiments/lambda/run_final_js_client_ofat_extensions.sh

Axes:
  dataset model fedprox moon local_epochs participation

Useful overrides:
  SEEDS_OVERRIDE="0"       Default pilot seed
  SKIP_EXISTING=1           Skip only complete files in this exact log root
  DRY_RUN=1                 Print the schedule without training
  USE_WANDB=0               Disabled by default
  IMAGENET100_DATADIR=...   Explicit ImageNet100-64 train/val root
EOF
    exit 0
fi

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3}"
(( ${#GPUS[@]} > 0 )) || { echo "GPUS_OVERRIDE is empty." >&2; exit 1; }
NUM_GPUS=${#GPUS[@]}

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN=venv/bin/python
else
    PYTHON_BIN=python3
fi

read -r -a AXES <<< "${AXES_OVERRIDE:-dataset model fedprox moon local_epochs participation}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-0}"
PARTITIONS=(iid beta_0.3 beta_0.1)
METHODS=(plain fixed_lambda0p30 adaptive_js_client)

NUM_CLIENTS="${NUM_CLIENTS:-100}"
BASE_LOCAL_EPOCHS="${BASE_LOCAL_EPOCHS:-5}"
BASE_PARTICIPATION="${BASE_PARTICIPATION:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"

FIXED_LAMBDA="${FIXED_LAMBDA:-0.3}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
JS_GAIN="${JS_GAIN:-1.0}"

FEDPROX_MU="${FEDPROX_MU:-0.01}"
MOON_MU="${MOON_MU:-0.01}"
MOON_TEMPERATURE="${MOON_TEMPERATURE:-0.5}"

TINYIMAGENET_DATADIR="${TINYIMAGENET_DATADIR:-./data/tiny-imagenet-200}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/final/logs_js_client_ofat_extensions_seed0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
USE_WANDB="${USE_WANDB:-0}"

if [[ "$MIN_REQUIRE_SIZE" != "64" ]]; then
    echo "Final protocol requires MIN_REQUIRE_SIZE=64; got ${MIN_REQUIRE_SIZE}." >&2
    exit 2
fi
if [[ "$KD_TEMPERATURE" != "1.0" || "$PROXY_TEMPERATURE" != "1.0" ]]; then
    echo "Final protocol requires KD_TEMPERATURE=1.0 and PROXY_TEMPERATURE=1.0." >&2
    exit 2
fi
if [[ "$FIXED_LAMBDA" != "0.3" || "$LAMBDA_MAX" != "1.0" ]]; then
    echo "Final protocol requires fixed lambda=0.3 and adaptive lambda max=1.0." >&2
    exit 2
fi

resolve_imagenet100_datadir() {
    local candidate
    if [[ -n "${IMAGENET100_DATADIR:-}" ]]; then
        printf '%s' "$IMAGENET100_DATADIR"
        return
    fi
    for candidate in \
        "./data/imagenet100_resized_64_png" \
        "$HOME/data/imagenet100_resized_64_png" \
        "/data/imagenet100_resized_64_png"; do
        if [[ -d "$candidate/train" && -d "$candidate/val" ]]; then
            printf '%s' "$candidate"
            return
        fi
    done
    printf '%s' "./data/imagenet100_resized_64_png"
}

IMAGENET100_DATADIR="$(resolve_imagenet100_datadir)"

dataset_config() {
    local dataset="$1"
    case "$dataset" in
        cifar10)
            DATA_DIR="./data"; NUM_CLASSES=10; ROUNDS=500; BASE_LR=0.1; NUM_WORKERS=0
            ;;
        cifar100)
            DATA_DIR="./data"; NUM_CLASSES=100; ROUNDS=500; BASE_LR=0.1; NUM_WORKERS=0
            ;;
        tinyimagenet)
            DATA_DIR="$TINYIMAGENET_DATADIR"; NUM_CLASSES=200; ROUNDS=100; BASE_LR=0.01; NUM_WORKERS=2
            ;;
        imagenet100_64)
            DATA_DIR="$IMAGENET100_DATADIR"; NUM_CLASSES=100; ROUNDS=100; BASE_LR=0.01; NUM_WORKERS=2
            ;;
        *) echo "Unknown dataset: ${dataset}" >&2; return 1 ;;
    esac
    WARMUP_ROUNDS=$((ROUNDS / 2))
}

validate_dataset() {
    local dataset="$1"
    dataset_config "$dataset"
    case "$dataset" in
        tinyimagenet|imagenet100_64)
            if [[ ! -d "$DATA_DIR/train" || ! -d "$DATA_DIR/val" ]]; then
                echo "Missing ${dataset} directories: ${DATA_DIR}/{train,val}" >&2
                return 1
            fi
            ;;
    esac
}

partition_flags() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

configure_job() {
    local axis="$1" value="$2"
    DATASET=cifar100
    BYOT_MODEL=resnet18_byot
    PLAIN_MODEL=resnet18
    LOCAL_EPOCHS="$BASE_LOCAL_EPOCHS"
    PARTICIPATION="$BASE_PARTICIPATION"
    MECHANISM=fedavg

    case "$axis" in
        dataset) DATASET="$value" ;;
        model)
            [[ "$value" == "mobilenetv2" ]] || { echo "Unknown model value: $value" >&2; return 1; }
            BYOT_MODEL=mobilenet_byot
            PLAIN_MODEL=mobilenet
            ;;
        fedprox) MECHANISM=fedprox ;;
        moon) MECHANISM=moon ;;
        local_epochs) LOCAL_EPOCHS="$value" ;;
        participation) PARTICIPATION="$value" ;;
        *) echo "Unknown axis: $axis" >&2; return 1 ;;
    esac
    dataset_config "$DATASET"
}

method_tag() {
    case "$1" in
        plain) printf '%s' plain ;;
        fixed_lambda0p30) printf '%s' fixed_lambda0p30_tkd1p00_nofeat ;;
        adaptive_js_client) printf '%s' js_client_lmax1p00_tau0p85_tkd1p00_nofeat ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

pkl_complete() {
    local path="$1" expected_rounds="$2"
    [[ -s "$path" ]] || return 1
    "$PYTHON_BIN" -c '
import pickle, sys
try:
    with open(sys.argv[1], "rb") as stream:
        payload = pickle.load(stream)
    expected = int(sys.argv[2])
    candidates = (
        payload.get("acc_global", []),
        payload.get("branch_acc", []),
        payload.get("test_loss", []),
    )
    ok = any(isinstance(values, (list, tuple)) and len(values) >= expected
             for values in candidates)
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
' "$path" "$expected_rounds"
}

append_mechanism_flags() {
    local method="$1"
    case "$MECHANISM" in
        fedavg)
            if [[ "$method" == plain ]]; then CMD+=(--alg fedavg); else CMD+=(--alg fedbyot); fi
            ;;
        fedprox)
            if [[ "$method" == plain ]]; then
                CMD+=(--alg fedprox --mu "$FEDPROX_MU")
            else
                CMD+=(--alg fedbyot --use_fedprox --mu "$FEDPROX_MU")
            fi
            ;;
        moon)
            if [[ "$method" == plain ]]; then
                CMD+=(--alg moon --mu "$MOON_MU" --temperature "$MOON_TEMPERATURE")
            else
                CMD+=(--alg fedbyot --use_moon --mu "$MOON_MU" --temperature "$MOON_TEMPERATURE")
            fi
            ;;
        *) echo "Unknown mechanism: ${MECHANISM}" >&2; return 1 ;;
    esac
}

append_byot_flags() {
    CMD+=(
        --model "$BYOT_MODEL"
        --byot_active_branches 1,2,3
        --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only
        --byot_beta 0.0
        --byot_teacher_source local
        --alpha_min_scale 0.0
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
    )
    if [[ "$MECHANISM" != moon ]]; then
        CMD+=(--temperature "$KD_TEMPERATURE")
    fi
}

append_adaptive_flags() {
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
}

run_job() {
    local gpu="$1" axis="$2" value="$3" partition="$4" method="$5" seed="$6"
    local tag name rel_dir pkl_file terminal
    local -a CMD PARTITION_FLAGS

    configure_job "$axis" "$value"
    mapfile -t PARTITION_FLAGS < <(partition_flags "$partition")
    tag="$(method_tag "$method")"
    name="${DATASET}_${axis}_${value}_${partition}_${tag}_canonical_seed${seed}_r${ROUNDS}"
    rel_dir="${axis}/${value}/${DATASET}/${partition}/seed${seed}/${method}"
    pkl_file="${LOG_ROOT}/${rel_dir}/${name}.pkl"
    terminal="${LOG_ROOT}/${rel_dir}/${name}_terminal.log"

    if [[ "$SKIP_EXISTING" == 1 ]] && pkl_complete "$pkl_file" "$ROUNDS"; then
        echo "[GPU ${gpu}] skip: ${axis}=${value} | ${partition} | ${method} | seed=${seed}"
        return 0
    fi
    mkdir -p "${LOG_ROOT}/${rel_dir}"

    CMD=(
        "$PYTHON_BIN" main.py
        --dataset "$DATASET" --datadir "$DATA_DIR"
        --in_channels 3 --num_classes "$NUM_CLASSES"
        "${PARTITION_FLAGS[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$PARTICIPATION"
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS"
        --optimizer sgd --lr "$BASE_LR" --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --seed "$seed"
        --device "cuda:${gpu}" --sequential_client_execution
        --logdir "$LOG_ROOT" --log_file_name "${rel_dir}/${name}"
    )

    if [[ "$method" == plain ]]; then
        CMD+=(--model "$PLAIN_MODEL")
        append_mechanism_flags "$method"
    else
        append_byot_flags
        append_mechanism_flags "$method"
        if [[ "$method" == fixed_lambda0p30 ]]; then
            CMD+=(--byot_alpha "$FIXED_LAMBDA")
        else
            append_adaptive_flags
        fi
    fi

    if [[ "$USE_WANDB" == 1 ]]; then
        CMD+=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
        [[ -n "${WANDB_ENTITY:-}" ]] && CMD+=(--wandb_entity "$WANDB_ENTITY")
    fi

    echo "[GPU ${gpu}] start: ${axis}=${value} | ${DATASET} | ${partition} | ${method} | seed=${seed} | R=${ROUNDS}, E=${LOCAL_EPOCHS}, C=${PARTICIPATION}"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run] '
        printf '%q ' "${CMD[@]}"
        printf '\n'
        return 0
    fi

    if ! "${CMD[@]}" > "$terminal" 2>&1; then
        echo "[GPU ${gpu}] failed: ${axis}=${value} | ${partition} | ${method} | seed=${seed}" >&2
        tail -40 "$terminal" >&2 || true
        return 1
    fi
    if ! pkl_complete "$pkl_file" "$ROUNDS"; then
        echo "[GPU ${gpu}] incomplete output: ${pkl_file}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${axis}=${value} | ${partition} | ${method} | seed=${seed}"
}

declare -a JOBS
add_family() {
    local axis="$1" value="$2" partition method seed
    for seed in "${SEEDS[@]}"; do
        for partition in "${PARTITIONS[@]}"; do
            for method in "${METHODS[@]}"; do
                JOBS+=("${axis}|${value}|${partition}|${method}|${seed}")
            done
        done
    done
}

for axis in "${AXES[@]}"; do
    case "$axis" in
        dataset)
            add_family dataset imagenet100_64
            add_family dataset tinyimagenet
            add_family dataset cifar10
            ;;
        model) add_family model mobilenetv2 ;;
        fedprox) add_family fedprox default ;;
        moon) add_family moon default ;;
        local_epochs)
            add_family local_epochs 10
            add_family local_epochs 1
            ;;
        participation)
            add_family participation 0.2
            add_family participation 0.05
            ;;
        *) echo "Unknown AXES_OVERRIDE entry: ${axis}" >&2; exit 2 ;;
    esac
done

# Validate only datasets that the selected axes need.
for job in "${JOBS[@]}"; do
    IFS='|' read -r axis value _ <<< "$job"
    configure_job "$axis" "$value"
    validate_dataset "$DATASET"
done

job_weight() {
    local job="$1" axis value partition method seed weight
    IFS='|' read -r axis value partition method seed <<< "$job"
    configure_job "$axis" "$value"
    weight=100
    case "$DATASET" in
        cifar10) weight=90 ;;
        tinyimagenet) weight=150 ;;
        imagenet100_64) weight=190 ;;
    esac
    [[ "$axis" == local_epochs && "$value" == 10 ]] && weight=$((weight * 2))
    [[ "$axis" == local_epochs && "$value" == 1 ]] && weight=$((weight / 3))
    [[ "$axis" == participation && "$value" == 0.2 ]] && weight=$((weight * 2))
    [[ "$axis" == participation && "$value" == 0.05 ]] && weight=$((weight / 2))
    [[ "$axis" == model ]] && weight=$((weight * 4 / 5))
    [[ "$axis" == fedprox ]] && weight=$((weight * 11 / 10))
    [[ "$axis" == moon ]] && weight=$((weight * 13 / 10))
    [[ "$method" == fixed_lambda0p30 ]] && weight=$((weight * 6 / 5))
    [[ "$method" == adaptive_js_client ]] && weight=$((weight * 3 / 2))
    [[ "$partition" == beta_0.1 ]] && weight=$((weight * 3 / 5))
    printf '%d' "$weight"
}

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

run_queue() {
    local gpu_index="$1" gpu="${GPUS[$1]}" failed=0 job axis value partition method seed
    while IFS= read -r job; do
        [[ -n "$job" ]] || continue
        IFS='|' read -r axis value partition method seed <<< "$job"
        run_job "$gpu" "$axis" "$value" "$partition" "$method" "$seed" || failed=1
    done <<< "${QUEUES[$gpu_index]}"
    return "$failed"
}

echo "========== Final JS-client OFAT extensions =========="
echo "GPUs=${GPUS[*]} | axes=${AXES[*]} | seeds=${SEEDS[*]} | jobs=${#JOBS[@]}"
echo "partitions=IID,beta0.3,beta0.1 | methods=Plain,Fixed0.3,Adaptive-JS-client"
echo "reference=CIFAR100/ResNet18/FedAvg/E5/C0.1"
echo "adaptive=no-feature,T_KD=1,T_proxy=1,lmax=1,tau=.85,warmup=.5R,JS-client"
echo "canonical RNG: paired/preserved RNG flags are absent"
echo "log_root=${LOG_ROOT} | skip_existing=${SKIP_EXISTING}"
for ((i=0; i<NUM_GPUS; i++)); do
    count="$(printf '%s' "${QUEUES[$i]}" | sed '/^$/d' | wc -l)"
    echo "GPU ${GPUS[$i]}: ${count} jobs | relative load=${LOADS[$i]}"
done

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
    [[ -n "${QUEUES[$i]}" ]] || continue
    run_queue "$i" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if [[ "$status" != 0 ]]; then
    echo "At least one job failed. Completed runs remain reusable in ${LOG_ROOT}." >&2
    exit "$status"
fi
echo "Final JS-client OFAT extension queue complete (${#JOBS[@]} jobs)."

