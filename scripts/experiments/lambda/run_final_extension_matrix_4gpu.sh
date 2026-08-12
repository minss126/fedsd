#!/usr/bin/env bash

# Final non-factorial extension matrix for the selected soft-b adaptive lambda.
#
# Newly run cells (57 total, seed 0):
#   Model axis (FedAvg, IID):
#     - MobileNetV2 on TinyImageNet and ImageNet100-64
#     - ResNet-50 on CIFAR-100, TinyImageNet, and ImageNet100-64
#   Mechanism axis (ResNet-18, beta={0.5,0.3}):
#     - FedProx and MOON on TinyImageNet and ImageNet100-64
#     - FedAvgM on CIFAR-100, TinyImageNet, and ImageNet100-64
#
# Each cell compares Plain / fixed lambda=0.3 / selected adaptive lambda.
# Adaptive uses the final, pre-registered configuration:
# T_KD=T_proxy=1, lambda_max=1, soft tau=.85, soft temperature=.05,
# skew power=2, and warm-up=0.5 * total communication rounds.
# Fixed lambda is constant from round 0 and never receives warm-up.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Keeps large ResNet50-BYOT allocations from accumulating unusable reserved
# blocks across train/evaluation transitions.  An explicit user value wins.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  GPUS_OVERRIDE="0 1 2 3" bash scripts/experiments/lambda/run_final_extension_matrix_4gpu.sh

Optional overrides:
  USE_WANDB=0                 Disable Weights & Biases.
  DRY_RUN=1                   Print the scheduled commands without running them.
  SKIP_EXISTING=0             Re-run completed cells.
  SEED=1                      Change the seed.
  BATCH_SIZE=64               Default batch size for ResNet-18/MobileNet.
  RESNET50_BATCH_SIZE=32      Batch size for both ResNet-50 variants.
  RESNET50_TEST_BATCH_SIZE=32 Evaluation batch size for ResNet-50 BYOT.
  FEDAVGM_MOMENTUM=0.9        Server momentum for FedAvgM.
EOF
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if (( ${#GPUS[@]} == 0 )); then
    echo "Set GPUS_OVERRIDE to one or more GPU ids." >&2
    exit 1
fi
NUM_GPUS=${#GPUS[@]}

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

SEED="${SEED:-0}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RESNET50_BATCH_SIZE="${RESNET50_BATCH_SIZE:-32}"
RESNET50_TEST_BATCH_SIZE="${RESNET50_TEST_BATCH_SIZE:-32}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"

CIFAR100_ROUNDS="${CIFAR100_ROUNDS:-500}"
TINYIMAGENET_ROUNDS="${TINYIMAGENET_ROUNDS:-100}"
IMAGENET100_ROUNDS="${IMAGENET100_ROUNDS:-100}"
TINYIMAGENET_DATADIR="${TINYIMAGENET_DATADIR:-./data/tiny-imagenet-200}"
IMAGENET100_DATADIR="${IMAGENET100_DATADIR:-/data/imagenet100_resized_64_png}"

FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
LAMBDA_WARMUP_RATIO="${LAMBDA_WARMUP_RATIO:-0.5}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
# This extension is pre-registered with one common fixed competitor.
FIXED_LAMBDA="0.30"

FEDPROX_MU="${FEDPROX_MU:-0.01}"
MOON_MU="${MOON_MU:-0.01}"
MOON_TEMPERATURE="${MOON_TEMPERATURE:-0.5}"
FEDAVGM_MOMENTUM="${FEDAVGM_MOMENTUM:-0.9}"

LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_final_extension_matrix}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

WANDB_FLAGS=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
fi

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

dataset_args() {
    DATASET="$1"
    case "$DATASET" in
        cifar100)
            DATA_DIR="./data"
            ROUNDS="$CIFAR100_ROUNDS"
            LR="0.1"
            NUM_WORKERS="0"
            ;;
        tinyimagenet)
            DATA_DIR="$TINYIMAGENET_DATADIR"
            ROUNDS="$TINYIMAGENET_ROUNDS"
            LR="0.01"
            NUM_WORKERS="2"
            ;;
        imagenet100_64)
            DATA_DIR="$IMAGENET100_DATADIR"
            ROUNDS="$IMAGENET100_ROUNDS"
            LR="0.01"
            NUM_WORKERS="2"
            ;;
        *)
            echo "Unknown dataset: $DATASET" >&2
            exit 1
            ;;
    esac
    WARMUP_ROUNDS=$(awk -v rounds="$ROUNDS" -v ratio="$LAMBDA_WARMUP_RATIO" \
        'BEGIN { printf "%d", int(rounds * ratio + 0.5) }')
}

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        *)
            echo "Unknown partition: $1" >&2
            exit 1
            ;;
    esac
}

configure_axis() {
    local axis="$1"
    BYOT_FL_ARGS=()
    PLAIN_FL_ARGS=()
    LOSS_TEMPERATURE="$KD_TEMPERATURE"
    case "$axis" in
        model_mobilenet)
            AXIS_GROUP="model"
            AXIS_LABEL="mobilenet"
            BYOT_MODEL="mobilenet_byot"
            PLAIN_MODEL="mobilenet"
            JOB_BATCH_SIZE="$BATCH_SIZE"
            JOB_TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
            ;;
        model_resnet50)
            AXIS_GROUP="model"
            AXIS_LABEL="resnet50"
            BYOT_MODEL="resnet50_byot"
            PLAIN_MODEL="resnet50"
            JOB_BATCH_SIZE="$RESNET50_BATCH_SIZE"
            # ResNet50-BYOT evaluates all three branch bottlenecks.  The
            # framework default (512) needs an extra ~4 GiB at 64x64 and
            # overflows a 20 GiB GPU, even when local training batch=32 fits.
            JOB_TEST_BATCH_SIZE="$RESNET50_TEST_BATCH_SIZE"
            ;;
        mechanism_fedprox)
            AXIS_GROUP="mechanism"
            AXIS_LABEL="fedprox"
            BYOT_MODEL="resnet18_byot"
            PLAIN_MODEL="resnet18"
            JOB_BATCH_SIZE="$BATCH_SIZE"
            JOB_TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
            BYOT_FL_ARGS=(--use_fedprox --mu "$FEDPROX_MU")
            PLAIN_FL_ARGS=(--alg fedprox --mu "$FEDPROX_MU")
            ;;
        mechanism_moon)
            AXIS_GROUP="mechanism"
            AXIS_LABEL="moon"
            BYOT_MODEL="resnet18_byot"
            PLAIN_MODEL="resnet18"
            JOB_BATCH_SIZE="$BATCH_SIZE"
            JOB_TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
            LOSS_TEMPERATURE="$MOON_TEMPERATURE"
            BYOT_FL_ARGS=(--use_moon --mu "$MOON_MU")
            PLAIN_FL_ARGS=(--alg moon --mu "$MOON_MU" --temperature "$MOON_TEMPERATURE")
            ;;
        mechanism_fedavgm)
            AXIS_GROUP="mechanism"
            AXIS_LABEL="fedavgm"
            BYOT_MODEL="resnet18_byot"
            PLAIN_MODEL="resnet18"
            JOB_BATCH_SIZE="$BATCH_SIZE"
            JOB_TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
            # fedbyot + positive server_momentum activates the compositional
            # FedAvgM path in fl_utils.apply_server_side_optimization.
            BYOT_FL_ARGS=(--server_momentum "$FEDAVGM_MOMENTUM")
            PLAIN_FL_ARGS=(--alg fedavgM --server_momentum "$FEDAVGM_MOMENTUM")
            ;;
        *)
            echo "Unknown axis: $axis" >&2
            exit 1
            ;;
    esac
}

method_name() {
    case "$1" in
        plain) echo "plain" ;;
        fixed_lambda0p30) echo "fixed_lambda0p30_tkd$(value_tag "$KD_TEMPERATURE")" ;;
        soft_b_adaptive)
            echo "soft_b_tkd$(value_tag "$KD_TEMPERATURE")_lmax$(value_tag "$LAMBDA_MAX")_warm${WARMUP_ROUNDS}_tau$(value_tag "$SOFT_TAU")"
            ;;
        *)
            echo "Unknown method: $1" >&2
            exit 1
            ;;
    esac
}

has_completed_log() {
    local log_file="$1" expected_rounds="$2"
    [[ -f "$log_file" ]] && grep -q "Round $((expected_rounds - 1)) result" "$log_file"
}

append_adaptive_args() {
    CMD+=(
        --byot_alpha "$LAMBDA_MAX"
        --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00
        --byot_round_lambda_warmup "$WARMUP_ROUNDS"
        --alpha_min_scale 0.0
        --byot_client_proxy teacher_label_prob
        --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00
        --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
        --byot_client_skew_proxy prediction_entropy
        --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.00
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau "$SOFT_TAU"
        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
    )
}

run_job() {
    local gpu_id="$1" axis="$2" dataset="$3" partition="$4" method="$5"
    local name log_dir log_file
    local -a PARTITION_FLAGS CMD

    configure_axis "$axis"
    dataset_args "$dataset"
    name="$(method_name "$method")_r${ROUNDS}"
    log_dir="${LOG_ROOT}/${AXIS_GROUP}/${AXIS_LABEL}/${DATASET}/${partition}/${AXIS_LABEL}"
    log_file="${log_dir}/${name}.log"
    mkdir -p "$log_dir"

    if [[ "$SKIP_EXISTING" == "1" ]] && has_completed_log "$log_file" "$ROUNDS"; then
        echo "[skip] ${axis} | ${dataset} | ${partition} | ${method}"
        return
    fi

    mapfile -t PARTITION_FLAGS < <(partition_args "$partition")
    CMD=(
        "$PYTHON_BIN" main.py
        --dataset "$DATASET" --datadir "$DATA_DIR" --num_classes 100
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$JOB_BATCH_SIZE"
        --test_batch_size "$JOB_TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED"
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT"
        --log_file_name "${AXIS_GROUP}/${AXIS_LABEL}/${DATASET}/${partition}/${AXIS_LABEL}/${name}"
    )
    CMD+=("${PARTITION_FLAGS[@]}")

    if [[ "$method" == "plain" ]]; then
        CMD+=(--model "$PLAIN_MODEL")
        if (( ${#PLAIN_FL_ARGS[@]} == 0 )); then
            CMD+=(--alg fedavg)
        else
            CMD+=("${PLAIN_FL_ARGS[@]}")
        fi
    else
        CMD+=(
            --model "$BYOT_MODEL" --alg fedbyot
            --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum
            --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
            --temperature "$LOSS_TEMPERATURE"
            --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
            --byot_proxy_temperature "$PROXY_TEMPERATURE"
        )
        CMD+=("${BYOT_FL_ARGS[@]}")
        if [[ "$method" == "fixed_lambda0p30" ]]; then
            CMD+=(--byot_alpha "$FIXED_LAMBDA")
        else
            append_adaptive_args
        fi
    fi
    CMD+=("${WANDB_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: ${axis} | ${dataset} | ${partition} | ${method} | R=${ROUNDS}, warm=${WARMUP_ROUNDS}, batch=${JOB_BATCH_SIZE}/${JOB_TEST_BATCH_SIZE}"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run][GPU %s] ' "$gpu_id"
        printf '%q ' "${CMD[@]}"
        printf '\n'
        return
    fi
    "${CMD[@]}" > "${log_dir}/${name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${axis} | ${dataset} | ${partition} | ${method}"
}

job_weight() {
    local axis="$1" dataset="$2"
    local weight
    case "$dataset" in
        cifar100) weight=500 ;;
        tinyimagenet) weight=700 ;;
        imagenet100_64) weight=900 ;;
    esac
    [[ "$axis" == "model_resnet50" ]] && weight=$((weight * 2))
    [[ "$axis" == "mechanism_moon" ]] && weight=$((weight + weight / 4))
    printf '%d' "$weight"
}

declare -a QUEUES LOADS
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
    LOADS[$i]=0
done
JOB_COUNT=0

enqueue() {
    local axis="$1" dataset="$2" partition="$3" method="$4"
    local weight target=0
    weight="$(job_weight "$axis" "$dataset")"
    for ((i = 1; i < NUM_GPUS; i++)); do
        (( LOADS[i] < LOADS[target] )) && target=$i
    done
    QUEUES[$target]+="${axis}|${dataset}|${partition}|${method}"$'\n'
    LOADS[$target]=$((LOADS[$target] + weight))
    JOB_COUNT=$((JOB_COUNT + 1))
}

add_methods() {
    local axis="$1" dataset="$2" partition="$3"
    enqueue "$axis" "$dataset" "$partition" plain
    enqueue "$axis" "$dataset" "$partition" fixed_lambda0p30
    enqueue "$axis" "$dataset" "$partition" soft_b_adaptive
}

# Existing CIFAR-100 MobileNet/FedProx/MOON cells are deliberately not repeated.
for dataset in tinyimagenet imagenet100_64; do
    add_methods model_mobilenet "$dataset" iid
done
for dataset in cifar100 tinyimagenet imagenet100_64; do
    add_methods model_resnet50 "$dataset" iid
done
for axis in mechanism_fedprox mechanism_moon; do
    for dataset in tinyimagenet imagenet100_64; do
        for partition in beta_0.5 beta_0.3; do
            add_methods "$axis" "$dataset" "$partition"
        done
    done
done
for dataset in cifar100 tinyimagenet imagenet100_64; do
    for partition in beta_0.5 beta_0.3; do
        add_methods mechanism_fedavgm "$dataset" "$partition"
    done
done

for dataset in tinyimagenet imagenet100_64; do
    dataset_args "$dataset"
    if [[ ! -d "${DATA_DIR}/train" || ! -d "${DATA_DIR}/val" ]]; then
        echo "Dataset directories are missing for ${dataset}: ${DATA_DIR}/{train,val}" >&2
        exit 1
    fi
done

run_queue() {
    local gpu_id="$1"
    shift
    local job axis dataset partition method
    for job in "$@"; do
        [[ -z "$job" ]] && continue
        IFS='|' read -r axis dataset partition method <<< "$job"
        run_job "$gpu_id" "$axis" "$dataset" "$partition" "$method"
    done
}

echo "========== Final adaptive-lambda extension matrix =========="
echo "jobs=${JOB_COUNT}, gpus=${GPUS[*]}, seed=${SEED}"
echo "methods=plain fixed_lambda=${FIXED_LAMBDA} soft_b_adaptive"
echo "adaptive: T_KD=${KD_TEMPERATURE}, T_proxy=${PROXY_TEMPERATURE}, lmax=${LAMBDA_MAX}, tau=${SOFT_TAU}, T_soft=${SOFT_TEMPERATURE}, warmup=${LAMBDA_WARMUP_RATIO}R"
echo "FedProx mu=${FEDPROX_MU}; MOON mu=${MOON_MU}, T=${MOON_TEMPERATURE}; FedAvgM momentum=${FEDAVGM_MOMENTUM}"
echo "rounds: cifar=${CIFAR100_ROUNDS}, tiny=${TINYIMAGENET_ROUNDS}, imagenet=${IMAGENET100_ROUNDS}; ResNet50 train/test batch=${RESNET50_BATCH_SIZE}/${RESNET50_TEST_BATCH_SIZE}"
for ((i = 0; i < NUM_GPUS; i++)); do
    job_lines=$(printf '%s' "${QUEUES[$i]}" | sed '/^$/d' | wc -l)
    echo "GPU ${GPUS[$i]}: ${job_lines} queued jobs (relative load ${LOADS[$i]})"
done

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    mapfile -t queue_jobs <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
    pids+=("$!")
done

failed=0
for ((i = 0; i < NUM_GPUS; i++)); do
    if ! wait "${pids[$i]}"; then
        echo "[ERROR] GPU ${GPUS[$i]} worker stopped. Its failing command is logged under ${LOG_ROOT}." >&2
        failed=1
    fi
done
if (( failed )); then
    echo "[ERROR] Recent terminal-log output follows:" >&2
    find "$LOG_ROOT" -type f -name '*_terminal.log' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -4 \
        | cut -d' ' -f2- \
        | while IFS= read -r log_file; do
            echo "--- ${log_file}" >&2
            tail -25 "$log_file" >&2 || true
        done
    exit 1
fi
echo "Final extension matrix complete (${JOB_COUNT} jobs)."
