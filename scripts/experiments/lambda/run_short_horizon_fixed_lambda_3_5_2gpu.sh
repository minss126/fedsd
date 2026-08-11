#!/usr/bin/env bash

# Fixed-lambda extension for the 100-round datasets.
# Adds lambda={3,5} under the same setup used by the existing TinyImageNet and
# ImageNet100-64 lambda comparisons: ResNet18-BYOT, FedAvg, E=5, C=0.1,
# T_KD=1, no adaptive proxy, and no warm-up for fixed lambda.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  GPUS_OVERRIDE="0 1" bash scripts/experiments/lambda/run_short_horizon_fixed_lambda_3_5_2gpu.sh

Defaults:
  datasets=tinyimagenet imagenet100_64
  partitions=iid beta_0.5 beta_0.3 beta_0.1
  fixed lambdas=3.0 5.0
  rounds=100, local epochs=5, seed=0

Useful overrides:
  USE_WANDB=0 DRY_RUN=1 GPUS_OVERRIDE="0 1" bash ...
  SKIP_EXISTING=0             Re-run completed jobs.
  FIXED_LAMBDAS="3.0 5.0"     Change the values explicitly.
EOF
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1})
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
ROUNDS="${ROUNDS:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
TINYIMAGENET_DATADIR="${TINYIMAGENET_DATADIR:-./data/tiny-imagenet-200}"
IMAGENET100_DATADIR="${IMAGENET100_DATADIR:-/data/imagenet100_resized_64_png}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/analysis/logs_short_horizon_fixed_lambda_3_5}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

DATASETS=(${DATASETS_OVERRIDE:-tinyimagenet imagenet100_64})
PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})
FIXED_LAMBDAS=(${FIXED_LAMBDAS:-3.0 5.0})

WANDB_FLAGS=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
fi

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

configure_dataset() {
    DATASET="$1"
    case "$DATASET" in
        tinyimagenet)
            DATA_DIR="$TINYIMAGENET_DATADIR"
            LR="0.01"
            NUM_WORKERS="2"
            ;;
        imagenet100_64)
            DATA_DIR="$IMAGENET100_DATADIR"
            LR="0.01"
            NUM_WORKERS="2"
            ;;
        *)
            echo "Unsupported dataset: $DATASET" >&2
            exit 1
            ;;
    esac
}

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *)
            echo "Unsupported partition: $1" >&2
            exit 1
            ;;
    esac
}

has_completed_log() {
    local log_file="$1"
    [[ -f "$log_file" ]] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id="$1" dataset="$2" partition="$3" lambda="$4"
    local tag name log_dir log_file
    local -a PARTITION_FLAGS CMD

    configure_dataset "$dataset"
    tag="$(value_tag "$lambda")"
    name="fixed_lambda${tag}_tkd$(value_tag "$KD_TEMPERATURE")_r${ROUNDS}"
    log_dir="${LOG_ROOT}/${DATASET}/${partition}/fedavg"
    log_file="${log_dir}/${name}.log"
    mkdir -p "$log_dir"

    if [[ "$SKIP_EXISTING" == "1" ]] && has_completed_log "$log_file"; then
        echo "[skip] ${dataset} | ${partition} | lambda=${lambda}"
        return
    fi

    mapfile -t PARTITION_FLAGS < <(partition_args "$partition")
    CMD=(
        "$PYTHON_BIN" main.py
        --dataset "$DATASET" --datadir "$DATA_DIR" --num_classes 100
        --model resnet18_byot --alg fedbyot
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED"
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT"
        --log_file_name "${DATASET}/${partition}/fedavg/${name}"
        --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_alpha "$lambda"
    )
    CMD+=("${PARTITION_FLAGS[@]}" "${WANDB_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: ${dataset} | ${partition} | fixed lambda=${lambda} | R=${ROUNDS}"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run][GPU %s] ' "$gpu_id"
        printf '%q ' "${CMD[@]}"
        printf '\n'
        return
    fi
    "${CMD[@]}" > "${log_dir}/${name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${dataset} | ${partition} | fixed lambda=${lambda}"
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done
JOB_COUNT=0
for dataset in "${DATASETS[@]}"; do
    for partition in "${PARTITIONS[@]}"; do
        for lambda in "${FIXED_LAMBDAS[@]}"; do
            gpu_index=$((JOB_COUNT % NUM_GPUS))
            QUEUES[$gpu_index]+="${dataset}|${partition}|${lambda}"$'\n'
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

for dataset in "${DATASETS[@]}"; do
    configure_dataset "$dataset"
    if [[ ! -d "${DATA_DIR}/train" || ! -d "${DATA_DIR}/val" ]]; then
        echo "Dataset directories are missing for ${dataset}: ${DATA_DIR}/{train,val}" >&2
        exit 1
    fi
done

run_queue() {
    local gpu_id="$1"
    shift
    local job dataset partition lambda
    for job in "$@"; do
        [[ -z "$job" ]] && continue
        IFS='|' read -r dataset partition lambda <<< "$job"
        run_job "$gpu_id" "$dataset" "$partition" "$lambda"
    done
}

echo "========== Short-horizon fixed-lambda sweep =========="
echo "jobs=${JOB_COUNT}, gpus=${GPUS[*]}, seed=${SEED}"
echo "datasets=${DATASETS[*]}, partitions=${PARTITIONS[*]}, lambdas=${FIXED_LAMBDAS[*]}"
echo "rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, batch_size=${BATCH_SIZE}, fixed warm-up=none"
echo "log_root=${LOG_ROOT}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    mapfile -t queue_jobs <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
    pids+=("$!")
done
wait "${pids[@]}"
echo "Short-horizon fixed-lambda sweep complete (${JOB_COUNT} jobs)."
