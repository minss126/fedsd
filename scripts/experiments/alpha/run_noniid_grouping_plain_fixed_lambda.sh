#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Plain and fixed-lambda controls for noniid_grouping.
#
# Motivation:
#   Adaptive lambda should be compared against plain FL and fixed lambda under
#   the same heterogeneous grouping partition. Jobs are intentionally ordered by
#   priority, so interrupting midway still leaves the most important controls.
#
# Priority order:
#   1) plain FedAvg / ResNet18
#   2) fixed lambda=0.0
#   3) fixed lambda=1.0
#   4) fixed lambda=3.0
#   5) fixed lambda=0.3
#   6) fixed lambda=0.5
#
# Usage:
#   bash scripts/experiments/alpha/run_noniid_grouping_plain_fixed_lambda.sh
#
# Useful overrides:
#   GPUS_OVERRIDE="0 1" USE_WANDB=1 bash scripts/experiments/alpha/run_noniid_grouping_plain_fixed_lambda.sh
#   JOBS_OVERRIDE="plain kd_0p000 kd_3p000" bash scripts/experiments/alpha/run_noniid_grouping_plain_fixed_lambda.sh

GPUS=(${GPUS_OVERRIDE:-0 1})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPU ids provided. Set GPUS_OVERRIDE." >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then
        PYTHON_BIN="venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
PARTITION_GROUPS="${PARTITION_GROUPS:-8}"
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_noniid_grouping_plain_fixed_lambda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

DEFAULT_JOBS=(
    "plain|plain|"
    "kd_0p000|fixed_lambda0p000|0.00"
    "kd_1p000|fixed_lambda1p000|1.00"
    "kd_3p000|fixed_lambda3p000|3.00"
    "kd_0p300|fixed_lambda0p300|0.30"
    "kd_0p500|fixed_lambda0p500|0.50"
)

if [ -n "${JOBS_OVERRIDE:-}" ]; then
    JOBS=()
    for job in ${JOBS_OVERRIDE}; do
        case "$job" in
            plain) JOBS+=("plain|plain|") ;;
            kd_0p000) JOBS+=("kd_0p000|fixed_lambda0p000|0.00") ;;
            kd_0p300) JOBS+=("kd_0p300|fixed_lambda0p300|0.30") ;;
            kd_0p500) JOBS+=("kd_0p500|fixed_lambda0p500|0.50") ;;
            kd_1p000) JOBS+=("kd_1p000|fixed_lambda1p000|1.00") ;;
            kd_3p000) JOBS+=("kd_3p000|fixed_lambda3p000|3.00") ;;
            *)
                echo "Unknown job in JOBS_OVERRIDE: ${job}" >&2
                exit 1
                ;;
        esac
    done
else
    JOBS=("${DEFAULT_JOBS[@]}")
fi

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

PARTITION_FLAGS="--partition noniid_grouping --partition_groups ${PARTITION_GROUPS}"

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_plain() {
    local gpu_id=$1
    local method_name=$2
    local log_dir="${LOG_ROOT}/noniid_grouping/fedavg"
    local log_file="${log_dir}/${method_name}.log"

    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] noniid_grouping | ${method_name}"
        return
    fi

    echo "[GPU ${gpu_id}] start: noniid_grouping | plain FedAvg | ResNet18"
    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --datadir ./data --num_classes 100 \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${LR}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "noniid_grouping/fedavg/${method_name}" \
        --model resnet18 --alg fedavg \
        ${PARTITION_FLAGS} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: noniid_grouping | ${method_name}"
}

run_fixed_lambda() {
    local gpu_id=$1
    local method_name=$2
    local lambda_val=$3
    local log_dir="${LOG_ROOT}/noniid_grouping/fedavg"
    local log_file="${log_dir}/${method_name}.log"

    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] noniid_grouping | ${method_name}"
        return
    fi

    echo "[GPU ${gpu_id}] start: noniid_grouping | ${method_name} | lambda=${lambda_val}"
    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --datadir ./data --num_classes 100 \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${LR}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "noniid_grouping/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_active_branches "1,2,3" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only \
        --byot_alpha "${lambda_val}" \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        ${PARTITION_FLAGS} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: noniid_grouping | ${method_name}"
}

run_job() {
    local gpu_id=$1
    local job_type=$2
    local method_name=$3
    local lambda_val=$4

    case "$job_type" in
        plain) run_plain "$gpu_id" "$method_name" ;;
        kd_*) run_fixed_lambda "$gpu_id" "$method_name" "$lambda_val" ;;
        *)
            echo "Unknown job type: ${job_type}" >&2
            exit 1
            ;;
    esac
}

echo "========== noniid_grouping Plain / Fixed Lambda Controls =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "partition_groups=${PARTITION_GROUPS}, feature_beta=${FEATURE_BETA}, temp=${TEMP_VAL}"
echo "log_root=${LOG_ROOT}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"
echo "jobs=${#JOBS[@]} (${JOBS[*]})"

job_count=0
for ((offset = 0; offset < ${#JOBS[@]}; offset += NUM_GPUS)); do
    pids=()
    for ((i = 0; i < NUM_GPUS; i++)); do
        idx=$((offset + i))
        if [ "$idx" -ge "${#JOBS[@]}" ]; then
            break
        fi
        IFS='|' read -r job_type method_name lambda_val <<< "${JOBS[$idx]}"
        run_job "${GPUS[$i]}" "$job_type" "$method_name" "$lambda_val" &
        pids+=("$!")
        job_count=$((job_count + 1))
    done
    wait "${pids[@]}"
    echo "priority batch complete (${job_count}/${#JOBS[@]} jobs)"
done

echo "noniid_grouping plain/fixed-lambda controls complete (${job_count} jobs)"
