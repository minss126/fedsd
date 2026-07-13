#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# KD-intensity sweep for BYOT/FedSD.
#
# This is different from the original BYOT blend alpha:
#   blend:   L = CE_teacher + (1-alpha) CE_branch + alpha KD_branch + beta_feat L_feat
#   kd_only: L = CE_teacher + lambda_kd KD_branch + beta_feat L_feat
#
# Here --byot_alpha is interpreted as lambda_kd because
# --byot_branch_objective kd_only is used.
#
# Default plan:
#   CIFAR-100 / ResNet18-BYOT / FedAvg
#   partitions: iid, beta_0.3, beta_0.5
#   lambda_kd: 0, 0.01, 0.1, 0.3, 1, 3, 10
#
# beta_0.1 kd_only results already exist in:
#   logs/alpha/logs_byot_beta_kd_only_alpha/beta_0.1/fedavg/
# Set INCLUDE_BETA01=1 if you want to rerun beta_0.1 in this log root too.
#
# Usage:
#   bash scripts/experiments/alpha/run_kd_lambda_sweep.sh
#
# Useful overrides:
#   USE_WANDB=0 bash scripts/experiments/alpha/run_kd_lambda_sweep.sh
#   INCLUDE_BETA01=1 bash scripts/experiments/alpha/run_kd_lambda_sweep.sh
#   ENVS_OVERRIDE="beta_0.3" LAMBDAS="0.1:0p100 1.0:1p000" bash ...

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPU ids provided. Set GPUS_OVERRIDE." >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then
        PYTHON_BIN="venv/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_kd_lambda_sweep}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [ -n "${ENVS_OVERRIDE:-}" ]; then
    ENVS=(${ENVS_OVERRIDE})
else
    ENVS=(iid beta_0.3 beta_0.5)
    if [ "${INCLUDE_BETA01:-0}" = "1" ]; then
        ENVS+=(beta_0.1)
    fi
fi

LAMBDAS=(${LAMBDAS:-0.00:0p000 0.01:0p010 0.10:0p100 0.30:0p300 1.00:1p000 3.00:3p000 10.00:10p000})

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

env_flags() {
    local env_name=$1
    case "$env_name" in
        iid)
            echo "--partition iid"
            ;;
        beta_0.1)
            echo "--partition noniid --beta 0.1"
            ;;
        beta_0.3)
            echo "--partition noniid --beta 0.3"
            ;;
        beta_0.5)
            echo "--partition noniid --beta 0.5"
            ;;
        *)
            echo "Unknown env: ${env_name}" >&2
            exit 1
            ;;
    esac
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1
    local env_name=$2
    local lambda_val=$3
    local lambda_tag=$4
    local method_name="kd_lambda${lambda_tag}"
    local log_dir="${LOG_ROOT}/${env_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local flags

    flags="$(env_flags "$env_name")"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | ${method_name}"
        return
    fi

    echo "[GPU ${gpu_id}] start: ${env_name} | ${method_name} | lambda_kd=${lambda_val}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${LR}" --batch_size "${BATCH_SIZE}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_active_branches "1,2,3" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only \
        --byot_alpha "${lambda_val}" \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        ${flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${env_name} | ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")

    for job in "${jobs[@]}"; do
        [ -z "$job" ] && continue
        IFS='|' read -r env_name lambda_val lambda_tag <<< "$job"
        run_job "$gpu_id" "$env_name" "$lambda_val" "$lambda_tag"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for env_name in "${ENVS[@]}"; do
    for lambda_pair in "${LAMBDAS[@]}"; do
        lambda_val="${lambda_pair%%:*}"
        lambda_tag="${lambda_pair##*:}"
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${env_name}|${lambda_val}|${lambda_tag}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== KD Lambda Sweep =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "envs=${ENVS[*]}"
echo "lambdas=${LAMBDAS[*]}"
echo "feature_beta=${FEATURE_BETA}, temp=${TEMP_VAL}, log_root=${LOG_ROOT}"
echo "jobs=${job_count}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done

wait "${pids[@]}"
echo "KD lambda sweep complete (${job_count} jobs)"
