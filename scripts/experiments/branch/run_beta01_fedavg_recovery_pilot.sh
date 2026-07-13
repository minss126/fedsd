#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Beta_0.1 FedAvg recovery pilot for BYOT/FedSD.
#
# Careful with naming:
#   - partition beta is fixed to Dirichlet beta=0.1
#   - --byot_beta is the BYOT feature imitation weight
#
# Goal:
#   Try to recover the beta_0.1 FedAvg gap against the plain FedAvg baseline
#   without switching to FedProx/MOON.
#
# Current references:
#   plain FedAvg beta_0.1 Last-10 ~= 55.299
#   all-branch alpha=1.0, T=0.5, byot_beta=0.01 Last-10 ~= 53.553
#   B1-only alpha=1.0, T=0.5, byot_beta=0.01 Last-10 ~= 54.714

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
JOB_COUNT=0

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then
        PYTHON_BIN="venv/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

SEED="${SEED:-0}"
BYOT_ALPHA_VAL="${BYOT_ALPHA_VAL:-1.0}"
LOG_ROOT="${LOG_ROOT:-logs/branch/logs_beta01_fedavg_recovery_pilot}"

# Format:
#   method_name:active_branches:temperature:byot_feature_beta
METHODS=(
    "b1_temp1_byotbeta0p01:1:1.0:0.01"
    "b1_temp2_byotbeta0p01:1:2.0:0.01"
    "b1_temp1_byotbeta0p00:1:1.0:0.00"
    "b1_temp2_byotbeta0p00:1:2.0:0.00"
    "all_temp1_byotbeta0p00:1,2,3:1.0:0.00"
)

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

run_job() {
    local method_name=$1
    local active_branches=$2
    local temp_val=$3
    local byot_feature_beta=$4

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local log_dir="${LOG_ROOT}/beta_0.1/fedavg"

    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] start: beta_0.1 | fedavg | ${method_name} | active=${active_branches} | T=${temp_val} | byot_beta=${byot_feature_beta}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --partition noniid --beta 0.1 \
        --logdir "${LOG_ROOT}" \
        --log_file_name "beta_0.1/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${BYOT_ALPHA_VAL}" \
        --byot_beta "${byot_feature_beta}" \
        --temperature "${temp_val}" \
        --byot_active_branches "${active_branches}" \
        ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

echo "========== Beta_0.1 FedAvg Recovery Pilot =========="
echo "seed=${SEED}, log_root=${LOG_ROOT}"
echo "byot_alpha=${BYOT_ALPHA_VAL}"
echo "NOTE: partition beta is fixed at 0.1; method tags use BYOT feature beta."
echo "methods=${METHODS[*]}"

for spec in "${METHODS[@]}"; do
    method_name="${spec%%:*}"
    rest="${spec#*:}"
    active_branches="${rest%%:*}"
    rest="${rest#*:}"
    temp_val="${rest%%:*}"
    byot_feature_beta="${rest#*:}"
    run_job "${method_name}" "${active_branches}" "${temp_val}" "${byot_feature_beta}"
done

wait
echo "Beta_0.1 FedAvg recovery pilot complete (${JOB_COUNT} jobs)"
