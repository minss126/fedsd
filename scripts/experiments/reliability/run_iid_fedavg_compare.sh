#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# IID FedAvg comparison set.
#
# Runs:
#   1) plain ResNet FedAvg baseline
#   2) BYOT/FedSD fixed alpha: 0.50 / 0.70 / 1.00
#   3) BYOT/FedSD client-wise adaptive alpha: branch_js, 0.50~1.00
#
# This is intentionally FedAvg-only to keep the IID sanity check short enough
# to run after the branch-wise non-IID sweep.

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
BYOT_BETA_VAL="${BYOT_BETA_VAL:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs_iid_fedavg_compare}"

FIXED_ALPHAS=(${FIXED_ALPHAS:-0.50:0p50 0.70:0p70 1.00:1p00})
ADAPTIVE_SPECS=(${ADAPTIVE_SPECS:-branch_js:0.50:1.00:client_branch_js_0p50_1p00})

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

run_job() {
    local method_name=$1
    local extra_flags=$2

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local log_dir="${LOG_ROOT}/iid/fedavg"

    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] start: iid | fedavg | ${method_name}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --partition iid \
        --logdir "${LOG_ROOT}" \
        --log_file_name "iid/fedavg/${method_name}" \
        ${extra_flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

echo "========== IID FedAvg Compare =========="
echo "seed=${SEED}, log_root=${LOG_ROOT}"
echo "byot_beta=${BYOT_BETA_VAL}, temp=${TEMP_VAL}"
echo "fixed_alphas=${FIXED_ALPHAS[*]}"
echo "adaptive_specs=${ADAPTIVE_SPECS[*]}"
echo "wandb=${USE_WANDB:-1}"

run_job "plain_baseline" "--model resnet18 --alg fedavg"

for alpha_pair in "${FIXED_ALPHAS[@]}"; do
    alpha_val="${alpha_pair%%:*}"
    alpha_tag="${alpha_pair##*:}"
    run_job "fixed_alpha${alpha_tag}" \
        "--model resnet18_byot --alg fedbyot --kd_conf_threshold 0.0 --byot_alpha ${alpha_val} --byot_beta ${BYOT_BETA_VAL} --temperature ${TEMP_VAL}"
done

for spec in "${ADAPTIVE_SPECS[@]}"; do
    proxy="${spec%%:*}"
    rest="${spec#*:}"
    alpha_min="${rest%%:*}"
    rest="${rest#*:}"
    alpha_max="${rest%%:*}"
    method_name="${rest#*:}"
    run_job "${method_name}" \
        "--model resnet18_byot --alg fedbyot --kd_conf_threshold 0.0 --byot_alpha ${alpha_max} --byot_beta ${BYOT_BETA_VAL} --temperature ${TEMP_VAL} --byot_client_proxy ${proxy} --byot_client_alpha_min ${alpha_min} --byot_client_alpha_max ${alpha_max}"
done

wait
echo "IID FedAvg compare complete (${JOB_COUNT} jobs)"
