#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Naive heterogeneity-aware BYOT branch gating pilot.
#
# Uses client normalized label entropy H_k instead of the unknown Dirichlet beta.
# Gate variants:
#   entropy_3stage:
#     H_k < off_threshold         -> branch off
#     off_threshold <= H_k < b1_threshold -> B1 only
#     H_k >= b1_threshold         -> B1+B2+B3
#   entropy_no_off:
#     H_k < b1_threshold          -> B1 only
#     H_k >= b1_threshold         -> B1+B2+B3
#
# This is FedAvg-only to quickly check whether entropy-based dynamic branch
# control can keep IID/moderate gains while reducing beta_0.1 damage.

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
BYOT_BETA_VAL="${BYOT_BETA_VAL:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs/branch/logs_entropy_branch_gate_pilot}"
ENVS=(${ENVS_OVERRIDE:-beta_0.1 beta_0.3 beta_0.5 iid})

# Format:
#   method_name:gate:off_threshold:b1_threshold
METHODS=(
    "gate_3stage_0p20_0p50:entropy_3stage:0.20:0.50"
    "gate_3stage_0p30_0p60:entropy_3stage:0.30:0.60"
    "gate_nooff_0p50:entropy_no_off:0.00:0.50"
)

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

run_job() {
    local env_name=$1
    local method_name=$2
    local gate=$3
    local off_threshold=$4
    local b1_threshold=$5

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local log_dir="${LOG_ROOT}/${env_name}/fedavg"
    local flags
    flags="$(env_flags "${env_name}")"

    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] start: ${env_name} | ${method_name} | gate=${gate} | off=${off_threshold} | b1=${b1_threshold}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${BYOT_ALPHA_VAL}" \
        --byot_beta "${BYOT_BETA_VAL}" \
        --temperature "${TEMP_VAL}" \
        --byot_branch_gate "${gate}" \
        --branch_entropy_off_threshold "${off_threshold}" \
        --branch_entropy_b1_threshold "${b1_threshold}" \
        ${flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

echo "========== Entropy Branch Gate Pilot =========="
echo "envs=${ENVS[*]}, seed=${SEED}, log_root=${LOG_ROOT}"
echo "alpha=${BYOT_ALPHA_VAL}, byot_beta=${BYOT_BETA_VAL}, temp=${TEMP_VAL}"
echo "methods=${METHODS[*]}"

for env_name in "${ENVS[@]}"; do
    for spec in "${METHODS[@]}"; do
        method_name="${spec%%:*}"
        rest="${spec#*:}"
        gate="${rest%%:*}"
        rest="${rest#*:}"
        off_threshold="${rest%%:*}"
        b1_threshold="${rest#*:}"
        run_job "${env_name}" "${method_name}" "${gate}" "${off_threshold}" "${b1_threshold}"
    done
done

wait
echo "Entropy branch gate pilot complete (${JOB_COUNT} jobs)"
