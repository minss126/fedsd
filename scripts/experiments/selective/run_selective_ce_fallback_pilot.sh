#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Quick pilot for Selective KD with CE fallback.
#
# Branch loss per sample:
#   (1 - alpha * r(x)) * CE + (alpha * r(x)) * KD
#
# Default target is beta_0.1/FedAvg because that is where fixed high-alpha
# BYOT/FedSD currently fails to beat the plain baseline.

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
LOG_ROOT="${LOG_ROOT:-logs_selective_ce_fallback_pilot}"
ENVS=(${ENVS_OVERRIDE:-beta_0.1})

METHODS=(
    "fallback_conf0p60:--kd_conf_threshold 0.60"
    "fallback_conf0p80:--kd_conf_threshold 0.80"
    "fallback_branch_js:--byot_sample_proxy branch_js --alpha_min_scale 0.0"
    "fallback_label_prob:--byot_sample_proxy teacher_label_prob --alpha_min_scale 0.0"
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
    local extra_flags=$3

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local log_dir="${LOG_ROOT}/${env_name}/fedavg"
    local flags
    flags="$(env_flags "${env_name}")"

    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] start: ${env_name} | ${method_name}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot_selective_ce_fallback \
        --byot_alpha "${BYOT_ALPHA_VAL}" --byot_beta "${BYOT_BETA_VAL}" --temperature "${TEMP_VAL}" \
        ${flags} ${extra_flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

echo "========== Selective CE Fallback Pilot =========="
echo "envs=${ENVS[*]}, seed=${SEED}, log_root=${LOG_ROOT}"
echo "alpha=${BYOT_ALPHA_VAL}, byot_beta=${BYOT_BETA_VAL}, temp=${TEMP_VAL}"
echo "methods=${METHODS[*]}"

for env_name in "${ENVS[@]}"; do
    for method_pair in "${METHODS[@]}"; do
        method_name="${method_pair%%:*}"
        extra_flags="${method_pair#*:}"
        run_job "${env_name}" "${method_name}" "${extra_flags}"
    done
done

wait
echo "Selective CE fallback pilot complete (${JOB_COUNT} jobs)"
