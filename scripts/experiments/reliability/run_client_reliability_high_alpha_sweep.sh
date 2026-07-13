#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Client-wise reliability alpha sweep with high-alpha ranges.
#
# Motivation:
#   The earlier client-wise adaptive runs used alpha_k in [0.01, 0.30],
#   but fixed FedSD improved further at alpha=0.70~1.00. This script tests
#   whether reliability-aware client-wise alpha can compete with fixed
#   high-alpha FedSD while keeping strong KD as the default regime.
#
# Default plan:
#   beta_0.1 / beta_0.3 / beta_0.5
#   proxies: teacher_label_prob, teacher_correctness, branch_js, teacher_entropy
#   alpha ranges:
#     0.05~1.00  : broad range, can suppress unreliable clients strongly
#     0.30~1.00  : high-KD range, keeps the fixed-alpha strength mostly intact
#     0.50~1.00  : direct competition with fixed high-alpha baselines
#
# Usage:
#   USE_WANDB=0 bash scripts/experiments/reliability/run_client_reliability_high_alpha_sweep.sh
#
# Useful overrides:
#   ENVS_OVERRIDE="beta_0.3 beta_0.5" USE_WANDB=0 bash scripts/experiments/reliability/run_client_reliability_high_alpha_sweep.sh
#   RANGES="0.30:1.00:0p30_1p00 0.50:1.00:0p50_1p00" USE_WANDB=0 bash scripts/experiments/reliability/run_client_reliability_high_alpha_sweep.sh
#   PROXIES="teacher_correctness branch_js" USE_WANDB=0 bash scripts/experiments/reliability/run_client_reliability_high_alpha_sweep.sh

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

BASE_ALGO="${BASE_ALGO:-fedavg}"
if [ "$BASE_ALGO" = "fedprox" ]; then
    BASE_ALGO_FLAGS="--use_fedprox --mu ${MU_VAL:-0.01}"
else
    BASE_ALGO_FLAGS=""
fi

SEED="${SEED:-0}"
BETA_VAL="${BYOT_BETA_VAL:-0.01}"
TEMP_VAL="${TEMP_VAL:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs/reliability/logs_client_reliability_high_alpha}"

ENVS=(${ENVS_OVERRIDE:-beta_0.1 beta_0.3 beta_0.5})
PROXIES=(${PROXIES:-teacher_label_prob teacher_correctness branch_js teacher_entropy})
RANGES=(${RANGES:-0.05:1.00:0p05_1p00 0.30:1.00:0p30_1p00 0.50:1.00:0p50_1p00})

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

proxy_tag() {
    local proxy=$1
    case "$proxy" in
        teacher_label_prob)
            echo "label_prob"
            ;;
        teacher_correctness)
            echo "correctness"
            ;;
        teacher_entropy)
            echo "entropy"
            ;;
        branch_js)
            echo "branch_js"
            ;;
        *)
            echo "${proxy}"
            ;;
    esac
}

run_job() {
    local env_name=$1
    local proxy=$2
    local alpha_min=$3
    local alpha_max=$4
    local range_tag=$5

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local tag
    tag="$(proxy_tag "${proxy}")"
    local method_name="client_${tag}_${range_tag}"
    local log_dir="${LOG_ROOT}/${env_name}/${BASE_ALGO}"
    local flags
    flags="$(env_flags "${env_name}")"

    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] start: ${env_name} | ${BASE_ALGO} | ${method_name} | client_alpha=${alpha_min}~${alpha_max}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/${BASE_ALGO}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_alpha "${alpha_max}" --byot_beta "${BETA_VAL}" --temperature "${TEMP_VAL}" \
        --byot_client_proxy "${proxy}" \
        --byot_client_alpha_min "${alpha_min}" \
        --byot_client_alpha_max "${alpha_max}" \
        ${flags} ${BASE_ALGO_FLAGS} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

echo "========== Client-wise Reliability High-Alpha Sweep =========="
echo "base_algo=${BASE_ALGO}, seed=${SEED}, log_root=${LOG_ROOT}"
echo "envs=${ENVS[*]}"
echo "proxies=${PROXIES[*]}"
echo "ranges=${RANGES[*]}"

for env_name in "${ENVS[@]}"; do
    for range_spec in "${RANGES[@]}"; do
        alpha_min="$(echo "${range_spec}" | cut -d: -f1)"
        alpha_max="$(echo "${range_spec}" | cut -d: -f2)"
        range_tag="$(echo "${range_spec}" | cut -d: -f3)"
        for proxy in "${PROXIES[@]}"; do
            run_job "${env_name}" "${proxy}" "${alpha_min}" "${alpha_max}" "${range_tag}"
        done
    done
done

wait
echo "Client-wise reliability high-alpha sweep complete (${JOB_COUNT} jobs)"
