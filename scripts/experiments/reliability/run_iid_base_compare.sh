#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# IID comparison for FedProx/MOON base methods.
#
# Runs per base method:
#   1) plain ResNet baseline
#   2) BYOT/FedSD fixed alpha: 0.50 / 0.70 / 1.00
#   3) BYOT/FedSD client-wise adaptive alpha: branch_js, 0.50~1.00
#
# This complements:
#   scripts/experiments/reliability/run_iid_fedavg_compare.sh

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
FEDPROX_MU="${FEDPROX_MU:-0.01}"
MOON_MU="${MOON_MU:-1.0}"
LOG_ROOT="${LOG_ROOT:-logs/reliability/logs_iid_base_compare}"

BASE_ALGOS=(${BASE_ALGOS:-fedprox moon})
FIXED_ALPHAS=(${FIXED_ALPHAS:-0.50:0p50 0.70:0p70 1.00:1p00})
ADAPTIVE_SPECS=(${ADAPTIVE_SPECS:-branch_js:0.50:1.00:client_branch_js_0p50_1p00})

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

plain_algo_flags() {
    local base_algo=$1
    case "$base_algo" in
        fedprox)
            echo "--model resnet18 --alg fedprox --mu ${FEDPROX_MU}"
            ;;
        moon)
            echo "--model resnet18 --alg moon --mu ${MOON_MU} --temperature ${TEMP_VAL}"
            ;;
        *)
            echo "Unknown base algorithm: ${base_algo}" >&2
            exit 1
            ;;
    esac
}

byot_regularizer_flags() {
    local base_algo=$1
    case "$base_algo" in
        fedprox)
            echo "--use_fedprox --mu ${FEDPROX_MU}"
            ;;
        moon)
            echo "--use_moon --mu ${MOON_MU}"
            ;;
        *)
            echo "Unknown base algorithm: ${base_algo}" >&2
            exit 1
            ;;
    esac
}

run_job() {
    local base_algo=$1
    local method_name=$2
    local extra_flags=$3

    local gpu_idx=$(( JOB_COUNT % NUM_GPUS ))
    local gpu_id=${GPUS[$gpu_idx]}
    local log_dir="${LOG_ROOT}/iid/${base_algo}"

    mkdir -p "$log_dir"

    echo "[GPU ${gpu_id}] start: iid | ${base_algo} | ${method_name}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --n_clients 100 --sample_fraction 0.1 \
        --epochs 5 --lr 0.1 --batch_size 64 --round 500 --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --partition iid \
        --logdir "${LOG_ROOT}" \
        --log_file_name "iid/${base_algo}/${method_name}" \
        ${extra_flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1 &

    JOB_COUNT=$((JOB_COUNT + 1))

    if (( JOB_COUNT % NUM_GPUS == 0 )); then
        wait
        echo "batch complete (${JOB_COUNT} jobs)"
    fi
}

echo "========== IID Base Compare =========="
echo "base_algos=${BASE_ALGOS[*]}, seed=${SEED}, log_root=${LOG_ROOT}"
echo "byot_beta=${BYOT_BETA_VAL}, temp=${TEMP_VAL}, fedprox_mu=${FEDPROX_MU}, moon_mu=${MOON_MU}"
echo "fixed_alphas=${FIXED_ALPHAS[*]}"
echo "adaptive_specs=${ADAPTIVE_SPECS[*]}"
echo "wandb=${USE_WANDB:-1}"

for base_algo in "${BASE_ALGOS[@]}"; do
    run_job "${base_algo}" "plain_baseline" "$(plain_algo_flags "${base_algo}")"

    regularizer_flags="$(byot_regularizer_flags "${base_algo}")"
    for alpha_pair in "${FIXED_ALPHAS[@]}"; do
        alpha_val="${alpha_pair%%:*}"
        alpha_tag="${alpha_pair##*:}"
        run_job "${base_algo}" "fixed_alpha${alpha_tag}" \
            "--model resnet18_byot --alg fedbyot --kd_conf_threshold 0.0 --byot_alpha ${alpha_val} --byot_beta ${BYOT_BETA_VAL} --temperature ${TEMP_VAL} ${regularizer_flags}"
    done

    for spec in "${ADAPTIVE_SPECS[@]}"; do
        proxy="${spec%%:*}"
        rest="${spec#*:}"
        alpha_min="${rest%%:*}"
        rest="${rest#*:}"
        alpha_max="${rest%%:*}"
        method_name="${rest#*:}"
        run_job "${base_algo}" "${method_name}" \
            "--model resnet18_byot --alg fedbyot --kd_conf_threshold 0.0 --byot_alpha ${alpha_max} --byot_beta ${BYOT_BETA_VAL} --temperature ${TEMP_VAL} --byot_client_proxy ${proxy} --byot_client_alpha_min ${alpha_min} --byot_client_alpha_max ${alpha_max} ${regularizer_flags}"
    done
done

wait
echo "IID base compare complete (${JOB_COUNT} jobs)"
