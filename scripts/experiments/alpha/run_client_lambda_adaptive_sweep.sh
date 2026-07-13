#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Client-wise adaptive KD-intensity sweep for BYOT/FedSD.
#
# Loss:
#   L = CE_teacher + lambda_k * KD_branch + beta_feat * L_feat
#
# Unlike scripts/experiments/alpha/run_round_lambda_adaptive_sweep.sh,
# lambda is estimated once per selected client before local training:
#   lambda_k = lambda_min + (lambda_max - lambda_min) * reliability_k
#
# The reliability score is computed from the client's local data using
# --byot_client_proxy. No branch CE is used because --byot_branch_objective
# is set to kd_only.
#
# Default plan:
#   CIFAR-100 / ResNet18-BYOT / FedAvg / kd_only
#   partitions: iid, beta_0.5, beta_0.3, beta_0.1
#   proxies: teacher_label_prob, teacher_correctness, branch_js, teacher_entropy
#   ranges:
#     0.00 -> 1.00
#     0.10 -> 3.00
#
# Usage:
#   bash scripts/experiments/alpha/run_client_lambda_adaptive_sweep.sh
#
# Useful overrides:
#   USE_WANDB=0 bash scripts/experiments/alpha/run_client_lambda_adaptive_sweep.sh
#   GPUS_OVERRIDE="0 1" USE_WANDB=0 bash ...
#   ENVS_OVERRIDE="beta_0.1 beta_0.3" bash ...
#   PROXIES_OVERRIDE="branch_js teacher_label_prob" bash ...
#   RANGES_OVERRIDE="0.00:1.00:0_1 0.10:3.00:0p1_3" bash ...

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
NUM_WORKERS="${NUM_WORKERS:-0}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_client_lambda_adaptive}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ENVS=(${ENVS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})
PROXIES=(${PROXIES_OVERRIDE:-teacher_label_prob teacher_correctness branch_js teacher_entropy})
RANGES=(${RANGES_OVERRIDE:-0.00:1.00:0_1 0.10:3.00:0p1_3})

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

proxy_tag() {
    local proxy=$1
    case "$proxy" in
        teacher_label_prob) echo "label_prob" ;;
        teacher_correctness) echo "correctness" ;;
        teacher_entropy) echo "entropy" ;;
        teacher_margin) echo "margin" ;;
        teacher_conf) echo "teacher_conf" ;;
        branch_js) echo "branch_js" ;;
        branch_soft_kl) echo "branch_soft_kl" ;;
        branch_agreement) echo "branch_agree" ;;
        *) echo "$proxy" ;;
    esac
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1
    local env_name=$2
    local proxy=$3
    local lambda_min=$4
    local lambda_max=$5
    local range_tag=$6
    local ptag
    local method_name
    local log_dir
    local log_file
    local flags

    ptag="$(proxy_tag "$proxy")"
    method_name="client_lambda_${ptag}_${range_tag}"
    log_dir="${LOG_ROOT}/${env_name}/fedavg"
    log_file="${log_dir}/${method_name}.log"
    flags="$(env_flags "$env_name")"

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | ${method_name}"
        return
    fi

    echo "[GPU ${gpu_id}] start: ${env_name} | ${method_name} | proxy=${proxy} | lambda=${lambda_min}->${lambda_max}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${LR}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_active_branches "1,2,3" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only \
        --byot_alpha "${lambda_max}" \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        --byot_client_proxy "${proxy}" \
        --byot_client_alpha_min "${lambda_min}" \
        --byot_client_alpha_max "${lambda_max}" \
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
        IFS='|' read -r env_name proxy lambda_min lambda_max range_tag <<< "$job"
        run_job "$gpu_id" "$env_name" "$proxy" "$lambda_min" "$lambda_max" "$range_tag"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for env_name in "${ENVS[@]}"; do
    for proxy in "${PROXIES[@]}"; do
        for range_spec in "${RANGES[@]}"; do
            IFS=':' read -r lambda_min lambda_max range_tag <<< "$range_spec"
            gpu_idx=$((job_count % NUM_GPUS))
            QUEUES[$gpu_idx]+="${env_name}|${proxy}|${lambda_min}|${lambda_max}|${range_tag}"$'\n'
            job_count=$((job_count + 1))
        done
    done
done

echo "========== Client-wise KD Lambda Adaptive Sweep =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "envs=${ENVS[*]}"
echo "proxies=${PROXIES[*]}"
echo "ranges=${RANGES[*]}"
echo "feature_beta=${FEATURE_BETA}, temp=${TEMP_VAL}, log_root=${LOG_ROOT}"
echo "skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"
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
echo "Client-wise KD lambda adaptive sweep complete (${job_count} jobs)"
