#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Compact follow-up sweep for KD-only BYOT lambda adaptation.
#
# Tests two extension axes without exploding the matrix:
#   A) proxy composition:
#      lambda_k = label_prob_k * entropy_confidence_k
#   B) unit composition:
#      lambda_{k,t} = round_schedule(t) * reliability_k
#   C) both:
#      lambda_{k,t} = round_schedule(t) * label_prob_entropy_k
#
# Loss:
#   L = CE_teacher + lambda * KD_branch + beta_feat * L_feat
#
# Default matrix:
#   CIFAR-100 / ResNet18-BYOT / FedAvg / kd_only
#   partitions: iid, beta_0.5, beta_0.3, beta_0.1
#   methods:
#     client_label_prob_entropy_0_1
#     round_client_label_prob_0_1_w250
#     round_client_label_prob_entropy_0_1_w250
#   total: 12 runs
#
# Usage:
#   bash scripts/experiments/alpha/run_proxy_unit_combo_lambda_sweep.sh
#
# Useful overrides:
#   USE_WANDB=1 bash ...
#   GPUS_OVERRIDE="0 1" bash ...
#   ENVS_OVERRIDE="beta_0.3 beta_0.1" bash ...

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
WARMUP="${WARMUP:-250}"
LOG_ROOT="${LOG_ROOT:-logs_proxy_unit_combo_lambda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ENVS=(${ENVS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})
METHODS=(
    "client|teacher_label_prob_entropy|map|none|client_label_prob_entropy_0_1"
    "round_client|teacher_label_prob|multiply|linear|round_client_label_prob_0_1_w${WARMUP}"
    "round_client|teacher_label_prob_entropy|multiply|linear|round_client_label_prob_entropy_0_1_w${WARMUP}"
)

WANDB_FLAGS=""
if [ "${USE_WANDB:-0}" = "1" ]; then
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
    local unit=$3
    local proxy=$4
    local mode=$5
    local schedule=$6
    local method_name=$7
    local log_dir="${LOG_ROOT}/${env_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local flags
    local schedule_flags=""

    flags="$(env_flags "$env_name")"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | ${method_name}"
        return
    fi

    if [ "$schedule" != "none" ]; then
        schedule_flags="--byot_round_lambda_schedule ${schedule} --byot_round_lambda_min 0.00 --byot_round_lambda_warmup ${WARMUP}"
    fi

    echo "[GPU ${gpu_id}] start: ${env_name} | ${method_name} | proxy=${proxy} | mode=${mode}"

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
        --byot_alpha 1.00 \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        --byot_client_proxy "${proxy}" \
        --byot_client_alpha_min 0.00 \
        --byot_client_alpha_max 1.00 \
        --byot_client_alpha_mode "${mode}" \
        ${schedule_flags} \
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
        IFS='|' read -r env_name unit proxy mode schedule method_name <<< "$job"
        run_job "$gpu_id" "$env_name" "$unit" "$proxy" "$mode" "$schedule" "$method_name"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for env_name in "${ENVS[@]}"; do
    for method_spec in "${METHODS[@]}"; do
        IFS='|' read -r unit proxy mode schedule method_name <<< "$method_spec"
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${env_name}|${unit}|${proxy}|${mode}|${schedule}|${method_name}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== Proxy/Unit Combo Lambda Sweep =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "envs=${ENVS[*]}"
echo "methods=${METHODS[*]}"
echo "feature_beta=${FEATURE_BETA}, temp=${TEMP_VAL}, warmup=${WARMUP}, log_root=${LOG_ROOT}"
echo "skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-0}"
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
echo "Proxy/unit combo lambda sweep complete (${job_count} jobs)"
