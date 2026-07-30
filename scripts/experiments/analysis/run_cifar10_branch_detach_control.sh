#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# C10 counterpart of the C100 attached/detached control.
#
# Branch CE/KD always trains its private branch heads.  `detach_shared` cuts
# only the branch-loss gradient into the shared trunk.  Feature imitation is
# disabled so attached vs detached differs solely in that gradient path.

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPU ids provided. Set GPUS_OVERRIDE." >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then PYTHON_BIN="venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

SEEDS=(${SEEDS_OVERRIDE:-0 1})
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DRIFT_INTERVAL="${DRIFT_INTERVAL:-10}"
BRANCH_GRADIENT_INTERVAL="${BRANCH_GRADIENT_INTERVAL:-50}"
BRANCH_GRADIENT_BATCHES="${BRANCH_GRADIENT_BATCHES:-1}"
REPRESENTATION_INTERVAL="${REPRESENTATION_INTERVAL:-50}"
REPRESENTATION_BATCHES="${REPRESENTATION_BATCHES:-8}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_cifar10_branch_detach_control_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

run_job() {
    local gpu_id=$1 variant=$2 branch_mode=$3 objective=$4 alpha=$5 seed=$6
    local setting="cifar10_resnet18/beta_0.5/fedavg/seed${seed}"
    local log_dir="${LOG_ROOT}/${setting}"
    local result="${log_dir}/${variant}.pkl"
    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$result" ]; then
        echo "[GPU ${gpu_id}] skip: ${setting} | ${variant}"
        return
    fi

    local branch_flags=(--byot_beta 0.00 --byot_branch_gradient_mode "$branch_mode")
    if [ "$variant" = "off" ]; then
        branch_flags+=(--byot_active_branches none)
    else
        branch_flags+=(
            --byot_active_branches 1,2,3
            --byot_branch_objective "$objective"
            --byot_alpha "$alpha"
        )
    fi

    echo "[GPU ${gpu_id}] start: ${setting} | ${variant}"
    "$PYTHON_BIN" main.py \
        --dataset cifar10 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$seed" \
        --device "cuda:${gpu_id}" \
        --logdir "$LOG_ROOT" --log_file_name "${setting}/${variant}" \
        --model resnet18_byot --alg fedbyot \
        --partition noniid --beta 0.5 \
        --log_client_drift --log_layerwise_client_update_drift \
        --drift_log_interval "$DRIFT_INTERVAL" \
        --log_branch_shared_gradient_dispersion \
        --branch_shared_gradient_probe_interval "$BRANCH_GRADIENT_INTERVAL" \
        --branch_shared_gradient_probe_batches "$BRANCH_GRADIENT_BATCHES" \
        --log_post_aggregation_representation \
        --representation_probe_interval "$REPRESENTATION_INTERVAL" \
        --representation_probe_batches "$REPRESENTATION_BATCHES" \
        "${branch_flags[@]}" \
        > "${log_dir}/${variant}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${setting} | ${variant}"
}

declare -a JOBS=()
add_variant() {
    local variant=$1 branch_mode=$2 objective=$3 alpha=$4
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${variant}|${branch_mode}|${objective}|${alpha}|${seed}")
    done
}

# Self-contained baseline plus the exact four-way causal control.
add_variant off attached blend 0.00
add_variant ce_only_attached attached blend 0.00
add_variant ce_only_detached detach_shared blend 0.00
add_variant kd_only_attached attached blend 1.00
add_variant kd_only_detached detach_shared blend 1.00

echo "========== CIFAR-10 Branch Detach Control =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seeds=${SEEDS[*]}"
echo "partition=beta_0.5, feature_beta=0.00"
echo "conditions=off,ce_attached,ce_detached,kd_attached,kd_detached"
echo "shared_gradient_probe=every ${BRANCH_GRADIENT_INTERVAL} rounds, ${BRANCH_GRADIENT_BATCHES} batch/client"
echo "representation_probe=every ${REPRESENTATION_INTERVAL} rounds, ${REPRESENTATION_BATCHES} common-reference test batches"
echo "log_root=${LOG_ROOT}, jobs=${#JOBS[@]}, skip_existing=${SKIP_EXISTING}"

run_queue() {
    local gpu_id=$1
    shift
    local job variant branch_mode objective alpha seed
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r variant branch_mode objective alpha seed <<< "$job"
        run_job "$gpu_id" "$variant" "$branch_mode" "$objective" "$alpha" "$seed"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done
for ((i = 0; i < ${#JOBS[@]}; i++)); do
    QUEUES[$((i % NUM_GPUS))]+="${JOBS[$i]}"$'\n'
done

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done
wait "${pids[@]}"
echo "CIFAR-10 branch detach control complete (${#JOBS[@]} jobs)"
