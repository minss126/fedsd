#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Complete mechanism suite for the C10-vs-C100 alpha reversal.
# Core conditions hold the BYOT architecture fixed.  The C100-only detach
# conditions are the causal control for whether branch losses act through the
# shared trunk.  All jobs log layer-wise client updates every DRIFT_INTERVAL.

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
        PYTHON_BIN="python3"
    fi
fi

SEEDS=(${SEEDS_OVERRIDE:-0 1})
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DRIFT_INTERVAL="${DRIFT_INTERVAL:-10}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_branch_supervision_causal_suite_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_CORE="${RUN_CORE:-1}"
RUN_DETACH="${RUN_DETACH:-1}"

run_job() {
    local gpu_id=$1 dataset=$2 partition_name=$3 variant=$4 branch_mode=$5 objective=$6 alpha=$7 beta=$8 seed=$9
    local setting="${dataset}_resnet18/${partition_name}/fedavg/seed${seed}"
    local name="${variant}"
    local log_dir="${LOG_ROOT}/${setting}"
    local result="${log_dir}/${name}.pkl"
    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$result" ]; then
        echo "[GPU ${gpu_id}] skip: ${setting} | ${name}"
        return
    fi

    local branch_flags=(--byot_beta "$beta" --byot_branch_gradient_mode "$branch_mode")
    if [ "$variant" = "off" ]; then
        branch_flags+=(--byot_active_branches none)
    else
        branch_flags+=(--byot_active_branches 1,2,3 --byot_branch_objective "$objective" --byot_alpha "$alpha")
    fi

    echo "[GPU ${gpu_id}] start: ${setting} | ${name}"
    "$PYTHON_BIN" main.py \
        --dataset "$dataset" --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$seed" \
        --device "cuda:${gpu_id}" \
        --logdir "$LOG_ROOT" --log_file_name "${setting}/${name}" \
        --model resnet18_byot --alg fedbyot \
        --partition noniid --beta 0.5 \
        --log_client_drift --log_layerwise_client_update_drift \
        --drift_log_interval "$DRIFT_INTERVAL" \
        "${branch_flags[@]}" \
        > "${log_dir}/${name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${setting} | ${name}"
}

declare -a JOBS=()
add_job() {
    local dataset=$1 variant=$2 branch_mode=$3 objective=$4 alpha=$5 beta=$6
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${dataset}|beta_0.5|${variant}|${branch_mode}|${objective}|${alpha}|${beta}|${seed}")
    done
}

add_core_jobs() {
    local dataset=$1
    add_job "$dataset" off attached blend 0.00 0.00
    add_job "$dataset" feature_only attached feature_only 0.00 "$FEATURE_BETA"
    add_job "$dataset" ce_feature attached blend 0.00 "$FEATURE_BETA"
    add_job "$dataset" kd_feature attached blend 1.00 "$FEATURE_BETA"
}

if [ "$RUN_CORE" = "1" ]; then
    # C100 is the primary claim, so it is queued before C10.  With round-robin
    # GPU assignment all C100 core conditions finish before the C10 controls.
    add_core_jobs cifar100
fi

if [ "$RUN_DETACH" = "1" ]; then
    # Feature loss is disabled here, so attached/detached differs only in the
    # path taken by branch CE or KD gradients into the shared trunk.
    add_job cifar100 ce_only_attached attached blend 0.00 0.00
    add_job cifar100 ce_only_detached detach_shared blend 0.00 0.00
    add_job cifar100 kd_only_attached attached blend 1.00 0.00
    add_job cifar100 kd_only_detached detach_shared blend 1.00 0.00
fi

if [ "$RUN_CORE" = "1" ]; then
    add_core_jobs cifar10
fi

echo "========== Branch Supervision Causal Suite =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seeds=${SEEDS[*]}"
echo "core=${RUN_CORE}, detach=${RUN_DETACH}, feature_beta=${FEATURE_BETA}, drift_interval=${DRIFT_INTERVAL}"
echo "log_root=${LOG_ROOT}, jobs=${#JOBS[@]}, skip_existing=${SKIP_EXISTING}"

run_queue() {
    local gpu_id=$1
    shift
    local job dataset partition variant branch_mode objective alpha beta seed
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset partition variant branch_mode objective alpha beta seed <<< "$job"
        run_job "$gpu_id" "$dataset" "$partition" "$variant" "$branch_mode" "$objective" "$alpha" "$beta" "$seed"
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
echo "Branch supervision causal suite complete (${#JOBS[@]} jobs)"
