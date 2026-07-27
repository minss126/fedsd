#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Controlled complexity study: CIFAR-100 image domain, transforms, model, FL
# protocol, beta, and 500 samples/original class are held fixed.  The class
# subsets are nested under one permutation.  Total data necessarily grows with
# class count; this therefore tests class-cardinality under fixed per-class
# evidence, not class count in isolation from total-data budget.

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

CLASS_COUNTS=(${CLASS_COUNTS_OVERRIDE:-10 20 50 100})
ALPHAS=(${ALPHAS_OVERRIDE:-0.00 1.00})
SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
N_CLIENTS="${N_CLIENTS:-50}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-32}"
SUBSET_SEED="${SUBSET_SEED:-0}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_cifar100_class_count_alpha_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

run_job() {
    local gpu_id=$1 class_count=$2 alpha=$3 seed=$4
    local alpha_tag="${alpha/./p}"
    local setting="cifar100k${class_count}_resnet18/beta_0.5/fedavg/seed${seed}"
    local name="alpha${alpha_tag}"
    local log_dir="${LOG_ROOT}/${setting}"
    local result="${log_dir}/${name}.pkl"
    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$result" ]; then
        echo "[GPU ${gpu_id}] skip: ${setting} | ${name}"
        return
    fi
    echo "[GPU ${gpu_id}] start: ${setting} | ${name}"
    "$PYTHON_BIN" main.py \
        --dataset cifar100 --datadir ./data \
        --cifar100_class_count "$class_count" --cifar100_subset_seed "$SUBSET_SEED" \
        --n_clients "$N_CLIENTS" --sample_fraction "$SAMPLE_FRACTION" --min_require_size "$MIN_REQUIRE_SIZE" \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$seed" \
        --device "cuda:${gpu_id}" \
        --logdir "$LOG_ROOT" --log_file_name "${setting}/${name}" \
        --model resnet18_byot --alg fedbyot \
        --partition noniid --beta 0.5 \
        --byot_active_branches 1,2,3 --byot_branch_objective blend \
        --byot_alpha "$alpha" --byot_beta 0.01 \
        > "${log_dir}/${name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${setting} | ${name}"
}

declare -a JOBS=()
for class_count in "${CLASS_COUNTS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            JOBS+=("${class_count}|${alpha}|${seed}")
        done
    done
done

echo "========== CIFAR-100 Class-Count Alpha Control =========="
echo "gpus=${GPUS[*]}, class_counts=${CLASS_COUNTS[*]}, alphas=${ALPHAS[*]}, seeds=${SEEDS[*]}"
echo "n_clients=${N_CLIENTS}, fixed_per_class=500, beta=0.5, log_root=${LOG_ROOT}, jobs=${#JOBS[@]}"

run_queue() {
    local gpu_id=$1
    shift
    local job class_count alpha seed
    for job in "$@"; do
        IFS='|' read -r class_count alpha seed <<< "$job"
        run_job "$gpu_id" "$class_count" "$alpha" "$seed"
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
echo "CIFAR-100 class-count alpha control complete (${#JOBS[@]} jobs)"
