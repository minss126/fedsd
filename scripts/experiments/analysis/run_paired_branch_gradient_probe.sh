#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Direct gradient-mechanism suite.  At every probe round, the same round-start
# model, same client, and same local batch yield all three gradients:
# teacher CE, branch CE, and branch KD.  Thus CE/KD cosine comparisons do not
# confound different checkpoints or different client batches.

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
FEATURE_BETA="${FEATURE_BETA:-0.01}"
PROBE_INTERVAL="${PROBE_INTERVAL:-50}"
PROBE_BATCHES="${PROBE_BATCHES:-1}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_paired_branch_gradient_probe_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

run_job() {
    local gpu_id=$1 dataset=$2 variant=$3 objective=$4 alpha=$5 seed=$6
    local setting="${dataset}_resnet18/beta_0.5/fedavg/seed${seed}"
    local log_dir="${LOG_ROOT}/${setting}"
    local result="${log_dir}/${variant}.pkl"
    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$result" ]; then
        echo "[GPU ${gpu_id}] skip: ${setting} | ${variant}"
        return
    fi

    echo "[GPU ${gpu_id}] start: ${setting} | ${variant}"
    "$PYTHON_BIN" main.py \
        --dataset "$dataset" --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$seed" \
        --device "cuda:${gpu_id}" \
        --logdir "$LOG_ROOT" --log_file_name "${setting}/${variant}" \
        --model resnet18_byot --alg fedbyot \
        --partition noniid --beta 0.5 \
        --byot_active_branches 1,2,3 \
        --byot_branch_objective "$objective" --byot_alpha "$alpha" \
        --byot_beta "$FEATURE_BETA" \
        --log_branch_gradient_alignment \
        --branch_gradient_probe_interval "$PROBE_INTERVAL" \
        --branch_gradient_probe_batches "$PROBE_BATCHES" \
        > "${log_dir}/${variant}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${setting} | ${variant}"
}

declare -a JOBS=()
add_variant() {
    local dataset=$1 variant=$2 objective=$3 alpha=$4
    for seed in "${SEEDS[@]}"; do JOBS+=("${dataset}|${variant}|${objective}|${alpha}|${seed}"); done
}

for dataset in cifar100 cifar10; do
    # A neutral reference plus models trained under branch CE and branch KD.
    # The probe itself always computes both hypothetical branch gradients.
    add_variant "$dataset" feature_only feature_only 0.00
    add_variant "$dataset" ce_feature blend 0.00
    add_variant "$dataset" kd_feature blend 1.00
done

echo "========== Paired Branch Gradient Probe =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seeds=${SEEDS[*]}"
echo "datasets=cifar100,cifar10 | partition=beta_0.5 | feature_beta=${FEATURE_BETA}"
echo "models=feature_only,ce_feature,kd_feature | probe every ${PROBE_INTERVAL} rounds"
echo "log_root=${LOG_ROOT}, jobs=${#JOBS[@]}, skip_existing=${SKIP_EXISTING}"

run_queue() {
    local gpu_id=$1; shift
    local job dataset variant objective alpha seed
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset variant objective alpha seed <<< "$job"
        run_job "$gpu_id" "$dataset" "$variant" "$objective" "$alpha" "$seed"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done
for ((i = 0; i < ${#JOBS[@]}; i++)); do QUEUES[$((i % NUM_GPUS))]+="${JOBS[$i]}"$'\n'; done

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done
wait "${pids[@]}"
echo "Paired branch gradient probe complete (${#JOBS[@]} jobs)"
