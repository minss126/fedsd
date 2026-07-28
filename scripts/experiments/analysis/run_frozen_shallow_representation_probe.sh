#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Train a BYOT-shaped teacher with every branch-side loss disabled, then train
# fresh B1/B2/B3 probe heads on their corresponding frozen shallow prefixes.
# The central reference data are used solely for this post-hoc diagnostic.

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
        PYTHON_BIN="python3"
    fi
fi

SEEDS=(${SEEDS_OVERRIDE:-0 1})
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-128}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_LR="${PROBE_LR:-0.05}"
PROBE_SAMPLES_PER_CLASS="${PROBE_SAMPLES_PER_CLASS:-500}"
PROBE_BRANCHES="${PROBE_BRANCHES:-1,2,3}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_frozen_shallow_representation_probe_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

CASES=(
    "cifar10|10|beta_0.5|--partition noniid --beta 0.5"
    "cifar100|100|beta_0.5|--partition noniid --beta 0.5"
)

run_job() {
    local gpu_id=$1 dataset=$2 num_classes=$3 env_name=$4 env_flags=$5 seed=$6
    local setting="${dataset}_resnet18/${env_name}/fedavg/seed${seed}"
    local method_name="teacher_only_frozen_shallow_probe"
    local log_dir="${LOG_ROOT}/${setting}"
    local checkpoint="${log_dir}/${method_name}_final.pt"
    local output="${log_dir}/${method_name}_metrics.json"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" != "1" ] || [ ! -f "$checkpoint" ]; then
        echo "[GPU ${gpu_id}] train: ${setting} | teacher-only"
        "$PYTHON_BIN" main.py \
            --dataset "$dataset" --datadir ./data \
            --n_clients 100 --sample_fraction 0.1 \
            --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$seed" \
            --device "cuda:${gpu_id}" \
            --logdir "$LOG_ROOT" --log_file_name "${setting}/${method_name}" \
            --model resnet18_byot --alg fedbyot \
            --byot_active_branches none --byot_beta 0.0 \
            --save_final_ckpt \
            $env_flags \
            > "${log_dir}/${method_name}_terminal.log" 2>&1
    else
        echo "[GPU ${gpu_id}] train checkpoint exists: ${setting}"
    fi

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$output" ]; then
        echo "[GPU ${gpu_id}] probe exists: ${setting}"
        return
    fi

    echo "[GPU ${gpu_id}] frozen shallow-branch probe: ${setting}"
    "$PYTHON_BIN" scripts/experiments/analysis/frozen_shallow_representation_probe.py \
        --checkpoint "$checkpoint" --dataset "$dataset" --datadir ./data \
        --device "cuda:${gpu_id}" --batch_size "$PROBE_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --probe_epochs "$PROBE_EPOCHS" \
        --probe_lr "$PROBE_LR" --samples_per_class "$PROBE_SAMPLES_PER_CLASS" \
        --branches "$PROBE_BRANCHES" \
        --seed "$seed" --output "$output" \
        > "${log_dir}/${method_name}_probe_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${setting}"
}

run_queue() {
    local gpu_id=$1
    shift
    local job
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset num_classes env_name env_flags seed <<< "$job"
        run_job "$gpu_id" "$dataset" "$num_classes" "$env_name" "$env_flags" "$seed"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done
job_count=0
for case_spec in "${CASES[@]}"; do
    IFS='|' read -r dataset num_classes env_name env_flags <<< "$case_spec"
    for seed in "${SEEDS[@]}"; do
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${dataset}|${num_classes}|${env_name}|${env_flags}|${seed}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== Frozen Shallow-Representation Probe =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seeds=${SEEDS[*]}"
echo "cases=${CASES[*]}"
echo "probe=frozen B1/B2/B3 prefixes + fresh matching heads, branches=${PROBE_BRANCHES}, epochs=${PROBE_EPOCHS}, samples/class=${PROBE_SAMPLES_PER_CLASS}"
echo "log_root=${LOG_ROOT}, jobs=${job_count}, skip_existing=${SKIP_EXISTING}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done
wait "${pids[@]}"
echo "Frozen shallow-representation probe complete (${job_count} jobs)"
