#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then
        PYTHON_BIN="venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPU ids supplied. Set GPUS_OVERRIDE."
    exit 1
fi

BASE_CHECKPOINT="${BASE_CHECKPOINT:-logs/analysis/logs_iid_client_count_representation/cifar100_resnet18/iid/clients_20/fedavg/seed0/teacher_only_client_count_final.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/analysis/logs_local_data_size_internal_probe}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:-${OUTPUT_ROOT}/shared_round_start_probe.pt}"

# Primary controlled design: the number of optimizer steps is fixed, while the
# finite local dataset and hence its reuse frequency change with SAMPLE_SIZES.
SAMPLE_SIZES=(${SAMPLE_SIZES_OVERRIDE:-100 250 500 1000 2500})
SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
CLIENTS_PER_CONDITION="${CLIENTS_PER_CONDITION:-10}"
TRAIN_BUDGET="${TRAIN_BUDGET:-steps}"
LOCAL_STEPS="${LOCAL_STEPS:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-50}"
LOCAL_LR="${LOCAL_LR:-0.01}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_LR="${PROBE_LR:-0.1}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-5e-4}"
PROBE_SAMPLES_PER_CLASS="${PROBE_SAMPLES_PER_CLASS:-500}"
PROBE_SEED="${PROBE_SEED:-3407}"
FORCE_PROBE="${FORCE_PROBE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [ ! -f "$BASE_CHECKPOINT" ]; then
    echo "Missing BASE_CHECKPOINT: $BASE_CHECKPOINT"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "========== Within-client local-data-size diagnostic =========="
echo "gpus=${GPUS[*]}"
echo "base_checkpoint=${BASE_CHECKPOINT}"
echo "sample_sizes=${SAMPLE_SIZES[*]}, seeds=${SEEDS[*]}, clients/condition=${CLIENTS_PER_CONDITION}"
echo "training=teacher CE only, budget=${TRAIN_BUDGET}, steps=${LOCAL_STEPS}, epochs=${LOCAL_EPOCHS}"
echo "optimizer=SGD(lr=${LOCAL_LR}, momentum=${MOMENTUM}, weight_decay=${WEIGHT_DECAY})"
echo "reference=full official CIFAR-100 test set"
if [ "$NUM_GPUS" -ge 4 ]; then
    echo "estimated default wall time: about 8-15 minutes after data are available"
elif [ "$NUM_GPUS" -ge 2 ]; then
    echo "estimated default wall time: about 15-25 minutes after data are available"
else
    echo "estimated default wall time: about 25-45 minutes after data are available"
fi

if [ "$FORCE_PROBE" = "1" ] || [ ! -s "$PROBE_CHECKPOINT" ]; then
    echo "Preparing one frozen shared probe on GPU ${GPUS[0]}"
    "$PYTHON_BIN" scripts/experiments/analysis/local_data_size_internal_probe.py prepare \
        --global_checkpoint "$BASE_CHECKPOINT" \
        --probe_output "$PROBE_CHECKPOINT" \
        --datadir ./data --device "cuda:${GPUS[0]}" \
        --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --probe_epochs "$PROBE_EPOCHS" --probe_lr "$PROBE_LR" \
        --probe_weight_decay "$PROBE_WEIGHT_DECAY" \
        --probe_samples_per_class "$PROBE_SAMPLES_PER_CLASS" \
        --test_samples_per_class 0 --seed "$PROBE_SEED" \
        > "${OUTPUT_ROOT}/prepare_probe.log" 2>&1
else
    echo "Using existing shared probe: $PROBE_CHECKPOINT"
fi

run_job() {
    local gpu=$1
    local sample_size=$2
    local seed=$3
    local job_dir="${OUTPUT_ROOT}/sample_${sample_size}/seed_${seed}"
    local output="${job_dir}/metrics.json"
    local terminal="${job_dir}/terminal.log"
    mkdir -p "$job_dir"

    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$output" ]; then
        echo "[GPU ${gpu}] exists n=${sample_size}, seed=${seed}"
        return
    fi

    echo "[GPU ${gpu}] start n=${sample_size}, seed=${seed}"
    "$PYTHON_BIN" scripts/experiments/analysis/local_data_size_internal_probe.py run \
        --global_checkpoint "$BASE_CHECKPOINT" \
        --probe_checkpoint "$PROBE_CHECKPOINT" \
        --output "$output" \
        --sample_size "$sample_size" --clients "$CLIENTS_PER_CONDITION" \
        --train_budget "$TRAIN_BUDGET" --local_steps "$LOCAL_STEPS" \
        --local_epochs "$LOCAL_EPOCHS" --local_batch_size "$LOCAL_BATCH_SIZE" \
        --lr "$LOCAL_LR" --momentum "$MOMENTUM" --weight_decay "$WEIGHT_DECAY" \
        --datadir ./data --device "cuda:${gpu}" \
        --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --test_samples_per_class 0 --seed "$seed" \
        > "$terminal" 2>&1
    echo "[GPU ${gpu}] done n=${sample_size}, seed=${seed}"
}

run_queue() {
    local gpu=$1
    shift
    local job
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r sample_size seed <<< "$job"
        run_job "$gpu" "$sample_size" "$seed"
    done
}

declare -a QUEUES
for ((index = 0; index < NUM_GPUS; index++)); do
    QUEUES[$index]=""
done

job_count=0
for sample_size in "${SAMPLE_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu_index=$((job_count % NUM_GPUS))
        QUEUES[$gpu_index]+="${sample_size}|${seed}"$'\n'
        job_count=$((job_count + 1))
    done
done

pids=()
for ((index = 0; index < NUM_GPUS; index++)); do
    if [ -n "${QUEUES[$index]}" ]; then
        mapfile -t jobs <<< "${QUEUES[$index]}"
        run_queue "${GPUS[$index]}" "${jobs[@]}" &
        pids+=("$!")
    fi
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

"$PYTHON_BIN" scripts/experiments/analysis/summarize_local_data_size_internal_probe.py \
    --input_root "$OUTPUT_ROOT" \
    --output_json "${OUTPUT_ROOT}/summary.json" \
    --output_csv "${OUTPUT_ROOT}/summary.csv"

echo "Completed ${job_count} jobs. Summary: ${OUTPUT_ROOT}/summary.json"
