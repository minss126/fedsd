#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then PYTHON_BIN="venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-4}"
if [ "$NUM_GPUS" -ne "$EXPECTED_GPU_COUNT" ]; then
    echo "Exactly ${EXPECTED_GPU_COUNT} GPU ids are required; got: ${GPUS[*]}" >&2
    exit 1
fi

DATASETS=(${DATASETS_OVERRIDE:-cifar10 cifar100})
SAMPLE_SIZES=(${SAMPLE_SIZES_OVERRIDE:-100 250 500 1000 2500})
LOCAL_SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
CHECKPOINT_SEED="${CHECKPOINT_SEED:-0}"
CLIENTS_PER_CONDITION="${CLIENTS_PER_CONDITION:-10}"

# K=20, 100 rounds, 5 local epochs processes 25M samples in total.  On the
# 50k-image CIFAR training split this is 500 centralized epochs.
CENTRAL_EPOCHS="${CENTRAL_EPOCHS:-500}"
CENTRAL_BATCH_SIZE="${CENTRAL_BATCH_SIZE:-50}"
CENTRAL_LR="${CENTRAL_LR:-0.1}"
CENTRAL_ETA_MIN="${CENTRAL_ETA_MIN:-0.0}"
CENTRAL_MOMENTUM="${CENTRAL_MOMENTUM:-0.9}"
CENTRAL_WEIGHT_DECAY="${CENTRAL_WEIGHT_DECAY:-0.001}"
CENTRAL_EVAL_INTERVAL="${CENTRAL_EVAL_INTERVAL:-50}"

LOCAL_STEPS="${LOCAL_STEPS:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-50}"
LOCAL_LR="${LOCAL_LR:-0.01}"
LOCAL_MOMENTUM="${LOCAL_MOMENTUM:-0.9}"
LOCAL_WEIGHT_DECAY="${LOCAL_WEIGHT_DECAY:-1e-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_LR="${PROBE_LR:-0.1}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-5e-4}"
PROBE_SAMPLES_PER_CLASS="${PROBE_SAMPLES_PER_CLASS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_PROBES="${FORCE_PROBES:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-logs/analysis/logs_centralized_checkpoint_source}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/analysis/logs_centralized_checkpoint_local_n}"

checkpoint_path() {
    local dataset=$1
    printf '%s/%s/seed_%s/central_teacher_epoch%04d.pt' \
        "$CHECKPOINT_ROOT" "$dataset" "$CHECKPOINT_SEED" "$CENTRAL_EPOCHS"
}

probe_path() {
    local dataset=$1
    printf '%s/%s/shared_logit_cka_probe.pt' "$OUTPUT_ROOT" "$dataset"
}

echo "========== Centralized-checkpoint local-n motivation =========="
echo "gpus=${GPUS[*]}"
echo "datasets=${DATASETS[*]}, centralized checkpoint seed=${CHECKPOINT_SEED}"
echo "central training=full train, teacher CE only, ${CENTRAL_EPOCHS} epochs, cosine LR"
echo "local sample sizes=${SAMPLE_SIZES[*]}, sampling seeds=${LOCAL_SEEDS[*]}, forks/condition=${CLIENTS_PER_CONDITION}"
echo "local budgets=fixed_step(${LOCAL_STEPS}) + fixed_epoch(${LOCAL_EPOCHS})"
echo "metrics=frozen-probe logits + cross-depth linear CKA on full official test set"
echo "data relation=central checkpoint and local forks reuse the official train split"
if [ "$NUM_GPUS" -eq 2 ]; then
    echo "estimated first-run 2-GPU wall time: about 3-7 hours at 500 central epochs"
    echo "estimated checkpoint-reuse wall time: about 35-80 minutes"
else
    echo "estimated first-run ${NUM_GPUS}-GPU wall time: about 3-6 hours at 500 central epochs"
    echo "estimated checkpoint-reuse wall time: about 20-45 minutes"
fi

train_central_checkpoint() {
    local gpu=$1 dataset=$2 checkpoint terminal
    checkpoint="$(checkpoint_path "$dataset")"
    terminal="${checkpoint%.pt}_terminal.log"
    mkdir -p "$(dirname "$checkpoint")"
    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$checkpoint" ]; then
        echo "[GPU ${gpu}] centralized checkpoint exists: ${dataset}"
        return
    fi
    echo "[GPU ${gpu}] train centralized checkpoint: ${dataset}"
    "$PYTHON_BIN" scripts/experiments/analysis/train_centralized_teacher_checkpoint.py \
        --dataset "$dataset" --datadir ./data --output "$checkpoint" \
        --device "cuda:${gpu}" --epochs "$CENTRAL_EPOCHS" \
        --batch_size "$CENTRAL_BATCH_SIZE" --test_batch_size 512 \
        --lr "$CENTRAL_LR" --eta_min "$CENTRAL_ETA_MIN" \
        --momentum "$CENTRAL_MOMENTUM" --weight_decay "$CENTRAL_WEIGHT_DECAY" \
        --num_workers "$NUM_WORKERS" --seed "$CHECKPOINT_SEED" \
        --eval_interval "$CENTRAL_EVAL_INTERVAL" \
        > "$terminal" 2>&1
    echo "[GPU ${gpu}] completed centralized checkpoint: ${dataset}"
}

central_pids=()
for index in "${!DATASETS[@]}"; do
    gpu_index=$((index % NUM_GPUS))
    train_central_checkpoint "${GPUS[$gpu_index]}" "${DATASETS[$index]}" &
    central_pids+=("$!")
done
for pid in "${central_pids[@]}"; do wait "$pid"; done

prepare_probe() {
    local gpu=$1 dataset=$2 checkpoint probe probe_dir
    checkpoint="$(checkpoint_path "$dataset")"
    probe="$(probe_path "$dataset")"
    probe_dir="$(dirname "$probe")"
    mkdir -p "$probe_dir"
    if [ "$FORCE_PROBES" != "1" ] && [ -s "$probe" ]; then
        echo "[GPU ${gpu}] probe exists: ${dataset}"
        return
    fi
    echo "[GPU ${gpu}] fit centralized frozen probes: ${dataset}"
    "$PYTHON_BIN" scripts/experiments/analysis/local_data_size_internal_probe.py prepare \
        --dataset "$dataset" --metrics logits_cka \
        --global_checkpoint "$checkpoint" --probe_output "$probe" \
        --datadir ./data --device "cuda:${gpu}" \
        --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --probe_epochs "$PROBE_EPOCHS" --probe_lr "$PROBE_LR" \
        --probe_weight_decay "$PROBE_WEIGHT_DECAY" \
        --probe_samples_per_class "$PROBE_SAMPLES_PER_CLASS" \
        --test_samples_per_class 0 --seed 3407 \
        > "${probe_dir}/prepare_probe.log" 2>&1
}

probe_pids=()
for index in "${!DATASETS[@]}"; do
    gpu_index=$((index % NUM_GPUS))
    prepare_probe "${GPUS[$gpu_index]}" "${DATASETS[$index]}" &
    probe_pids+=("$!")
done
for pid in "${probe_pids[@]}"; do wait "$pid"; done

run_job() {
    local gpu=$1 dataset=$2 budget=$3 sample_size=$4 seed=$5 train_budget
    local job_dir="${OUTPUT_ROOT}/${dataset}/${budget}/sample_${sample_size}/seed_${seed}"
    local output="${job_dir}/metrics.json"
    mkdir -p "$job_dir"
    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$output" ] && grep -q '"metrics": "logits_cka"' "$output"; then
        echo "[GPU ${gpu}] exists ${dataset} ${budget} n=${sample_size} seed=${seed}"
        return
    fi
    train_budget=steps
    if [ "$budget" = "fixed_epoch" ]; then train_budget=epochs; fi
    echo "[GPU ${gpu}] start ${dataset} ${budget} n=${sample_size} seed=${seed}"
    "$PYTHON_BIN" scripts/experiments/analysis/local_data_size_internal_probe.py run \
        --dataset "$dataset" --metrics logits_cka \
        --global_checkpoint "$(checkpoint_path "$dataset")" \
        --probe_checkpoint "$(probe_path "$dataset")" --output "$output" \
        --sample_size "$sample_size" --clients "$CLIENTS_PER_CONDITION" \
        --train_budget "$train_budget" --local_steps "$LOCAL_STEPS" \
        --local_epochs "$LOCAL_EPOCHS" --local_batch_size "$LOCAL_BATCH_SIZE" \
        --lr "$LOCAL_LR" --momentum "$LOCAL_MOMENTUM" \
        --weight_decay "$LOCAL_WEIGHT_DECAY" \
        --datadir ./data --device "cuda:${gpu}" \
        --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --test_samples_per_class 0 --seed "$seed" \
        > "${job_dir}/terminal.log" 2>&1
    echo "[GPU ${gpu}] done ${dataset} ${budget} n=${sample_size} seed=${seed}"
}

run_queue() {
    local gpu=$1
    shift
    local job dataset budget sample_size seed
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset budget sample_size seed <<< "$job"
        run_job "$gpu" "$dataset" "$budget" "$sample_size" "$seed"
    done
}

declare -a QUEUES
for ((index = 0; index < NUM_GPUS; index++)); do QUEUES[$index]=""; done
job_count=0
condition_count=0
mapfile -t ORDERED_SAMPLE_SIZES < <(printf '%s\n' "${SAMPLE_SIZES[@]}" | sort -nr)
for sample_size in "${ORDERED_SAMPLE_SIZES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for seed in "${LOCAL_SEEDS[@]}"; do
            epoch_gpu=$((condition_count % NUM_GPUS))
            step_gpu=$(((condition_count + 2) % NUM_GPUS))
            QUEUES[$epoch_gpu]+="${dataset}|fixed_epoch|${sample_size}|${seed}"$'\n'
            QUEUES[$step_gpu]+="${dataset}|fixed_step|${sample_size}|${seed}"$'\n'
            condition_count=$((condition_count + 1))
            job_count=$((job_count + 2))
        done
    done
done

job_pids=()
for ((index = 0; index < NUM_GPUS; index++)); do
    mapfile -t jobs <<< "${QUEUES[$index]}"
    run_queue "${GPUS[$index]}" "${jobs[@]}" &
    job_pids+=("$!")
done
status=0
for pid in "${job_pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
if [ "$status" -ne 0 ]; then
    echo "At least one centralized-checkpoint diagnostic failed." >&2
    exit "$status"
fi

for dataset in "${DATASETS[@]}"; do
    for budget in fixed_step fixed_epoch; do
        "$PYTHON_BIN" scripts/experiments/analysis/summarize_local_data_size_internal_probe.py \
            --input_root "${OUTPUT_ROOT}/${dataset}/${budget}" \
            --output_json "${OUTPUT_ROOT}/${dataset}/${budget}/summary.json" \
            --output_csv "${OUTPUT_ROOT}/${dataset}/${budget}/summary.csv"
    done
done

echo "Completed ${job_count} centralized-checkpoint jobs under ${OUTPUT_ROOT}"
