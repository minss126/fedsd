#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Post-hoc diagnostic for checkpoints trained with final CE only.  The FL
# checkpoints are never modified or retrained.  At B1/B2/B3/final, raw trunk
# features are GAP-pooled and decoded by fresh linear classifiers.

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

DATASETS=(${DATASETS_OVERRIDE:-cifar10 cifar100})
SEEDS=(${SEEDS_OVERRIDE:-0 1})
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-logs/analysis/logs_frozen_shallow_representation_probe_r500}"
CHECKPOINT_PARTITION="${CHECKPOINT_PARTITION:-beta_0.5}"
CHECKPOINT_METHOD="${CHECKPOINT_METHOD:-teacher_only_frozen_shallow_probe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/analysis/logs_strict_linear_representation_probe_r500}"

PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-512}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_LR="${PROBE_LR:-0.1}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-5e-4}"
PROBE_SAMPLES_PER_CLASS="${PROBE_SAMPLES_PER_CLASS:-0}"
TEST_SAMPLES_PER_CLASS="${TEST_SAMPLES_PER_CLASS:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

checkpoint_path() {
    local dataset=$1 seed=$2
    printf '%s/%s_resnet18/%s/fedavg/seed%s/%s_final.pt\n' \
        "$CHECKPOINT_ROOT" "$dataset" "$CHECKPOINT_PARTITION" "$seed" \
        "$CHECKPOINT_METHOD"
}

output_dir() {
    local dataset=$1 seed=$2
    printf '%s/%s_resnet18/%s/fedavg/seed%s\n' \
        "$OUTPUT_ROOT" "$dataset" "$CHECKPOINT_PARTITION" "$seed"
}

# Validate all sources before starting so a typo cannot leave a partial sweep.
missing=0
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        checkpoint="$(checkpoint_path "$dataset" "$seed")"
        if [ ! -s "$checkpoint" ]; then
            echo "Missing plain/final-CE checkpoint: $checkpoint" >&2
            missing=1
        fi
    done
done
if [ "$missing" -ne 0 ]; then
    exit 1
fi

run_job() {
    local gpu=$1 dataset=$2 seed=$3
    local checkpoint run_dir output terminal
    checkpoint="$(checkpoint_path "$dataset" "$seed")"
    run_dir="$(output_dir "$dataset" "$seed")"
    output="${run_dir}/strict_linear_probe_metrics.json"
    terminal="${run_dir}/strict_linear_probe_terminal.log"
    mkdir -p "$run_dir"

    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$output" ]; then
        echo "[GPU ${gpu}] exists: ${dataset}, seed=${seed}"
        return
    fi

    echo "[GPU ${gpu}] start: ${dataset}, seed=${seed}"
    "$PYTHON_BIN" scripts/experiments/analysis/strict_linear_representation_probe.py \
        --checkpoint "$checkpoint" --dataset "$dataset" --datadir ./data \
        --device "cuda:${gpu}" --batch_size "$PROBE_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --probe_epochs "$PROBE_EPOCHS" \
        --probe_lr "$PROBE_LR" --probe_weight_decay "$PROBE_WEIGHT_DECAY" \
        --probe_samples_per_class "$PROBE_SAMPLES_PER_CLASS" \
        --test_samples_per_class "$TEST_SAMPLES_PER_CLASS" \
        --seed "$seed" --output "$output" \
        > "$terminal" 2>&1
    echo "[GPU ${gpu}] complete: ${dataset}, seed=${seed}"
}

run_queue() {
    local gpu=$1
    shift
    local job dataset seed
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset seed <<< "$job"
        run_job "$gpu" "$dataset" "$seed"
    done
}

declare -a QUEUES
for ((index = 0; index < NUM_GPUS; index++)); do
    QUEUES[$index]=""
done

job_count=0
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu_index=$((job_count % NUM_GPUS))
        QUEUES[$gpu_index]+="${dataset}|${seed}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== Strict linear representation probe =========="
echo "gpus=${GPUS[*]}"
echo "datasets=${DATASETS[*]}, seeds=${SEEDS[*]}, jobs=${job_count}"
echo "checkpoint=existing plain/final-CE FL model (no FL retraining)"
echo "probe=raw B1/B2/B3/Final -> GAP -> fresh Linear only"
echo "probe train=official full train set without augmentation"
echo "evaluation=official full test set"
echo "epochs=${PROBE_EPOCHS}, batch=${PROBE_BATCH_SIZE}, lr=${PROBE_LR}"
echo "output_root=${OUTPUT_ROOT}"
echo "estimated 4-GPU wall time=about 2-5 minutes when GPUs are idle"

pids=()
for ((index = 0; index < NUM_GPUS; index++)); do
    if [ -n "${QUEUES[$index]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$index]}"
        run_queue "${GPUS[$index]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if [ "$failed" -ne 0 ]; then
    echo "At least one strict-linear probe failed; inspect *_terminal.log." >&2
    exit 1
fi

"$PYTHON_BIN" scripts/experiments/analysis/summarize_strict_linear_representation_probes.py \
    --input_root "$OUTPUT_ROOT" --output_dir "$OUTPUT_ROOT"

echo "Strict linear representation probe complete (${job_count} jobs)."
echo "Summary: ${OUTPUT_ROOT}/summary.md"
