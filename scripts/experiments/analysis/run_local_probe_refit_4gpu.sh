#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then PYTHON_BIN="venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
GPU_COUNT="${#GPUS[@]}"
DATASETS=(${DATASETS_OVERRIDE:-cifar10 cifar100})
ROUNDS=(${ROUNDS_OVERRIDE:-100})
BUDGETS=(${BUDGETS_OVERRIDE:-fixed_step fixed_epoch})
SAMPLE_SIZES=(${SAMPLE_SIZES_OVERRIDE:-100 250 500 1000 2500})
SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
CLIENTS_PER_CONDITION="${CLIENTS_PER_CONDITION:-10}"

if [ "$GPU_COUNT" -lt 1 ]; then
    echo "At least one GPU id is required." >&2
    exit 1
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-logs/analysis/logs_independent_round_budget_source}"
GLOBAL_PROBE_ROOT="${GLOBAL_PROBE_ROOT:-logs/analysis/logs_independent_round_budget_logit_cka}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/analysis/logs_local_probe_refit}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-logs/analysis/local_probe_refit_analysis}"

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
PROBE_SAMPLES_PER_CLASS="${PROBE_SAMPLES_PER_CLASS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

checkpoint_path() {
    local dataset=$1 completed_rounds=$2
    local source_budget=$completed_rounds
    # Match the independent-budget experiment's shared initialization control.
    if [ "$completed_rounds" -eq 0 ]; then source_budget=10; fi
    printf '%s/%s_resnet18/iid/clients_20/fedavg/seed0/budget_r%04d/teacher_only_independent_endpoint_round%04d.pt' \
        "$CHECKPOINT_ROOT" "$dataset" "$source_budget" "$completed_rounds"
}

probe_path() {
    local dataset=$1 completed_rounds=$2
    printf '%s/%s/round_%04d/shared_logit_cka_probe.pt' \
        "$GLOBAL_PROBE_ROOT" "$dataset" "$completed_rounds"
}

echo "========== Local-probe refit diagnostic =========="
echo "gpus=${GPUS[*]}"
echo "datasets=${DATASETS[*]}, independent endpoint rounds=${ROUNDS[*]}"
echo "budgets=${BUDGETS[*]}, local sample sizes=${SAMPLE_SIZES[*]}"
echo "seeds=${SEEDS[*]}, local forks/condition=${CLIENTS_PER_CONDITION}"
echo "probe=one B1/B2/B3/Final head refitted per local fork"
echo "probe train=full official train set; evaluation=full official test set"
echo "primary comparison=global frozen accuracy vs local-refit accuracy"
if [ "$GPU_COUNT" -eq 2 ]; then
    echo "estimated R=100-only 2-GPU wall time: about 4-6 hours"
elif [ "$GPU_COUNT" -eq 4 ]; then
    echo "estimated R=100-only 4-GPU wall time: about 2-3 hours"
else
    echo "estimated time scales approximately inversely with ${GPU_COUNT} GPUs"
fi

for dataset in "${DATASETS[@]}"; do
    for completed_rounds in "${ROUNDS[@]}"; do
        checkpoint="$(checkpoint_path "$dataset" "$completed_rounds")"
        probe="$(probe_path "$dataset" "$completed_rounds")"
        if [ ! -s "$checkpoint" ]; then
            echo "Missing independent endpoint checkpoint: $checkpoint" >&2
            exit 1
        fi
        if [ ! -s "$probe" ]; then
            echo "Missing global frozen probe: $probe" >&2
            echo "Finish run_independent_round_budget_logit_cka_4gpu.sh first." >&2
            exit 1
        fi
    done
done

run_job() {
    local gpu=$1 dataset=$2 completed_rounds=$3 budget=$4 sample_size=$5 seed=$6
    local job_dir output train_budget
    job_dir="${OUTPUT_ROOT}/${dataset}/round_$(printf '%04d' "$completed_rounds")/${budget}/sample_${sample_size}/seed_${seed}"
    output="${job_dir}/metrics.json"
    mkdir -p "$job_dir"
    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$output" ] && grep -q 'local_probe_refit_diagnostic' "$output"; then
        echo "[GPU ${gpu}] exists ${dataset} R=${completed_rounds} ${budget} n=${sample_size} seed=${seed}"
        return
    fi
    train_budget=steps
    if [ "$budget" = "fixed_epoch" ]; then train_budget=epochs; fi
    echo "[GPU ${gpu}] start ${dataset} R=${completed_rounds} ${budget} n=${sample_size} seed=${seed}"
    "$PYTHON_BIN" scripts/experiments/analysis/local_probe_refit_diagnostic.py \
        --dataset "$dataset" \
        --global_checkpoint "$(checkpoint_path "$dataset" "$completed_rounds")" \
        --global_probe_checkpoint "$(probe_path "$dataset" "$completed_rounds")" \
        --allow_relocated_global_checkpoint \
        --output "$output" --sample_size "$sample_size" \
        --clients "$CLIENTS_PER_CONDITION" --train_budget "$train_budget" \
        --local_steps "$LOCAL_STEPS" --local_epochs "$LOCAL_EPOCHS" \
        --local_batch_size "$LOCAL_BATCH_SIZE" --lr "$LOCAL_LR" \
        --momentum "$MOMENTUM" --weight_decay "$WEIGHT_DECAY" \
        --probe_epochs "$PROBE_EPOCHS" --probe_lr "$PROBE_LR" \
        --probe_weight_decay "$PROBE_WEIGHT_DECAY" \
        --probe_samples_per_class "$PROBE_SAMPLES_PER_CLASS" \
        --test_samples_per_class 0 --datadir ./data \
        --device "cuda:${gpu}" --batch_size "$EVAL_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --seed "$seed" \
        > "${job_dir}/terminal.log" 2>&1
    echo "[GPU ${gpu}] done ${dataset} R=${completed_rounds} ${budget} n=${sample_size} seed=${seed}"
}

run_queue() {
    local gpu=$1
    shift
    local job dataset completed_rounds budget sample_size seed
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset completed_rounds budget sample_size seed <<< "$job"
        run_job "$gpu" "$dataset" "$completed_rounds" "$budget" "$sample_size" "$seed"
    done
}

declare -a QUEUES
for ((index = 0; index < GPU_COUNT; index++)); do QUEUES[$index]=""; done
job_count=0
mapfile -t DESC_SIZES < <(printf '%s\n' "${SAMPLE_SIZES[@]}" | sort -nr)
for sample_size in "${DESC_SIZES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for completed_rounds in "${ROUNDS[@]}"; do
            for budget in "${BUDGETS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    index=$((job_count % GPU_COUNT))
                    QUEUES[$index]+="${dataset}|${completed_rounds}|${budget}|${sample_size}|${seed}"$'\n'
                    job_count=$((job_count + 1))
                done
            done
        done
    done
done

pids=()
for ((index = 0; index < GPU_COUNT; index++)); do
    mapfile -t jobs <<< "${QUEUES[$index]}"
    run_queue "${GPUS[$index]}" "${jobs[@]}" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
if [ "$status" -ne 0 ]; then
    echo "At least one local-probe refit queue failed." >&2
    exit "$status"
fi

for dataset in "${DATASETS[@]}"; do
    for completed_rounds in "${ROUNDS[@]}"; do
        for budget in "${BUDGETS[@]}"; do
            condition_root="${OUTPUT_ROOT}/${dataset}/round_$(printf '%04d' "$completed_rounds")/${budget}"
            "$PYTHON_BIN" scripts/experiments/analysis/summarize_local_data_size_internal_probe.py \
                --input_root "$condition_root" \
                --output_json "${condition_root}/summary.json" \
                --output_csv "${condition_root}/summary.csv"
        done
    done
done

"$PYTHON_BIN" scripts/experiments/analysis/analyze_local_probe_refit.py \
    --input_root "$OUTPUT_ROOT" --output_dir "$ANALYSIS_ROOT"

echo "Completed ${job_count} local-probe refit jobs under ${OUTPUT_ROOT}"
echo "Analysis output: ${ANALYSIS_ROOT}"
