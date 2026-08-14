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
if [ "${#GPUS[@]}" -ne 4 ]; then
    echo "Exactly four GPU ids are required; got: ${GPUS[*]}" >&2
    exit 1
fi

DATASETS=(${DATASETS_OVERRIDE:-cifar10 cifar100})
CHECKPOINT_ROUNDS=(${CHECKPOINT_ROUNDS_OVERRIDE:-0 10 50 100})
SAMPLE_SIZES=(${SAMPLE_SIZES_OVERRIDE:-100 250 500 1000 2500})
SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
CLIENTS_PER_CONDITION="${CLIENTS_PER_CONDITION:-10}"

TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-logs/analysis/logs_round_checkpoint_cka_source}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/analysis/logs_round_checkpoint_logit_cka}"
TRAJECTORY_METHOD="teacher_only_round_trajectory"
CHECKPOINT_LOCAL_EPOCHS="${CHECKPOINT_LOCAL_EPOCHS:-5}"
CHECKPOINT_LR="${CHECKPOINT_LR:-0.1}"
CHECKPOINT_BATCH_SIZE="${CHECKPOINT_BATCH_SIZE:-50}"

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
# Zero uses all 50,000 training images for both CIFAR-10 and CIFAR-100.
PROBE_SAMPLES_PER_CLASS="${PROBE_SAMPLES_PER_CLASS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_REFERENCES="${FORCE_REFERENCES:-0}"

MAX_ROUNDS=0
CHECKPOINT_ROUNDS_CSV=""
for completed_rounds in "${CHECKPOINT_ROUNDS[@]}"; do
    if [ "$completed_rounds" -gt "$MAX_ROUNDS" ]; then
        MAX_ROUNDS=$completed_rounds
    fi
    if [ -z "$CHECKPOINT_ROUNDS_CSV" ]; then
        CHECKPOINT_ROUNDS_CSV="$completed_rounds"
    else
        CHECKPOINT_ROUNDS_CSV+=",${completed_rounds}"
    fi
done

trajectory_setting() {
    local dataset=$1
    echo "${dataset}_resnet18/iid/clients_20/fedavg/seed0"
}

checkpoint_path() {
    local dataset=$1 completed_rounds=$2
    echo "${TRAJECTORY_ROOT}/$(trajectory_setting "$dataset")/${TRAJECTORY_METHOD}_round$(printf '%04d' "$completed_rounds").pt"
}

trajectory_complete() {
    local dataset=$1 completed_rounds
    for completed_rounds in "${CHECKPOINT_ROUNDS[@]}"; do
        [ -s "$(checkpoint_path "$dataset" "$completed_rounds")" ] || return 1
    done
}

run_trajectory() {
    local gpu=$1 dataset=$2
    local setting
    setting="$(trajectory_setting "$dataset")"
    local run_dir="${TRAJECTORY_ROOT}/${setting}"
    mkdir -p "$run_dir"

    if [ "$SKIP_EXISTING" = "1" ] && trajectory_complete "$dataset"; then
        echo "[GPU ${gpu}] trajectory checkpoints exist: ${dataset}"
        return
    fi

    echo "[GPU ${gpu}] start ${dataset} trajectory through ${MAX_ROUNDS} rounds"
    "$PYTHON_BIN" main.py \
        --dataset "$dataset" --datadir ./data \
        --model resnet18_byot --alg fedavg --partition iid \
        --n_clients 20 --sample_fraction 1.0 --client_keep_last_batch \
        --round "$MAX_ROUNDS" --epochs "$CHECKPOINT_LOCAL_EPOCHS" \
        --lr "$CHECKPOINT_LR" --scheduler cosine --eta_min 0.0 \
        --batch_size "$CHECKPOINT_BATCH_SIZE" --test_batch_size 512 \
        --num_workers "$NUM_WORKERS" --seed 0 --device "cuda:${gpu}" \
        --byot_active_branches none \
        --byot_branch_objective kd_only --byot_alpha 0.0 --byot_beta 0.0 \
        --sequential_client_execution \
        --save_global_ckpt_rounds "$CHECKPOINT_ROUNDS_CSV" \
        --save_final_ckpt \
        --logdir "$TRAJECTORY_ROOT" \
        --log_file_name "${setting}/${TRAJECTORY_METHOD}" \
        > "${run_dir}/${TRAJECTORY_METHOD}_terminal.log" 2>&1

    if ! trajectory_complete "$dataset"; then
        echo "Missing one or more ${dataset} trajectory checkpoints." >&2
        return 1
    fi
    echo "[GPU ${gpu}] completed ${dataset} trajectory"
}

echo "========== Round-checkpoint logit + CKA motivation =========="
echo "gpus=${GPUS[*]}"
echo "datasets=${DATASETS[*]}, completed_rounds=${CHECKPOINT_ROUNDS[*]}"
echo "checkpoint protocol=K20 IID, full participation, teacher CE only"
echo "local sample sizes=${SAMPLE_SIZES[*]}, seeds=${SEEDS[*]}, forks/condition=${CLIENTS_PER_CONDITION}"
echo "budgets=fixed_step(${LOCAL_STEPS}) + fixed_epoch(${LOCAL_EPOCHS})"
echo "metrics=pairwise logit cosine + directional logit variance + branch-pair cosine + linear CKA; no W/B"
echo "probe=checkpoint-specific frozen linear heads fitted on the full train set"
echo "reference=full official test set"
echo "estimated first-run wall time: about 4-6.5 hours"
echo "estimated diagnostic-only rerun with checkpoints present: about 50-110 minutes"

# A trajectory cannot be split across GPUs. Train CIFAR-10 and CIFAR-100 in
# parallel, then use all four GPUs for checkpoint-local diagnostics.
trajectory_pids=()
for dataset_index in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$dataset_index]}"
    gpu="${GPUS[$((dataset_index % 2))]}"
    run_trajectory "$gpu" "$dataset" &
    trajectory_pids+=("$!")
done
trajectory_status=0
for pid in "${trajectory_pids[@]}"; do
    if ! wait "$pid"; then
        trajectory_status=1
    fi
done
if [ "$trajectory_status" -ne 0 ]; then
    echo "At least one checkpoint trajectory failed." >&2
    exit "$trajectory_status"
fi

prepare_reference() {
    local gpu=$1 dataset=$2 completed_rounds=$3
    local checkpoint reference_dir reference
    checkpoint="$(checkpoint_path "$dataset" "$completed_rounds")"
    reference_dir="${OUTPUT_ROOT}/${dataset}/round_$(printf '%04d' "$completed_rounds")"
    reference="${reference_dir}/shared_logit_cka_probe.pt"
    mkdir -p "$reference_dir"
    if [ "$FORCE_REFERENCES" != "1" ] && [ -s "$reference" ]; then
        echo "[GPU ${gpu}] logit+CKA reference exists ${dataset} round=${completed_rounds}"
        return
    fi
    echo "[GPU ${gpu}] prepare logit probes + CKA reference ${dataset} round=${completed_rounds}"
    "$PYTHON_BIN" scripts/experiments/analysis/local_data_size_internal_probe.py prepare \
        --dataset "$dataset" --metrics logits_cka \
        --global_checkpoint "$checkpoint" --probe_output "$reference" \
        --datadir ./data --device "cuda:${gpu}" \
        --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --probe_epochs "$PROBE_EPOCHS" --probe_lr "$PROBE_LR" \
        --probe_weight_decay "$PROBE_WEIGHT_DECAY" \
        --probe_samples_per_class "$PROBE_SAMPLES_PER_CLASS" \
        --test_samples_per_class 0 --seed 3407 \
        > "${reference_dir}/prepare_logit_cka_reference.log" 2>&1
}

run_prepare_queue() {
    local gpu=$1
    shift
    local job dataset completed_rounds
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset completed_rounds <<< "$job"
        prepare_reference "$gpu" "$dataset" "$completed_rounds"
    done
}

declare -a PREPARE_QUEUES
for ((index = 0; index < 4; index++)); do PREPARE_QUEUES[$index]=""; done
prepare_count=0
for dataset in "${DATASETS[@]}"; do
    for completed_rounds in "${CHECKPOINT_ROUNDS[@]}"; do
        gpu_index=$((prepare_count % 4))
        PREPARE_QUEUES[$gpu_index]+="${dataset}|${completed_rounds}"$'\n'
        prepare_count=$((prepare_count + 1))
    done
done
prepare_pids=()
for ((index = 0; index < 4; index++)); do
    mapfile -t jobs <<< "${PREPARE_QUEUES[$index]}"
    run_prepare_queue "${GPUS[$index]}" "${jobs[@]}" &
    prepare_pids+=("$!")
done
for pid in "${prepare_pids[@]}"; do wait "$pid"; done

run_job() {
    local gpu=$1 dataset=$2 completed_rounds=$3 budget=$4 sample_size=$5 seed=$6
    local round_dir="${OUTPUT_ROOT}/${dataset}/round_$(printf '%04d' "$completed_rounds")"
    local job_dir="${round_dir}/${budget}/sample_${sample_size}/seed_${seed}"
    local output="${job_dir}/metrics.json"
    mkdir -p "$job_dir"
    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$output" ]; then
        if grep -q '"metrics": "logits_cka"' "$output"; then
            echo "[GPU ${gpu}] exists ${dataset} r=${completed_rounds} ${budget} n=${sample_size} seed=${seed}"
            return
        fi
        echo "[GPU ${gpu}] replacing incompatible metrics file: ${output}"
    fi
    local train_budget=steps
    if [ "$budget" = "fixed_epoch" ]; then train_budget=epochs; fi
    echo "[GPU ${gpu}] start ${dataset} r=${completed_rounds} ${budget} n=${sample_size} seed=${seed}"
    "$PYTHON_BIN" scripts/experiments/analysis/local_data_size_internal_probe.py run \
        --dataset "$dataset" --metrics logits_cka \
        --global_checkpoint "$(checkpoint_path "$dataset" "$completed_rounds")" \
        --probe_checkpoint "${round_dir}/shared_logit_cka_probe.pt" \
        --output "$output" --sample_size "$sample_size" \
        --clients "$CLIENTS_PER_CONDITION" \
        --train_budget "$train_budget" --local_steps "$LOCAL_STEPS" \
        --local_epochs "$LOCAL_EPOCHS" --local_batch_size "$LOCAL_BATCH_SIZE" \
        --lr "$LOCAL_LR" --momentum "$MOMENTUM" --weight_decay "$WEIGHT_DECAY" \
        --datadir ./data --device "cuda:${gpu}" \
        --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --test_samples_per_class 0 --seed "$seed" \
        > "${job_dir}/terminal.log" 2>&1
    echo "[GPU ${gpu}] done ${dataset} r=${completed_rounds} ${budget} n=${sample_size} seed=${seed}"
}

run_job_queue() {
    local gpu=$1
    shift
    local job dataset completed_rounds budget sample_size seed
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset completed_rounds budget sample_size seed <<< "$job"
        run_job "$gpu" "$dataset" "$completed_rounds" "$budget" "$sample_size" "$seed"
    done
}

declare -a JOB_QUEUES
for ((index = 0; index < 4; index++)); do JOB_QUEUES[$index]=""; done
job_count=0
condition_count=0
# Put expensive fixed-epoch jobs at large n first and distribute them round-robin.
mapfile -t ORDERED_SAMPLE_SIZES < <(printf '%s\n' "${SAMPLE_SIZES[@]}" | sort -nr)
for sample_size in "${ORDERED_SAMPLE_SIZES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for completed_rounds in "${CHECKPOINT_ROUNDS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                epoch_gpu_index=$((condition_count % 4))
                step_gpu_index=$(((condition_count + 2) % 4))
                JOB_QUEUES[$epoch_gpu_index]+="${dataset}|${completed_rounds}|fixed_epoch|${sample_size}|${seed}"$'\n'
                JOB_QUEUES[$step_gpu_index]+="${dataset}|${completed_rounds}|fixed_step|${sample_size}|${seed}"$'\n'
                condition_count=$((condition_count + 1))
                job_count=$((job_count + 2))
            done
        done
    done
done

job_pids=()
for ((index = 0; index < 4; index++)); do
    mapfile -t jobs <<< "${JOB_QUEUES[$index]}"
    run_job_queue "${GPUS[$index]}" "${jobs[@]}" &
    job_pids+=("$!")
done
job_status=0
for pid in "${job_pids[@]}"; do
    if ! wait "$pid"; then job_status=1; fi
done
if [ "$job_status" -ne 0 ]; then
    echo "At least one logit/CKA diagnostic queue failed." >&2
    exit "$job_status"
fi

for dataset in "${DATASETS[@]}"; do
    for completed_rounds in "${CHECKPOINT_ROUNDS[@]}"; do
        round_dir="${OUTPUT_ROOT}/${dataset}/round_$(printf '%04d' "$completed_rounds")"
        for budget in fixed_step fixed_epoch; do
            "$PYTHON_BIN" scripts/experiments/analysis/summarize_local_data_size_internal_probe.py \
                --input_root "${round_dir}/${budget}" \
                --output_json "${round_dir}/${budget}/summary.json" \
                --output_csv "${round_dir}/${budget}/summary.csv"
        done
    done
done

echo "Completed ${job_count} logit/CKA jobs under ${OUTPUT_ROOT}"
