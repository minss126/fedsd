#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    cat <<'EOF'
Usage:
  scripts/experiments/analysis/run_centralized_prefix_gradient_agreement_4gpu.sh
  scripts/experiments/analysis/run_centralized_prefix_gradient_agreement_2gpu.sh

Key overrides:
  GPUS_OVERRIDE="0 1 2 3"
  BUDGETS_OVERRIDE="fixed_step"
  SAMPLE_SIZES_OVERRIDE="100 500 2500"
  CENTRAL_EPOCHS=100
  CLIENTS_PER_CONDITION=10 SEEDS_OVERRIDE="0 1 2"
  REFERENCE_BATCH_SIZE=512
  LOCAL_OBJECTIVE=teacher_ce|kd_only|blend|adaptive_kd
  SD_ALPHA=<coefficient> SD_BETA=<coefficient> SD_TEMPERATURE=<temperature>
  (the simple-BYOT 4-GPU wrapper defaults to blend/0.15/0.05/0.5)
  ADAPTIVE_LAMBDA_MAX=1 ADAPTIVE_ROUND_SCALE=1
EOF
    exit 0
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then PYTHON_BIN="venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "At least one GPU id is required." >&2
    exit 1
fi

DATASETS=(${DATASETS_OVERRIDE:-cifar10 cifar100})
SAMPLE_SIZES=(${SAMPLE_SIZES_OVERRIDE:-100 500 2500})
LOCAL_SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
BUDGETS=(${BUDGETS_OVERRIDE:-fixed_step})
CHECKPOINT_SEED="${CHECKPOINT_SEED:-0}"
CLIENTS_PER_CONDITION="${CLIENTS_PER_CONDITION:-10}"

CENTRAL_EPOCHS="${CENTRAL_EPOCHS:-100}"
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
LOCAL_OBJECTIVE="${LOCAL_OBJECTIVE:-teacher_ce}"
SD_ALPHA="${SD_ALPHA:-1.0}"
SD_BETA="${SD_BETA:-0.01}"
SD_TEMPERATURE="${SD_TEMPERATURE:-0.5}"
SD_BRANCH_REDUCTION="${SD_BRANCH_REDUCTION:-sum}"
ADAPTIVE_LAMBDA_MAX="${ADAPTIVE_LAMBDA_MAX:-1.0}"
ADAPTIVE_ROUND_SCALE="${ADAPTIVE_ROUND_SCALE:-1.0}"
ADAPTIVE_PROXY_TEMPERATURE="${ADAPTIVE_PROXY_TEMPERATURE:-1.0}"
ADAPTIVE_RELIABILITY_POWER="${ADAPTIVE_RELIABILITY_POWER:-1.0}"
ADAPTIVE_SKEW_POWER="${ADAPTIVE_SKEW_POWER:-2.0}"
ADAPTIVE_SOFT_TAU="${ADAPTIVE_SOFT_TAU:-0.85}"
ADAPTIVE_SOFT_TEMPERATURE="${ADAPTIVE_SOFT_TEMPERATURE:-0.05}"

REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-512}"
TEST_SAMPLES_PER_CLASS="${TEST_SAMPLES_PER_CLASS:-0}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_LR="${PROBE_LR:-0.1}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-5e-4}"
PROBE_SAMPLES_PER_CLASS="${PROBE_SAMPLES_PER_CLASS:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-logs/analysis/logs_centralized_prefix_gradient_source}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/analysis/logs_centralized_prefix_gradient_agreement}"
REFERENCE_ROOT="${REFERENCE_ROOT:-$OUTPUT_ROOT}"

checkpoint_path() {
    local dataset=$1
    printf '%s/%s/seed_%s/central_teacher_epoch%04d.pt' \
        "$CHECKPOINT_ROOT" "$dataset" "$CHECKPOINT_SEED" "$CENTRAL_EPOCHS"
}

condition_root_path() {
    local dataset=$1
    printf '%s/%s/epoch_%04d' "$OUTPUT_ROOT" "$dataset" "$CENTRAL_EPOCHS"
}

reference_root_path() {
    local dataset=$1
    printf '%s/%s/epoch_%04d' "$REFERENCE_ROOT" "$dataset" "$CENTRAL_EPOCHS"
}

reference_path() {
    local dataset=$1
    printf '%s/global_gradient_reference.json' "$(reference_root_path "$dataset")"
}

probe_path() {
    local dataset=$1
    printf '%s/shared_logit_cka_probe.pt' "$(reference_root_path "$dataset")"
}

echo "========== Centralized-checkpoint Final-CE prefix-gradient agreement =========="
echo "gpus=${GPUS[*]}"
echo "datasets=${DATASETS[*]}"
echo "central checkpoint=${CENTRAL_EPOCHS} centralized epochs"
echo "checkpoint training=full official train set, Final CE only, cosine LR"
echo "local sample sizes=${SAMPLE_SIZES[*]}, seeds=${LOCAL_SEEDS[*]}, forks/condition=${CLIENTS_PER_CONDITION}"
echo "local budgets=${BUDGETS[*]}: fixed_step=${LOCAL_STEPS}, fixed_epoch=${LOCAL_EPOCHS}"
echo "local objective=${LOCAL_OBJECTIVE}; SD lambda/alpha=${SD_ALPHA}, beta=${SD_BETA}, T=${SD_TEMPERATURE}, branch reduction=${SD_BRANCH_REDUCTION}"
if [ "$LOCAL_OBJECTIVE" != "teacher_ce" ]; then
    echo "SD branch initialization=common checkpoint state (private branches were not trained by centralized Final CE)"
fi
if [ "$LOCAL_OBJECTIVE" = "adaptive_kd" ]; then
    echo "adaptive lambda=max${ADAPTIVE_LAMBDA_MAX}*round${ADAPTIVE_ROUND_SCALE}*teacher_label_prob*soft_b(prediction_entropy^${ADAPTIVE_SKEW_POWER}; tau=${ADAPTIVE_SOFT_TAU}, temp=${ADAPTIVE_SOFT_TEMPERATURE})"
fi
echo "measurement=U/A/rho + centered logits/JSD/directional variance/probe accuracy/linear CKA"
echo "reference=full official test set, batch=${REFERENCE_BATCH_SIZE}, eval mode, no optimizer update"
echo "probe=depth-specific frozen linear heads fitted on the full augmentation-free train set"
echo "data relation=central checkpoint and local forks reuse the official train split"
echo "output=${OUTPUT_ROOT}"
echo "shared baseline reference/probe root=${REFERENCE_ROOT}"
if [ "$NUM_GPUS" -ge 4 ]; then
    echo "rough first-run estimate: about 3-6 hours on ${NUM_GPUS} GPUs"
else
    echo "rough first-run estimate: about 5-10 hours on ${NUM_GPUS} GPUs"
fi

train_checkpoint() {
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

checkpoint_pids=()
for index in "${!DATASETS[@]}"; do
    gpu_index=$((index % NUM_GPUS))
    train_checkpoint "${GPUS[$gpu_index]}" "${DATASETS[$index]}" &
    checkpoint_pids+=("$!")
done
checkpoint_status=0
for pid in "${checkpoint_pids[@]}"; do
    if ! wait "$pid"; then checkpoint_status=1; fi
done
if [ "$checkpoint_status" -ne 0 ]; then
    echo "Centralized checkpoint preparation failed; inspect *_terminal.log." >&2
    exit "$checkpoint_status"
fi

prepare_references() {
    local gpu=$1 dataset=$2 output terminal probe probe_terminal
    output="$(reference_path "$dataset")"
    terminal="${output%.json}_terminal.log"
    mkdir -p "$(dirname "$output")"
    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$output" ] \
        && grep -q 'final_ce_prefix_gradient_agreement_reference' "$output"; then
        echo "[GPU ${gpu}] global gradient reference exists: ${dataset}"
    else
        echo "[GPU ${gpu}] measure global gradient reference: ${dataset}"
        "$PYTHON_BIN" scripts/experiments/analysis/final_ce_prefix_gradient_agreement.py prepare \
            --dataset "$dataset" --datadir ./data \
            --global_checkpoint "$(checkpoint_path "$dataset")" \
            --output "$output" --device "cuda:${gpu}" \
            --reference_batch_size "$REFERENCE_BATCH_SIZE" \
            --test_samples_per_class "$TEST_SAMPLES_PER_CLASS" \
            --num_workers "$NUM_WORKERS" --seed 3407 \
            > "$terminal" 2>&1
        echo "[GPU ${gpu}] completed global gradient reference: ${dataset}"
    fi

    probe="$(probe_path "$dataset")"
    probe_terminal="${probe%.pt}_terminal.log"
    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$probe" ]; then
        echo "[GPU ${gpu}] shared logit/CKA probe exists: ${dataset}"
    else
        echo "[GPU ${gpu}] fit shared logit/CKA probes: ${dataset}"
        "$PYTHON_BIN" scripts/experiments/analysis/local_data_size_internal_probe.py prepare \
            --dataset "$dataset" --metrics logits_cka \
            --global_checkpoint "$(checkpoint_path "$dataset")" \
            --probe_output "$probe" --datadir ./data --device "cuda:${gpu}" \
            --batch_size "$REFERENCE_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
            --probe_epochs "$PROBE_EPOCHS" --probe_lr "$PROBE_LR" \
            --probe_weight_decay "$PROBE_WEIGHT_DECAY" \
            --probe_samples_per_class "$PROBE_SAMPLES_PER_CLASS" \
            --test_samples_per_class "$TEST_SAMPLES_PER_CLASS" --seed 3407 \
            > "$probe_terminal" 2>&1
        echo "[GPU ${gpu}] completed shared logit/CKA probes: ${dataset}"
    fi
}

reference_pids=()
for index in "${!DATASETS[@]}"; do
    gpu_index=$((index % NUM_GPUS))
    prepare_references "${GPUS[$gpu_index]}" "${DATASETS[$index]}" &
    reference_pids+=("$!")
done
reference_status=0
for pid in "${reference_pids[@]}"; do
    if ! wait "$pid"; then reference_status=1; fi
done
if [ "$reference_status" -ne 0 ]; then
    echo "Global gradient-reference preparation failed; inspect *_terminal.log." >&2
    exit "$reference_status"
fi

run_job() {
    local gpu=$1 dataset=$2 budget=$3 sample_size=$4 seed=$5 train_budget
    local condition_root
    condition_root="$(condition_root_path "$dataset")"
    local job_dir="${condition_root}/${budget}/sample_${sample_size}/seed_${seed}"
    local output="${job_dir}/metrics.json"
    mkdir -p "$job_dir"
    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$output" ] \
        && grep -q 'logits_cka_final_ce_prefix_gradient_U_A_rho' "$output"; then
        echo "[GPU ${gpu}] exists ${dataset} ${budget} n=${sample_size} seed=${seed}"
        return
    fi
    train_budget=steps
    if [ "$budget" = "fixed_epoch" ]; then train_budget=epochs; fi
    echo "[GPU ${gpu}] start ${dataset} ${budget} n=${sample_size} seed=${seed}"
    "$PYTHON_BIN" scripts/experiments/analysis/final_ce_prefix_gradient_agreement.py run \
        --dataset "$dataset" --datadir ./data \
        --global_checkpoint "$(checkpoint_path "$dataset")" \
        --global_reference "$(reference_path "$dataset")" \
        --probe_checkpoint "$(probe_path "$dataset")" \
        --output "$output" --sample_size "$sample_size" \
        --clients "$CLIENTS_PER_CONDITION" --train_budget "$train_budget" \
        --local_steps "$LOCAL_STEPS" --local_epochs "$LOCAL_EPOCHS" \
        --local_batch_size "$LOCAL_BATCH_SIZE" --lr "$LOCAL_LR" \
        --momentum "$LOCAL_MOMENTUM" --weight_decay "$LOCAL_WEIGHT_DECAY" \
        --local_objective "$LOCAL_OBJECTIVE" --sd_alpha "$SD_ALPHA" \
        --sd_beta "$SD_BETA" --sd_temperature "$SD_TEMPERATURE" \
        --sd_branch_reduction "$SD_BRANCH_REDUCTION" \
        --adaptive_lambda_max "$ADAPTIVE_LAMBDA_MAX" \
        --adaptive_round_scale "$ADAPTIVE_ROUND_SCALE" \
        --adaptive_proxy_temperature "$ADAPTIVE_PROXY_TEMPERATURE" \
        --adaptive_reliability_power "$ADAPTIVE_RELIABILITY_POWER" \
        --adaptive_skew_power "$ADAPTIVE_SKEW_POWER" \
        --adaptive_soft_tau "$ADAPTIVE_SOFT_TAU" \
        --adaptive_soft_temperature "$ADAPTIVE_SOFT_TEMPERATURE" \
        --device "cuda:${gpu}" --reference_batch_size "$REFERENCE_BATCH_SIZE" \
        --test_samples_per_class "$TEST_SAMPLES_PER_CLASS" \
        --num_workers "$NUM_WORKERS" --seed "$seed" \
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
queue_index=0
mapfile -t ORDERED_SAMPLE_SIZES < <(printf '%s\n' "${SAMPLE_SIZES[@]}" | sort -nr)
for sample_size in "${ORDERED_SAMPLE_SIZES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for budget in "${BUDGETS[@]}"; do
            if [ "$budget" != "fixed_step" ] && [ "$budget" != "fixed_epoch" ]; then
                echo "Unsupported budget: ${budget}" >&2
                exit 1
            fi
            for seed in "${LOCAL_SEEDS[@]}"; do
                gpu_index=$((queue_index % NUM_GPUS))
                QUEUES[$gpu_index]+="${dataset}|${budget}|${sample_size}|${seed}"$'\n'
                queue_index=$((queue_index + 1))
                job_count=$((job_count + 1))
            done
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
for pid in "${job_pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
    echo "At least one prefix-gradient diagnostic failed; inspect terminal logs." >&2
    exit "$status"
fi

for dataset in "${DATASETS[@]}"; do
    condition_root="$(condition_root_path "$dataset")"
    for budget in "${BUDGETS[@]}"; do
        "$PYTHON_BIN" scripts/experiments/analysis/summarize_local_data_size_internal_probe.py \
            --input_root "${condition_root}/${budget}" \
            --output_json "${condition_root}/${budget}/summary.json" \
            --output_csv "${condition_root}/${budget}/summary.csv"
    done
done

echo "Completed ${job_count} prefix-gradient jobs under ${OUTPUT_ROOT}"
