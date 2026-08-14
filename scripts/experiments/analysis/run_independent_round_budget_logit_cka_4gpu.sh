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
ENDPOINT_ROUNDS=(${ENDPOINT_ROUNDS_OVERRIDE:-10 50 100})
SAMPLE_SIZES=(${SAMPLE_SIZES_OVERRIDE:-100 250 500 1000 2500})
SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
CLIENTS_PER_CONDITION="${CLIENTS_PER_CONDITION:-10}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-logs/analysis/logs_independent_round_budget_source}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/analysis/logs_independent_round_budget_logit_cka}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-logs/analysis/independent_round_budget_logit_cka_analysis}"
CHECKPOINT_METHOD="teacher_only_independent_endpoint"
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
PROBE_SAMPLES_PER_CLASS="${PROBE_SAMPLES_PER_CLASS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_REFERENCES="${FORCE_REFERENCES:-0}"
RUN_FINAL_ANALYSIS="${RUN_FINAL_ANALYSIS:-1}"

if [ "${#ENDPOINT_ROUNDS[@]}" -eq 0 ]; then
    echo "At least one positive endpoint round is required." >&2
    exit 1
fi
for total_rounds in "${ENDPOINT_ROUNDS[@]}"; do
    if [ "$total_rounds" -le 0 ]; then
        echo "Endpoint rounds must be positive; got ${total_rounds}." >&2
        exit 1
    fi
done

# The smallest independently trained endpoint supplies the shared R=0 control.
mapfile -t SORTED_ENDPOINT_ROUNDS < <(printf '%s\n' "${ENDPOINT_ROUNDS[@]}" | sort -n -u)
CONTROL_SOURCE_ROUND="${SORTED_ENDPOINT_ROUNDS[0]}"
DIAGNOSTIC_ROUNDS=(0 "${SORTED_ENDPOINT_ROUNDS[@]}")

endpoint_setting() {
    local dataset=$1 total_rounds=$2
    printf '%s_resnet18/iid/clients_20/fedavg/seed%s/budget_r%04d' \
        "$dataset" "$GLOBAL_SEED" "$total_rounds"
}

endpoint_checkpoint_path() {
    local dataset=$1 total_rounds=$2 completed_rounds=$3
    printf '%s/%s/%s_round%04d.pt' \
        "$CHECKPOINT_ROOT" "$(endpoint_setting "$dataset" "$total_rounds")" \
        "$CHECKPOINT_METHOD" "$completed_rounds"
}

diagnostic_checkpoint_path() {
    local dataset=$1 completed_rounds=$2
    if [ "$completed_rounds" -eq 0 ]; then
        endpoint_checkpoint_path "$dataset" "$CONTROL_SOURCE_ROUND" 0
    else
        endpoint_checkpoint_path "$dataset" "$completed_rounds" "$completed_rounds"
    fi
}

endpoint_complete() {
    local dataset=$1 total_rounds=$2
    [ -s "$(endpoint_checkpoint_path "$dataset" "$total_rounds" 0)" ] && \
        [ -s "$(endpoint_checkpoint_path "$dataset" "$total_rounds" "$total_rounds")" ]
}

run_endpoint() {
    local gpu=$1 dataset=$2 total_rounds=$3
    local setting run_dir
    setting="$(endpoint_setting "$dataset" "$total_rounds")"
    run_dir="${CHECKPOINT_ROOT}/${setting}"
    mkdir -p "$run_dir"

    if [ "$SKIP_EXISTING" = "1" ] && endpoint_complete "$dataset" "$total_rounds"; then
        echo "[GPU ${gpu}] endpoint exists: ${dataset} R=${total_rounds}"
        return
    fi

    echo "[GPU ${gpu}] train independent endpoint ${dataset} R=${total_rounds} cosine_horizon=${total_rounds}"
    "$PYTHON_BIN" main.py \
        --dataset "$dataset" --datadir ./data \
        --model resnet18_byot --alg fedavg --partition iid \
        --n_clients 20 --sample_fraction 1.0 --client_keep_last_batch \
        --round "$total_rounds" --epochs "$CHECKPOINT_LOCAL_EPOCHS" \
        --lr "$CHECKPOINT_LR" --scheduler cosine --eta_min 0.0 \
        --batch_size "$CHECKPOINT_BATCH_SIZE" --test_batch_size 512 \
        --num_workers "$NUM_WORKERS" --seed "$GLOBAL_SEED" --device "cuda:${gpu}" \
        --byot_active_branches none \
        --byot_branch_objective kd_only --byot_alpha 0.0 --byot_beta 0.0 \
        --sequential_client_execution \
        --save_global_ckpt_rounds "0,${total_rounds}" \
        --logdir "$CHECKPOINT_ROOT" \
        --log_file_name "${setting}/${CHECKPOINT_METHOD}" \
        > "${run_dir}/${CHECKPOINT_METHOD}_terminal.log" 2>&1

    if ! endpoint_complete "$dataset" "$total_rounds"; then
        echo "Independent endpoint training did not produce both R=0 and R=${total_rounds} checkpoints." >&2
        return 1
    fi
    echo "[GPU ${gpu}] completed independent endpoint ${dataset} R=${total_rounds}"
}

run_endpoint_queue() {
    local gpu=$1
    shift
    local job dataset total_rounds
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset total_rounds <<< "$job"
        run_endpoint "$gpu" "$dataset" "$total_rounds"
    done
}

echo "========== Independent FL-budget logit + CKA motivation =========="
echo "gpus=${GPUS[*]}"
echo "datasets=${DATASETS[*]}, independently trained final endpoints=${SORTED_ENDPOINT_ROUNDS[*]}"
echo "R=0 role=initialization sanity control from independent R=${CONTROL_SOURCE_ROUND} run"
echo "global checkpoint protocol=K20 IID, full participation, teacher CE only"
echo "global LR=cosine with horizon equal to each endpoint budget; lr=${CHECKPOINT_LR}, eta_min=0"
echo "local sample sizes=${SAMPLE_SIZES[*]}, sampling seeds=${SEEDS[*]}, forks/condition=${CLIENTS_PER_CONDITION}"
echo "local budgets=fixed_step(${LOCAL_STEPS}) + fixed_epoch(${LOCAL_EPOCHS}); identical local lr=${LOCAL_LR}"
echo "metrics=mean/pairwise centered-logit cosine + directional variance + mean/pairwise linear CKA"
echo "probe=endpoint-specific frozen heads fitted on full train set; reference=full official test set"
echo "estimated clean 4-GPU wall time: about 4-6.5 hours"
echo "estimated diagnostic-only rerun with endpoints/probes present: about 50-110 minutes"

# Greedy scheduling keeps the two R=100 jobs on separate GPUs while placing
# R=50 and R=10 work on the remaining queues. A trajectory itself is not split.
declare -a ENDPOINT_QUEUES ENDPOINT_LOADS
for ((index = 0; index < 4; index++)); do
    ENDPOINT_QUEUES[$index]=""
    ENDPOINT_LOADS[$index]=0
done
mapfile -t DESC_ENDPOINT_ROUNDS < <(printf '%s\n' "${SORTED_ENDPOINT_ROUNDS[@]}" | sort -nr)
for total_rounds in "${DESC_ENDPOINT_ROUNDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        min_index=0
        for ((index = 1; index < 4; index++)); do
            if [ "${ENDPOINT_LOADS[$index]}" -lt "${ENDPOINT_LOADS[$min_index]}" ]; then
                min_index=$index
            fi
        done
        ENDPOINT_QUEUES[$min_index]+="${dataset}|${total_rounds}"$'\n'
        ENDPOINT_LOADS[$min_index]=$((ENDPOINT_LOADS[$min_index] + total_rounds))
    done
done

endpoint_pids=()
for ((index = 0; index < 4; index++)); do
    mapfile -t jobs <<< "${ENDPOINT_QUEUES[$index]}"
    run_endpoint_queue "${GPUS[$index]}" "${jobs[@]}" &
    endpoint_pids+=("$!")
done
endpoint_status=0
for pid in "${endpoint_pids[@]}"; do
    if ! wait "$pid"; then endpoint_status=1; fi
done
if [ "$endpoint_status" -ne 0 ]; then
    echo "At least one independent endpoint queue failed." >&2
    exit "$endpoint_status"
fi

mkdir -p "$OUTPUT_ROOT" "$ANALYSIS_ROOT"
"$PYTHON_BIN" scripts/experiments/analysis/validate_independent_round_budget_checkpoints.py \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --datasets "$(IFS=,; echo "${DATASETS[*]}")" \
    --rounds "$(IFS=,; echo "${SORTED_ENDPOINT_ROUNDS[*]}")" \
    --global_seed "$GLOBAL_SEED" --expected_lr "$CHECKPOINT_LR" \
    --manifest_output "${OUTPUT_ROOT}/independent_endpoint_manifest.json" \
    > "${OUTPUT_ROOT}/checkpoint_validation.log"

prepare_reference() {
    local gpu=$1 dataset=$2 completed_rounds=$3
    local checkpoint round_dir reference
    checkpoint="$(diagnostic_checkpoint_path "$dataset" "$completed_rounds")"
    round_dir="${OUTPUT_ROOT}/${dataset}/round_$(printf '%04d' "$completed_rounds")"
    reference="${round_dir}/shared_logit_cka_probe.pt"
    mkdir -p "$round_dir"
    if [ "$FORCE_REFERENCES" != "1" ] && [ -s "$reference" ]; then
        echo "[GPU ${gpu}] reference exists ${dataset} independent R=${completed_rounds}"
        return
    fi
    echo "[GPU ${gpu}] prepare probe ${dataset} independent R=${completed_rounds}"
    "$PYTHON_BIN" scripts/experiments/analysis/local_data_size_internal_probe.py prepare \
        --dataset "$dataset" --metrics logits_cka \
        --global_checkpoint "$checkpoint" --probe_output "$reference" \
        --datadir ./data --device "cuda:${gpu}" \
        --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --probe_epochs "$PROBE_EPOCHS" --probe_lr "$PROBE_LR" \
        --probe_weight_decay "$PROBE_WEIGHT_DECAY" \
        --probe_samples_per_class "$PROBE_SAMPLES_PER_CLASS" \
        --test_samples_per_class 0 --seed 3407 \
        > "${round_dir}/prepare_logit_cka_reference.log" 2>&1
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
    for completed_rounds in "${DIAGNOSTIC_ROUNDS[@]}"; do
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
    local round_dir job_dir output train_budget
    round_dir="${OUTPUT_ROOT}/${dataset}/round_$(printf '%04d' "$completed_rounds")"
    job_dir="${round_dir}/${budget}/sample_${sample_size}/seed_${seed}"
    output="${job_dir}/metrics.json"
    mkdir -p "$job_dir"
    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$output" ] && \
        grep -q '"metrics": "logits_cka"' "$output"; then
        echo "[GPU ${gpu}] exists ${dataset} R=${completed_rounds} ${budget} n=${sample_size} seed=${seed}"
        return
    fi
    train_budget=steps
    if [ "$budget" = "fixed_epoch" ]; then train_budget=epochs; fi
    echo "[GPU ${gpu}] start ${dataset} R=${completed_rounds} ${budget} n=${sample_size} seed=${seed}"
    "$PYTHON_BIN" scripts/experiments/analysis/local_data_size_internal_probe.py run \
        --dataset "$dataset" --metrics logits_cka \
        --global_checkpoint "$(diagnostic_checkpoint_path "$dataset" "$completed_rounds")" \
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
    echo "[GPU ${gpu}] done ${dataset} R=${completed_rounds} ${budget} n=${sample_size} seed=${seed}"
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
mapfile -t ORDERED_SAMPLE_SIZES < <(printf '%s\n' "${SAMPLE_SIZES[@]}" | sort -nr)
for sample_size in "${ORDERED_SAMPLE_SIZES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for completed_rounds in "${DIAGNOSTIC_ROUNDS[@]}"; do
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
    echo "At least one local diagnostic queue failed." >&2
    exit "$job_status"
fi

for dataset in "${DATASETS[@]}"; do
    for completed_rounds in "${DIAGNOSTIC_ROUNDS[@]}"; do
        round_dir="${OUTPUT_ROOT}/${dataset}/round_$(printf '%04d' "$completed_rounds")"
        for budget in fixed_step fixed_epoch; do
            "$PYTHON_BIN" scripts/experiments/analysis/summarize_local_data_size_internal_probe.py \
                --input_root "${round_dir}/${budget}" \
                --output_json "${round_dir}/${budget}/summary.json" \
                --output_csv "${round_dir}/${budget}/summary.csv"
        done
    done
done

if [ "$RUN_FINAL_ANALYSIS" = "1" ]; then
    "$PYTHON_BIN" scripts/experiments/analysis/analyze_round_checkpoint_logit_cka.py \
        --input_root "$OUTPUT_ROOT" --output_dir "$ANALYSIS_ROOT"
    "$PYTHON_BIN" scripts/experiments/analysis/analyze_independent_round_budget_consistency.py \
        --input_root "$OUTPUT_ROOT" --output_dir "$ANALYSIS_ROOT"
fi

echo "Completed ${job_count} independent-endpoint logit/CKA jobs under ${OUTPUT_ROOT}"
echo "Analysis output: ${ANALYSIS_ROOT}"
