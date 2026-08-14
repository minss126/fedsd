#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then PYTHON_BIN="venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -ne 4 ]; then
    echo "Exactly four GPU ids are required; got: ${GPUS[*]}" >&2
    exit 1
fi

DATASETS=(${DATASETS_OVERRIDE:-cifar10 cifar100})
SAMPLE_SIZES=(${SAMPLE_SIZES_OVERRIDE:-100 500 2500})
SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
BUDGETS=(${BUDGETS_OVERRIDE:-fixed_step fixed_epoch})
N_CLIENTS="${N_CLIENTS:-20}"
ROUNDS="${ROUNDS:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LOCAL_STEPS="${LOCAL_STEPS:-100}"
LR="${LR:-0.1}"
ETA_MIN="${ETA_MIN:-0.0}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-50}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"

# With interval=ROUNDS, main.py measures the first local update and the final
# local update.  The final record is the primary endpoint.
GEOMETRY_INTERVAL="${GEOMETRY_INTERVAL:-$ROUNDS}"
GEOMETRY_CLIENT_COUNT="${GEOMETRY_CLIENT_COUNT:-0}"
GEOMETRY_REFERENCE_BATCHES="${GEOMETRY_REFERENCE_BATCHES:-0}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_LR="${PROBE_LR:-0.1}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-5e-4}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-512}"
PROBE_SAMPLES_PER_CLASS="${PROBE_SAMPLES_PER_CLASS:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/analysis/logs_end_to_end_local_n_fl}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "========== End-to-end local-n FedAvg motivation =========="
echo "gpus=${GPUS[*]}"
echo "datasets=${DATASETS[*]}, K=${N_CLIENTS}, rounds=${ROUNDS}, seeds=${SEEDS[*]}"
echo "local sample sizes=${SAMPLE_SIZES[*]} (disjoint client pools; nested within client across n)"
echo "budgets=${BUDGETS[*]}: fixed_step=${LOCAL_STEPS}, fixed_epoch=${LOCAL_EPOCHS}"
echo "trajectory=independent from round 0 for every dataset/budget/n/seed condition"
echo "LR=cosine over ${ROUNDS} rounds, lr=${LR}, eta_min=${ETA_MIN}"
echo "measurement=first and final post-local/pre-aggregation models, all selected clients"
echo "metrics=frozen-probe logits + cross-depth linear CKA on full official test set"
echo "probe=trajectory-specific round-start heads fitted on full official train set"
echo "estimated clean 4-GPU wall time: about 18-32 hours for both budgets/datasets/3 seeds"
echo "fixed-epoch only estimate: about 9-15 hours"

run_job() {
    local gpu=$1 dataset=$2 budget=$3 sample_size=$4 seed=$5
    local setting="${dataset}/${budget}/sample_${sample_size}/seed_${seed}"
    local method="teacher_only_end_to_end_local_n"
    local run_dir="${OUTPUT_ROOT}/${setting}"
    local result_json="${run_dir}/${method}_postlocal_internal_geometry.json"
    local terminal_log="${run_dir}/${method}_terminal.log"
    local step_flags=()
    mkdir -p "$run_dir"

    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$result_json" ]; then
        echo "[GPU ${gpu}] exists ${dataset} ${budget} n=${sample_size} seed=${seed}"
        return
    fi
    if [ "$budget" = "fixed_step" ]; then
        step_flags=(--local_steps_per_round "$LOCAL_STEPS")
    elif [ "$budget" != "fixed_epoch" ]; then
        echo "Unknown budget: ${budget}" >&2
        return 1
    fi

    echo "[GPU ${gpu}] start ${dataset} ${budget} n=${sample_size} seed=${seed}"
    "$PYTHON_BIN" main.py \
        --dataset "$dataset" --datadir ./data \
        --model resnet18_byot --alg fedavg --partition iid \
        --n_clients "$N_CLIENTS" --sample_fraction 1.0 \
        --client_samples_per_client "$sample_size" \
        --client_keep_last_batch \
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS" \
        --lr "$LR" --scheduler cosine --eta_min "$ETA_MIN" \
        --momentum "$MOMENTUM" --reg "$WEIGHT_DECAY" \
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --seed "$seed" --device "cuda:${gpu}" \
        --byot_active_branches none \
        --byot_branch_objective kd_only --byot_alpha 0.0 --byot_beta 0.0 \
        --sequential_client_execution \
        --log_postlocal_feature_geometry \
        --log_postlocal_logit_geometry \
        --postlocal_geometry_interval "$GEOMETRY_INTERVAL" \
        --postlocal_geometry_client_count "$GEOMETRY_CLIENT_COUNT" \
        --postlocal_geometry_reference_batches "$GEOMETRY_REFERENCE_BATCHES" \
        --postlocal_logit_probe_epochs "$PROBE_EPOCHS" \
        --postlocal_logit_probe_lr "$PROBE_LR" \
        --postlocal_logit_probe_weight_decay "$PROBE_WEIGHT_DECAY" \
        --postlocal_logit_probe_batch_size "$PROBE_BATCH_SIZE" \
        --postlocal_logit_probe_samples_per_class "$PROBE_SAMPLES_PER_CLASS" \
        --save_final_ckpt \
        --logdir "$OUTPUT_ROOT" --log_file_name "${setting}/${method}" \
        "${step_flags[@]}" \
        > "$terminal_log" 2>&1
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

# Greedy scheduling by approximate local optimizer steps.  A whole FL
# trajectory remains on one GPU; only independent conditions run in parallel.
declare -a QUEUES LOADS
for ((index = 0; index < NUM_GPUS; index++)); do
    QUEUES[$index]=""
    LOADS[$index]=0
done

jobs_file="$(mktemp)"
trap 'rm -f "$jobs_file"' EXIT
for dataset in "${DATASETS[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        for sample_size in "${SAMPLE_SIZES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                if [ "$budget" = "fixed_step" ]; then
                    weight=$((N_CLIENTS * LOCAL_STEPS * ROUNDS))
                else
                    batches=$(((sample_size + BATCH_SIZE - 1) / BATCH_SIZE))
                    weight=$((N_CLIENTS * batches * LOCAL_EPOCHS * ROUNDS))
                fi
                printf '%012d|%s|%s|%s|%s\n' \
                    "$weight" "$dataset" "$budget" "$sample_size" "$seed" >> "$jobs_file"
            done
        done
    done
done

job_count=0
while IFS='|' read -r weight dataset budget sample_size seed; do
    min_index=0
    for ((index = 1; index < NUM_GPUS; index++)); do
        if [ "${LOADS[$index]}" -lt "${LOADS[$min_index]}" ]; then min_index=$index; fi
    done
    numeric_weight=$((10#$weight))
    QUEUES[$min_index]+="${dataset}|${budget}|${sample_size}|${seed}"$'\n'
    LOADS[$min_index]=$((LOADS[$min_index] + numeric_weight))
    job_count=$((job_count + 1))
done < <(sort -t'|' -k1,1nr "$jobs_file")

echo "approximate optimizer-step loads per GPU=${LOADS[*]}"
pids=()
for ((index = 0; index < NUM_GPUS; index++)); do
    mapfile -t jobs <<< "${QUEUES[$index]}"
    run_queue "${GPUS[$index]}" "${jobs[@]}" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
if [ "$status" -ne 0 ]; then
    echo "At least one end-to-end FL trajectory failed; inspect terminal logs." >&2
    exit "$status"
fi

for dataset in "${DATASETS[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        "$PYTHON_BIN" scripts/experiments/analysis/summarize_end_to_end_local_n_fl.py \
            --input_root "${OUTPUT_ROOT}/${dataset}/${budget}" \
            --output_json "${OUTPUT_ROOT}/${dataset}/${budget}/summary.json" \
            --output_csv "${OUTPUT_ROOT}/${dataset}/${budget}/summary.csv"
    done
done

echo "Completed ${job_count} independent local-n FL trajectories under ${OUTPUT_ROOT}"
