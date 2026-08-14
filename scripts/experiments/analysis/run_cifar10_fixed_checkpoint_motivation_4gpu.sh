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
if [ "$NUM_GPUS" -ne 4 ]; then
    echo "This launcher expects exactly four GPU ids; got: ${GPUS[*]}" >&2
    echo 'Set GPUS_OVERRIDE="0 1 2 3" (or four other ids).' >&2
    exit 1
fi

# Build one CIFAR-10 checkpoint with the same training protocol used by the
# CIFAR-100 K=20 checkpoint. This is a single shared initialization, not a K
# sweep. Set BASE_CHECKPOINT to reuse another compatible CIFAR-10 checkpoint.
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-logs/analysis/logs_cifar10_fixed_checkpoint_source}"
CHECKPOINT_SETTING="cifar10_resnet18/iid/clients_20/fedavg/seed0"
CHECKPOINT_METHOD="teacher_only_fixed_checkpoint"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${CHECKPOINT_SETTING}"
DEFAULT_BASE_CHECKPOINT="${CHECKPOINT_DIR}/${CHECKPOINT_METHOD}_final.pt"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$DEFAULT_BASE_CHECKPOINT}"

ROUNDS="${ROUNDS:-50}"
CHECKPOINT_LOCAL_EPOCHS="${CHECKPOINT_LOCAL_EPOCHS:-5}"
CHECKPOINT_LR="${CHECKPOINT_LR:-0.1}"
CHECKPOINT_BATCH_SIZE="${CHECKPOINT_BATCH_SIZE:-50}"
NUM_WORKERS="${NUM_WORKERS:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/analysis/logs_cifar10_fixed_checkpoint_motivation}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:-${OUTPUT_ROOT}/shared_round_start_probe.pt}"
SAMPLE_SIZES=(${SAMPLE_SIZES_OVERRIDE:-100 250 500 1000 2500})
SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
CLIENTS_PER_CONDITION="${CLIENTS_PER_CONDITION:-10}"
LOCAL_STEPS="${LOCAL_STEPS:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-50}"
LOCAL_LR="${LOCAL_LR:-0.01}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_LR="${PROBE_LR:-0.1}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-5e-4}"
# Zero means the entire CIFAR-10 train set (5,000 images per class).
PROBE_SAMPLES_PER_CLASS="${PROBE_SAMPLES_PER_CLASS:-0}"
PROBE_SEED="${PROBE_SEED:-3407}"
FORCE_PROBE="${FORCE_PROBE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_ROOT"

echo "========== CIFAR-10 fixed-checkpoint motivation =========="
echo "gpus=${GPUS[*]}"
echo "checkpoint=${BASE_CHECKPOINT}"
echo "checkpoint protocol=K20 IID, full participation, ${ROUNDS} rounds, ${CHECKPOINT_LOCAL_EPOCHS} local epochs"
echo "local sample sizes=${SAMPLE_SIZES[*]}, seeds=${SEEDS[*]}, forks/condition=${CLIENTS_PER_CONDITION}"
echo "budgets=fixed_step(${LOCAL_STEPS} steps) + fixed_epoch(${LOCAL_EPOCHS} epochs)"
echo "metrics=logits only (no within/between feature variance)"
echo "probe=full CIFAR-10 train set; reference=full official CIFAR-10 test set"
echo "first run estimate=about 1h45m-2h30m including checkpoint preparation"
echo "reuse-checkpoint estimate=about 15-30m"

if [ ! -s "$BASE_CHECKPOINT" ]; then
    if [ "$BASE_CHECKPOINT" != "$DEFAULT_BASE_CHECKPOINT" ]; then
        echo "Missing user-supplied BASE_CHECKPOINT: $BASE_CHECKPOINT" >&2
        exit 1
    fi
    echo "[GPU ${GPUS[0]}] preparing the single shared CIFAR-10 global checkpoint"
    "$PYTHON_BIN" main.py \
        --dataset cifar10 --datadir ./data \
        --model resnet18_byot --alg fedavg --partition iid \
        --n_clients 20 --sample_fraction 1.0 --client_keep_last_batch \
        --round "$ROUNDS" --epochs "$CHECKPOINT_LOCAL_EPOCHS" \
        --lr "$CHECKPOINT_LR" --scheduler cosine --eta_min 0.0 \
        --batch_size "$CHECKPOINT_BATCH_SIZE" --test_batch_size 512 \
        --num_workers "$NUM_WORKERS" --seed 0 --device "cuda:${GPUS[0]}" \
        --byot_active_branches none \
        --byot_branch_objective kd_only --byot_alpha 0.0 --byot_beta 0.0 \
        --sequential_client_execution --save_final_ckpt \
        --logdir "$CHECKPOINT_ROOT" \
        --log_file_name "${CHECKPOINT_SETTING}/${CHECKPOINT_METHOD}" \
        > "${CHECKPOINT_DIR}/${CHECKPOINT_METHOD}_terminal.log" 2>&1
    if [ ! -s "$BASE_CHECKPOINT" ]; then
        echo "Checkpoint training ended without producing: $BASE_CHECKPOINT" >&2
        exit 1
    fi
else
    echo "Using existing shared global checkpoint: $BASE_CHECKPOINT"
fi

if [ "$FORCE_PROBE" = "1" ] || [ ! -s "$PROBE_CHECKPOINT" ]; then
    echo "[GPU ${GPUS[0]}] fitting the shared frozen probes once"
    "$PYTHON_BIN" scripts/experiments/analysis/local_data_size_internal_probe.py prepare \
        --dataset cifar10 --metrics logits \
        --global_checkpoint "$BASE_CHECKPOINT" --probe_output "$PROBE_CHECKPOINT" \
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
    local gpu=$1 budget=$2 sample_size=$3 seed=$4
    local job_dir="${OUTPUT_ROOT}/${budget}/sample_${sample_size}/seed_${seed}"
    local output="${job_dir}/metrics.json"
    local terminal="${job_dir}/terminal.log"
    mkdir -p "$job_dir"

    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$output" ]; then
        echo "[GPU ${gpu}] exists ${budget}, n=${sample_size}, seed=${seed}"
        return
    fi

    local train_budget=steps
    if [ "$budget" = "fixed_epoch" ]; then
        train_budget=epochs
    fi
    echo "[GPU ${gpu}] start ${budget}, n=${sample_size}, seed=${seed}"
    "$PYTHON_BIN" scripts/experiments/analysis/local_data_size_internal_probe.py run \
        --dataset cifar10 --metrics logits \
        --global_checkpoint "$BASE_CHECKPOINT" \
        --probe_checkpoint "$PROBE_CHECKPOINT" --output "$output" \
        --sample_size "$sample_size" --clients "$CLIENTS_PER_CONDITION" \
        --train_budget "$train_budget" --local_steps "$LOCAL_STEPS" \
        --local_epochs "$LOCAL_EPOCHS" --local_batch_size "$LOCAL_BATCH_SIZE" \
        --lr "$LOCAL_LR" --momentum "$MOMENTUM" --weight_decay "$WEIGHT_DECAY" \
        --datadir ./data --device "cuda:${gpu}" \
        --batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
        --test_samples_per_class 0 --seed "$seed" \
        > "$terminal" 2>&1
    echo "[GPU ${gpu}] done ${budget}, n=${sample_size}, seed=${seed}"
}

run_queue() {
    local gpu=$1
    shift
    local job
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r budget sample_size seed <<< "$job"
        run_job "$gpu" "$budget" "$sample_size" "$seed"
    done
}

declare -a QUEUES
for ((index = 0; index < NUM_GPUS; index++)); do
    QUEUES[$index]=""
done

job_count=0
sample_index=0
for sample_size in "${SAMPLE_SIZES[@]}"; do
    seed_index=0
    for seed in "${SEEDS[@]}"; do
        # Reverse the pair order on alternating conditions so the more
        # expensive large-n fixed-epoch jobs do not all land on GPUs 1 and 3.
        if [ $(((sample_index + seed_index) % 2)) -eq 0 ]; then
            budget_order=(fixed_step fixed_epoch)
        else
            budget_order=(fixed_epoch fixed_step)
        fi
        for budget in "${budget_order[@]}"; do
            gpu_index=$((job_count % NUM_GPUS))
            QUEUES[$gpu_index]+="${budget}|${sample_size}|${seed}"$'\n'
            job_count=$((job_count + 1))
        done
        seed_index=$((seed_index + 1))
    done
    sample_index=$((sample_index + 1))
done

pids=()
for ((index = 0; index < NUM_GPUS; index++)); do
    mapfile -t jobs <<< "${QUEUES[$index]}"
    run_queue "${GPUS[$index]}" "${jobs[@]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [ "$status" -ne 0 ]; then
    echo "At least one GPU queue failed; inspect terminal.log files." >&2
    exit "$status"
fi

for budget in fixed_step fixed_epoch; do
    "$PYTHON_BIN" scripts/experiments/analysis/summarize_local_data_size_internal_probe.py \
        --input_root "${OUTPUT_ROOT}/${budget}" \
        --output_json "${OUTPUT_ROOT}/${budget}/summary.json" \
        --output_csv "${OUTPUT_ROOT}/${budget}/summary.csv"
done

echo "Completed ${job_count} jobs: ${OUTPUT_ROOT}/{fixed_step,fixed_epoch}/summary.json"
