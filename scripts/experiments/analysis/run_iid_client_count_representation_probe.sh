#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Diagnosis-only experiment:
#   * CIFAR-100 IID
#   * K in {1, 20, 100}
#   * every client participates in every round
#   * only the final teacher CE updates the model
#   * branch/private heads never supervise the backbone
#
# With 50 rounds and 5 local epochs every training sample is processed 250
# times, matching the expected sample exposure of the usual 500-round,
# 10%-participation configuration while avoiding a 10x compute increase.

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

SEEDS=(${SEEDS_OVERRIDE:-0 1})
CLIENT_COUNTS=(${CLIENT_COUNTS_OVERRIDE:-1 20 100})
ROUNDS="${ROUNDS:-50}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-50}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DRIFT_INTERVAL="${DRIFT_INTERVAL:-10}"
CLIENT_PROBE_COUNT="${CLIENT_PROBE_COUNT:-10}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_LR="${PROBE_LR:-0.1}"
PROBE_SAMPLES_PER_CLASS="${PROBE_SAMPLES_PER_CLASS:-500}"
COMMON_SAMPLES_PER_CLASS="${COMMON_SAMPLES_PER_CLASS:-20}"
GRADIENT_BATCHES="${GRADIENT_BATCHES:-2}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_iid_client_count_representation}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

WANDB_FLAGS=()
if [ "${USE_WANDB:-0}" = "1" ]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
fi

training_complete() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu=$1
    local n_clients=$2
    local seed=$3
    local setting="cifar100_resnet18/iid/clients_${n_clients}/fedavg/seed${seed}"
    local method="teacher_only_client_count"
    local run_dir="${LOG_ROOT}/${setting}"
    local log_file="${run_dir}/${method}.log"
    local terminal_file="${run_dir}/${method}_terminal.log"
    local checkpoint="${run_dir}/${method}_final.pt"
    local client_dir="${run_dir}/${method}_final_clients"
    local probe_output="${run_dir}/${method}_representation_probe.json"
    local probe_terminal="${run_dir}/${method}_representation_probe_terminal.log"
    mkdir -p "$run_dir"

    if [ "$SKIP_EXISTING" != "1" ] || ! training_complete "$log_file" || \
       [ ! -f "$checkpoint" ] || [ ! -f "${client_dir}/manifest.json" ]; then
        echo "[GPU ${gpu}] train K=${n_clients}, seed=${seed}, full participation"
        "$PYTHON_BIN" main.py \
            --dataset cifar100 --datadir ./data \
            --model resnet18_byot --alg fedavg \
            --partition iid \
            --n_clients "$n_clients" --sample_fraction 1.0 \
            --client_keep_last_batch \
            --round "$ROUNDS" --epochs "$LOCAL_EPOCHS" \
            --lr "$LR" --scheduler cosine --eta_min 0.0 \
            --batch_size "$BATCH_SIZE" --test_batch_size 512 \
            --num_workers "$NUM_WORKERS" --seed "$seed" \
            --device "cuda:${gpu}" \
            --byot_active_branches none \
            --byot_branch_objective kd_only --byot_alpha 0.0 --byot_beta 0.0 \
            --sequential_client_execution \
            --log_client_drift --log_layerwise_client_update_drift \
            --drift_log_interval "$DRIFT_INTERVAL" \
            --save_final_ckpt --save_final_client_ckpts \
            --final_client_ckpt_count "$CLIENT_PROBE_COUNT" \
            --logdir "$LOG_ROOT" --log_file_name "${setting}/${method}" \
            "${WANDB_FLAGS[@]}" \
            > "$terminal_file" 2>&1
    else
        echo "[GPU ${gpu}] training exists K=${n_clients}, seed=${seed}"
    fi

    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$probe_output" ]; then
        echo "[GPU ${gpu}] probe exists K=${n_clients}, seed=${seed}"
        return
    fi

    echo "[GPU ${gpu}] frozen/shared probe K=${n_clients}, seed=${seed}"
    "$PYTHON_BIN" scripts/experiments/analysis/client_count_representation_probe.py \
        --global_checkpoint "$checkpoint" \
        --client_checkpoint_dir "$client_dir" \
        --output "$probe_output" \
        --dataset cifar100 --datadir ./data \
        --device "cuda:${gpu}" --batch_size "$PROBE_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --probe_epochs "$PROBE_EPOCHS" --probe_lr "$PROBE_LR" \
        --probe_samples_per_class "$PROBE_SAMPLES_PER_CLASS" \
        --common_samples_per_class "$COMMON_SAMPLES_PER_CLASS" \
        --gradient_batches "$GRADIENT_BATCHES" --seed "$seed" \
        > "$probe_terminal" 2>&1
    echo "[GPU ${gpu}] complete K=${n_clients}, seed=${seed}"
}

run_queue() {
    local gpu=$1
    shift
    local job
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r n_clients seed <<< "$job"
        run_job "$gpu" "$n_clients" "$seed"
    done
}

declare -a QUEUES
for ((index = 0; index < NUM_GPUS; index++)); do
    QUEUES[$index]=""
done

# Prefer larger-client jobs first because their client-loop and drift logging
# overhead is higher.  Append any custom client counts that are not in the
# usual sweep so CLIENT_COUNTS_OVERRIDE works for arbitrary values as well.
ORDERED_CLIENT_COUNTS=()
for preferred in 100 50 20 5 1; do
    for requested in "${CLIENT_COUNTS[@]}"; do
        if [ "$requested" = "$preferred" ]; then
            ORDERED_CLIENT_COUNTS+=("$requested")
            break
        fi
    done
done
for requested in "${CLIENT_COUNTS[@]}"; do
    already_added=0
    for ordered in "${ORDERED_CLIENT_COUNTS[@]}"; do
        if [ "$requested" = "$ordered" ]; then
            already_added=1
            break
        fi
    done
    if [ "$already_added" = "0" ]; then
        ORDERED_CLIENT_COUNTS+=("$requested")
    fi
done

job_count=0
for n_clients in "${ORDERED_CLIENT_COUNTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu_index=$((job_count % NUM_GPUS))
        QUEUES[$gpu_index]+="${n_clients}|${seed}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== IID Client-Count Representation Diagnostic =========="
echo "gpus=${GPUS[*]}, clients=${CLIENT_COUNTS[*]}, seeds=${SEEDS[*]}"
echo "rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, participation=1.0, partition=iid"
echo "training=final teacher CE only; branch loss=off; feature loss=off"
echo "probe=strict frozen global linear probes + shared-head post-local client diagnostics"
echo "saved_clients/run=${CLIENT_PROBE_COUNT}, probe_epochs=${PROBE_EPOCHS}, gradient_batches=${GRADIENT_BATCHES}"
echo "log_root=${LOG_ROOT}, jobs=${job_count}, skip_existing=${SKIP_EXISTING}"

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

echo "========== IID client-count representation diagnostic complete =========="
