#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Multi-round motivation experiment:
#   * independent teacher-only FedAvg trajectory for every K condition
#   * fixed CIFAR-100 train total (50,000), hence 50,000/K samples/client
#   * full client participation and 5 local epochs
#   * W/B and frozen-probe logits measured on the full official test set after
#     local update and before aggregation
#   * probes are fitted at each measured round on augmentation-free round-start
#     global features; private branch modules never participate

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

SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
CLIENT_COUNTS=(${CLIENT_COUNTS_OVERRIDE:-1 5 20 50 100})
ROUNDS="${ROUNDS:-50}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-50}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GEOMETRY_INTERVAL="${GEOMETRY_INTERVAL:-5}"
GEOMETRY_CLIENT_COUNT="${GEOMETRY_CLIENT_COUNT:-0}"
GEOMETRY_REFERENCE_BATCHES="${GEOMETRY_REFERENCE_BATCHES:-0}"
LOGIT_PROBE_EPOCHS="${LOGIT_PROBE_EPOCHS:-30}"
LOGIT_PROBE_LR="${LOGIT_PROBE_LR:-0.1}"
LOGIT_PROBE_WEIGHT_DECAY="${LOGIT_PROBE_WEIGHT_DECAY:-0.0005}"
LOGIT_PROBE_BATCH_SIZE="${LOGIT_PROBE_BATCH_SIZE:-512}"
LOGIT_PROBE_SAMPLES_PER_CLASS="${LOGIT_PROBE_SAMPLES_PER_CLASS:-500}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_iid_multiround_internal_geometry}"
PLOT_DIR="${PLOT_DIR:-${LOG_ROOT}/plots}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
PLOT_AFTER_RUN="${PLOT_AFTER_RUN:-1}"

WANDB_FLAGS=()
if [ "${USE_WANDB:-0}" = "1" ]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
fi

run_job() {
    local gpu=$1
    local n_clients=$2
    local seed=$3
    local setting="cifar100_resnet18/iid/clients_${n_clients}/fedavg/seed${seed}"
    local method="teacher_only_multiround_internal_geometry"
    local run_dir="${LOG_ROOT}/${setting}"
    local result_json="${run_dir}/${method}_postlocal_internal_geometry.json"
    local terminal_log="${run_dir}/${method}_terminal.log"
    mkdir -p "$run_dir"

    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$result_json" ]; then
        echo "[GPU ${gpu}] exists K=${n_clients}, seed=${seed}"
        return
    fi

    echo "[GPU ${gpu}] start K=${n_clients}, samples/client=$((50000 / n_clients)), seed=${seed}"
    "$PYTHON_BIN" main.py \
        --dataset cifar100 --datadir ./data \
        --model resnet18_byot --alg fedavg --partition iid \
        --n_clients "$n_clients" --sample_fraction 1.0 \
        --client_keep_last_batch \
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS" \
        --lr "$LR" --scheduler cosine --eta_min 0.0 \
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --seed "$seed" \
        --device "cuda:${gpu}" \
        --byot_active_branches none \
        --byot_branch_objective kd_only --byot_alpha 0.0 --byot_beta 0.0 \
        --sequential_client_execution \
        --log_postlocal_feature_geometry \
        --postlocal_geometry_interval "$GEOMETRY_INTERVAL" \
        --postlocal_geometry_client_count "$GEOMETRY_CLIENT_COUNT" \
        --postlocal_geometry_reference_batches "$GEOMETRY_REFERENCE_BATCHES" \
        --log_postlocal_logit_geometry \
        --postlocal_logit_probe_epochs "$LOGIT_PROBE_EPOCHS" \
        --postlocal_logit_probe_lr "$LOGIT_PROBE_LR" \
        --postlocal_logit_probe_weight_decay "$LOGIT_PROBE_WEIGHT_DECAY" \
        --postlocal_logit_probe_batch_size "$LOGIT_PROBE_BATCH_SIZE" \
        --postlocal_logit_probe_samples_per_class "$LOGIT_PROBE_SAMPLES_PER_CLASS" \
        --logdir "$LOG_ROOT" --log_file_name "${setting}/${method}" \
        "${WANDB_FLAGS[@]}" \
        > "$terminal_log" 2>&1
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

# Long client-loop jobs are enqueued first; unlike the old two-stage probe,
# every GPU starts an independent training job immediately.
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
    present=0
    for ordered in "${ORDERED_CLIENT_COUNTS[@]}"; do
        [ "$requested" = "$ordered" ] && present=1
    done
    [ "$present" = "0" ] && ORDERED_CLIENT_COUNTS+=("$requested")
done

job_count=0
for n_clients in "${ORDERED_CLIENT_COUNTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu_index=$((job_count % NUM_GPUS))
        QUEUES[$gpu_index]+="${n_clients}|${seed}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== Multi-round Within-client Feature + Logit Geometry =========="
echo "gpus=${GPUS[*]}, clients=${CLIENT_COUNTS[*]}, seeds=${SEEDS[*]}"
echo "rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, full participation, IID"
if [ "$GEOMETRY_CLIENT_COUNT" -le 0 ]; then
    GEOMETRY_CLIENT_DESCRIPTION="all participating clients"
else
    GEOMETRY_CLIENT_DESCRIPTION="up to ${GEOMETRY_CLIENT_COUNT} participating clients"
fi
echo "measurement=post-local/pre-aggregation, interval=${GEOMETRY_INTERVAL}, clients=${GEOMETRY_CLIENT_DESCRIPTION}"
echo "reference_batches=${GEOMETRY_REFERENCE_BATCHES} (0 means full official test set)"
echo "round-wise probes: epochs=${LOGIT_PROBE_EPOCHS}, train_samples/class=${LOGIT_PROBE_SAMPLES_PER_CLASS}"
echo "training=final teacher CE only; private branch objectives=off"
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

if [ "$PLOT_AFTER_RUN" = "1" ]; then
    "$PYTHON_BIN" scripts/experiments/analysis/plot_multiround_feature_geometry.py \
        --input_root "$LOG_ROOT" --output_dir "$PLOT_DIR"
fi
echo "========== Complete: ${LOG_ROOT} =========="
