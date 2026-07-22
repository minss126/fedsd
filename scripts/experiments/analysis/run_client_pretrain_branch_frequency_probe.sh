#!/bin/bash

set -euo pipefail

# Client-wise branch-logit probe at the *start* of a FedAvg round, before
# local optimization.  For every selected client, group target classes by that
# client's local class frequency and save the across-client mean/std/min/max
# for B1/B2/B3.  The std is the desired client-disagreement (instability)
# quantity; it is distinct from the mean entropy (softness).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS=(${GPUS_OVERRIDE:-0 1})
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
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LOW_RATIO="${LOW_RATIO:-0.5}"
HIGH_RATIO="${HIGH_RATIO:-1.5}"
PROBE_INTERVAL="${PROBE_INTERVAL:-10}"
PROBE_BATCHES="${PROBE_BATCHES:-0}"
POSTLOCAL_REF_PROBE_INTERVAL="${POSTLOCAL_REF_PROBE_INTERVAL:-10}"
POSTLOCAL_REF_SAMPLES_PER_CLASS="${POSTLOCAL_REF_SAMPLES_PER_CLASS:-8}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_client_pretrain_branch_frequency_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ALPHAS=(${ALPHAS_OVERRIDE:-0.00 1.00})
# C100/C10 comparison across IID, moderate non-IID, and severe non-IID.
# With two alpha values and two seeds this is 6 x 2 x 2 = 24 jobs.
CASES=(
    "cifar100|100|iid|--partition iid"
    "cifar100|100|beta_0.5|--partition noniid --beta 0.5"
    "cifar100|100|beta_0.1|--partition noniid --beta 0.1"
    "cifar10|10|iid|--partition iid"
    "cifar10|10|beta_0.5|--partition noniid --beta 0.5"
    "cifar10|10|beta_0.1|--partition noniid --beta 0.1"
)

WANDB_FLAGS=""
if [ "${USE_WANDB:-0}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

POSTLOCAL_FLAGS=()
if [ "${ENABLE_POSTLOCAL_BRANCH_DISTRIBUTION:-0}" = "1" ]; then
    POSTLOCAL_FLAGS=(
        --log_postlocal_branch_distribution_stats
        --postlocal_ref_probe_interval "${POSTLOCAL_REF_PROBE_INTERVAL}"
        --postlocal_ref_samples_per_class "${POSTLOCAL_REF_SAMPLES_PER_CLASS}"
    )
    if [ "${SAVE_POSTLOCAL_FULL_LOGITS:-0}" = "1" ]; then
        POSTLOCAL_FLAGS+=(--save_postlocal_full_logits)
    fi
fi

alpha_tag() {
    printf "%s" "$1" | tr '.' 'p'
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1
    local dataset=$2
    local num_classes=$3
    local env_name=$4
    local env_flags=$5
    local alpha=$6
    local seed=$7
    local alpha_name
    alpha_name="$(alpha_tag "$alpha")"

    local setting="${dataset}_resnet18/${env_name}/fedavg"
    local method_name="alpha${alpha_name}_seed${seed}_client_pretrain_branch_freq"
    local log_dir="${LOG_ROOT}/${setting}"
    local log_file="${log_dir}/${method_name}.log"

    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${setting} | alpha=${alpha} | seed=${seed}"
        return
    fi

    echo "[GPU ${gpu_id}] start: ${setting} | alpha=${alpha} | seed=${seed} | rounds=${ROUNDS}"
    "${PYTHON_BIN}" main.py \
        --dataset "${dataset}" --datadir ./data \
        --num_classes "${num_classes}" \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${LR}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${seed}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${setting}/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_active_branches "1,2,3" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective blend \
        --byot_alpha "${alpha}" \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        --log_train_branch_frequency_stats \
        --train_branch_freq_low_ratio "${LOW_RATIO}" \
        --train_branch_freq_high_ratio "${HIGH_RATIO}" \
        --log_client_branch_frequency_stats \
        --client_branch_freq_probe_interval "${PROBE_INTERVAL}" \
        --client_branch_freq_probe_batches "${PROBE_BATCHES}" \
        "${POSTLOCAL_FLAGS[@]}" \
        ${env_flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${setting} | alpha=${alpha} | seed=${seed}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")
    for job in "${jobs[@]}"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset num_classes env_name env_flags alpha seed <<< "$job"
        run_job "$gpu_id" "$dataset" "$num_classes" "$env_name" "$env_flags" "$alpha" "$seed"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for case_spec in "${CASES[@]}"; do
    IFS='|' read -r dataset num_classes env_name env_flags <<< "$case_spec"
    for alpha in "${ALPHAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            gpu_idx=$((job_count % NUM_GPUS))
            QUEUES[$gpu_idx]+="${dataset}|${num_classes}|${env_name}|${env_flags}|${alpha}|${seed}"$'\n'
            job_count=$((job_count + 1))
        done
    done
done

echo "========== Client Pretrain Branch-Frequency Probe =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seeds=${SEEDS[*]}"
echo "cases=${CASES[*]}"
echo "alphas=${ALPHAS[*]}, probe_interval=${PROBE_INTERVAL}, probe_batches=${PROBE_BATCHES}"
echo "postlocal_distribution=${ENABLE_POSTLOCAL_BRANCH_DISTRIBUTION:-0}, postlocal_interval=${POSTLOCAL_REF_PROBE_INTERVAL}, postlocal_samples_per_class=${POSTLOCAL_REF_SAMPLES_PER_CLASS}, save_full_logits=${SAVE_POSTLOCAL_FULL_LOGITS:-0}"
echo "low_ratio=${LOW_RATIO}, high_ratio=${HIGH_RATIO}, log_root=${LOG_ROOT}"
echo "jobs=${job_count}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-0}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    mapfile -t queue_items <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_items[@]}" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "========== Client pretrain branch-frequency probe complete (${job_count} jobs) =========="
