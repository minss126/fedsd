#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Decomposed BYOT gradient probe for the alpha=0 vs. alpha=1 comparison.
#
# At each probe round and for every sampled client, it records gradients of
#   1) teacher CE,
#   2) branch CE,
#   3) branch KD,
#   4) feature imitation,
# separately.  It also derives the configured training gradient
#   teacher CE + (1-alpha)*branch CE + alpha*branch KD + beta*feature.
#
# The central comparison is branch CE vs. branch KD.  Both all-parameter and
# shared-backbone-only gradients are saved.  The shared-backbone scope excludes
# the teacher classifier and the branch-private adapters/classifiers.

# Default to the requested two-GPU setup; override with GPUS_OVERRIDE if needed.
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
PROBE_INTERVAL="${PROBE_INTERVAL:-10}"
PROBE_BATCHES="${PROBE_BATCHES:-1}"
PROBE_SCOPES="${PROBE_SCOPES:-all,shared_backbone}"
CIFAR10_BETA="${CIFAR10_BETA:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_alpha_branch_ce_kd_gradient_probe_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ALPHAS=(${ALPHAS_OVERRIDE:-0.00 1.00})
CASES=(
    "cifar100|100|beta_0.5|--partition noniid --beta 0.5"
    "cifar100|100|beta_0.1|--partition noniid --beta 0.1"
    "cifar10|10|beta_${CIFAR10_BETA}|--partition noniid --beta ${CIFAR10_BETA}"
)

WANDB_FLAGS=""
if [ "${USE_WANDB:-0}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

alpha_tag() {
    printf "%s" "$1" | sed 's/\./p/g'
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

    local setting="${dataset}_resnet18/${env_name}/fedavg/seed${seed}"
    local method_name="alpha${alpha_name}_decomposed_gradient_probe"
    local log_dir="${LOG_ROOT}/${setting}"
    local log_file="${log_dir}/${method_name}.log"

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${setting} | alpha=${alpha}"
        return
    fi

    echo "[GPU ${gpu_id}] start: ${setting} | alpha=${alpha} | rounds=${ROUNDS}"
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
        --log_gradient_probe \
        --gradient_probe_interval "${PROBE_INTERVAL}" \
        --gradient_probe_batches "${PROBE_BATCHES}" \
        --gradient_probe_scopes "${PROBE_SCOPES}" \
        ${env_flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${setting} | alpha=${alpha}"
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
        for seed in "${SEEDS[@]}"; doss
            gpu_idx=$((job_count % NUM_GPUS))
            QUEUES[$gpu_idx]+="${dataset}|${num_classes}|${env_name}|${env_flags}|${alpha}|${seed}"$'\n'
            job_count=$((job_count + 1))
        done
    done
done

echo "========== Decomposed Branch-CE / Branch-KD Gradient Probe =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seeds=${SEEDS[*]}"
echo "cases=${CASES[*]}"
echo "alphas=${ALPHAS[*]}"
echo "probe_interval=${PROBE_INTERVAL}, probe_batches=${PROBE_BATCHES}, scopes=${PROBE_SCOPES}"
echo "feature_beta=${FEATURE_BETA}, temp=${TEMP_VAL}"
echo "log_root=${LOG_ROOT}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-0}"
echo "jobs=${job_count}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done

wait "${pids[@]}"
echo "Decomposed branch-CE / branch-KD gradient probe complete (${job_count} jobs)"
