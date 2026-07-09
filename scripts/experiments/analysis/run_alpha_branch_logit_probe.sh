#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Alpha probe with CE/KD gradients + branch logit statistics.
#
# Same representative cases as the CE-KD gradient probe:
#   1. CIFAR-100 + ResNet18-BYOT + beta=0.5
#   2. CIFAR-100 + ResNet18-BYOT + beta=0.1
#   3. CIFAR-10  + ResNet18-BYOT + beta=0.5
#
# The probe logs branch-wise:
#   - entropy / true-label probability / non-target mass / top-2 margin
#   - confidence / train-batch accuracy
#   - teacher-branch JS / KL / entropy gap

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

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
PROBE_INTERVAL="${PROBE_INTERVAL:-10}"
PROBE_BATCHES="${PROBE_BATCHES:-1}"
EXAMPLE_ROUNDS="${EXAMPLE_ROUNDS:-0,100,250,last}"
EXAMPLE_CLIENTS="${EXAMPLE_CLIENTS:-2}"
EXAMPLE_SAMPLES="${EXAMPLE_SAMPLES:-8}"
EXAMPLE_TOPK="${EXAMPLE_TOPK:-5}"
CIFAR10_BETA="${CIFAR10_BETA:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs_alpha_branch_logit_probe_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ALPHAS=(${ALPHAS_OVERRIDE:-0.00 1.00})
CASES=(
    "cifar100|100|beta_0.5|--partition noniid --beta 0.5"
    "cifar100|100|beta_0.1|--partition noniid --beta 0.1"
    "cifar10|10|beta_${CIFAR10_BETA}|--partition noniid --beta ${CIFAR10_BETA}"
)

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
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
    local alpha_name
    alpha_name="$(alpha_tag "$alpha")"

    local setting="${dataset}_resnet18/${env_name}/fedavg"
    local method_name="alpha${alpha_name}_branch_probe"
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
        --round "${ROUNDS}" --seed "${SEED}" \
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
        --log_branch_logit_examples \
        --branch_logit_example_rounds "${EXAMPLE_ROUNDS}" \
        --branch_logit_example_clients "${EXAMPLE_CLIENTS}" \
        --branch_logit_example_samples "${EXAMPLE_SAMPLES}" \
        --branch_logit_example_topk "${EXAMPLE_TOPK}" \
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
        IFS='|' read -r dataset num_classes env_name env_flags alpha <<< "$job"
        run_job "$gpu_id" "$dataset" "$num_classes" "$env_name" "$env_flags" "$alpha"
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
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${dataset}|${num_classes}|${env_name}|${env_flags}|${alpha}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== Alpha Branch Logit Probe =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "cases=${CASES[*]}"
echo "alphas=${ALPHAS[*]}"
echo "probe_interval=${PROBE_INTERVAL}, probe_batches=${PROBE_BATCHES}"
echo "example_rounds=${EXAMPLE_ROUNDS}, example_clients=${EXAMPLE_CLIENTS}, example_samples=${EXAMPLE_SAMPLES}, example_topk=${EXAMPLE_TOPK}"
echo "feature_beta=${FEATURE_BETA}, temp=${TEMP_VAL}"
echo "log_root=${LOG_ROOT}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"
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
echo "Alpha branch logit probe complete (${job_count} jobs)"
