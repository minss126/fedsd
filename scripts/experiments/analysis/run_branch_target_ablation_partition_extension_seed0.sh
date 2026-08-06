#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Extend the existing beta=0.5 branch-target ablation to IID and beta=0.1.
# This is the minimal trend screen requested for the paper table:
#   feature-only vs. CE+feature vs. KD+feature
#
# The settings intentionally match logs_branch_target_ablation_r500:
#   * CIFAR-10 and CIFAR-100
#   * B1+B2+B3, sum reduction, feature beta=0.01
#   * legacy branch-KD behavior: teacher/student T=0.5 and T^2 scaling
#   * 500 rounds, 5 local epochs, 100 clients, 10% participation
#   * seed 0 only

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

SEED="${SEED_OVERRIDE:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_branch_target_ablation_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

run_job() {
    local gpu_id=$1 dataset=$2 partition_name=$3 partition_flags=$4
    local variant=$5 objective=$6 alpha=$7
    local setting="${dataset}_resnet18/${partition_name}/fedavg/seed${SEED}"
    local log_dir="${LOG_ROOT}/${setting}"
    local result="${log_dir}/${variant}.pkl"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$result" ]; then
        echo "[GPU ${gpu_id}] skip: ${setting} | ${variant}"
        return
    fi

    echo "[GPU ${gpu_id}] start: ${setting} | ${variant}"
    "$PYTHON_BIN" main.py \
        --dataset "$dataset" --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED" \
        --device "cuda:${gpu_id}" \
        --logdir "$LOG_ROOT" --log_file_name "${setting}/${variant}" \
        --model resnet18_byot --alg fedbyot \
        --byot_active_branches 1,2,3 \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective "$objective" --byot_alpha "$alpha" \
        --byot_beta "$FEATURE_BETA" \
        --byot_branch_ce_label_smoothing 0.0 \
        --byot_branch_kd_filter none \
        --byot_branch_kd_target_mode full_teacher \
        --temperature "$KD_TEMPERATURE" \
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE" \
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE" \
        --byot_branch_kd_loss_scale_mode native_t2 \
        --byot_proxy_temperature 1.0 \
        $partition_flags \
        > "${log_dir}/${variant}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${setting} | ${variant}"
}

declare -a JOBS=()
add_job() {
    local dataset=$1 partition_name=$2 partition_flags=$3
    local variant=$4 objective=$5 alpha=$6
    JOBS+=("${dataset}|${partition_name}|${partition_flags}|${variant}|${objective}|${alpha}")
}

# Put beta=0.1 first because its client-size imbalance can make runtime less
# predictable; round-robin assignment keeps the GPU queues balanced.
for partition_spec in \
    "beta_0.1|--partition noniid --beta 0.1" \
    "iid|--partition iid"; do
    IFS='|' read -r partition_name partition_flags <<< "$partition_spec"
    for dataset in cifar100 cifar10; do
        add_job "$dataset" "$partition_name" "$partition_flags" feature_only feature_only 0.00
        add_job "$dataset" "$partition_name" "$partition_flags" ce_feature blend 0.00
        add_job "$dataset" "$partition_name" "$partition_flags" kd_feature blend 1.00
    done
done

echo "========== Branch Target Ablation: IID/beta=0.1 Seed-0 Extension =========="
echo "gpus=${GPUS[*]}, seed=${SEED}, jobs=${#JOBS[@]}"
echo "datasets=cifar100,cifar10 | partitions=iid,beta_0.1"
echo "conditions=feature_only,ce_feature,kd_feature"
echo "rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, reduction=sum"
echo "feature_beta=${FEATURE_BETA}, kd_temperature=${KD_TEMPERATURE}, kd_scale=native_t2"
echo "log_root=${LOG_ROOT}, skip_existing=${SKIP_EXISTING}"

run_queue() {
    local gpu_id=$1
    shift
    local job dataset partition_name partition_flags variant objective alpha
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset partition_name partition_flags variant objective alpha <<< "$job"
        run_job "$gpu_id" "$dataset" "$partition_name" "$partition_flags" \
            "$variant" "$objective" "$alpha"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done
for ((i = 0; i < ${#JOBS[@]}; i++)); do
    QUEUES[$((i % NUM_GPUS))]+="${JOBS[$i]}"$'\n'
done

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "Branch target partition extension complete (${#JOBS[@]} jobs)"
