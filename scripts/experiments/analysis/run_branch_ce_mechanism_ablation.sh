#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Complementary suite for the question:
# "Why can branch CE hurt the final teacher even when it is less harmful than KD?"
#
# A. CE-strength: all exits active, mean reduction; isolates total CE weight.
# B. CE-depth: feature imitation disabled, mean reduction; isolates the exit
#    depth at which hard-label CE is imposed without changing total CE weight.
# C. CE-skew: feature-only versus fixed-strength CE over IID/non-IID levels.
# Every run records same-client/same-batch branch-CE versus teacher-CE gradient
# alignment on the shared prefix through each branch.

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPU ids provided. Set GPUS_OVERRIDE." >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then PYTHON_BIN="venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

SEEDS=(${SEEDS_OVERRIDE:-0 1})
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
GRADIENT_INTERVAL="${GRADIENT_INTERVAL:-50}"
GRADIENT_BATCHES="${GRADIENT_BATCHES:-1}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_branch_ce_mechanism_ablation_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

run_job() {
    local gpu_id=$1 dataset=$2 partition_name=$3 partition_flags=$4 variant=$5 active_branches=$6 objective=$7 ce_weight=$8 feature_beta=$9 seed=${10}
    local setting="${dataset}_resnet18/${partition_name}/fedavg/seed${seed}"
    local log_dir="${LOG_ROOT}/${setting}"
    local result="${log_dir}/${variant}.pkl"
    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$result" ]; then
        echo "[GPU ${gpu_id}] skip: ${setting} | ${variant}"
        return
    fi

    echo "[GPU ${gpu_id}] start: ${setting} | ${variant}"
    "$PYTHON_BIN" main.py \
        --dataset "$dataset" --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$seed" \
        --device "cuda:${gpu_id}" \
        --logdir "$LOG_ROOT" --log_file_name "${setting}/${variant}" \
        --model resnet18_byot --alg fedbyot \
        --byot_active_branches "$active_branches" \
        --byot_branch_objective "$objective" --byot_alpha 0.00 \
        --byot_beta "$feature_beta" --byot_branch_ce_weight "$ce_weight" \
        --byot_branch_loss_reduction mean \
        --log_branch_gradient_alignment \
        --branch_gradient_probe_interval "$GRADIENT_INTERVAL" \
        --branch_gradient_probe_batches "$GRADIENT_BATCHES" \
        $partition_flags \
        > "${log_dir}/${variant}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${setting} | ${variant}"
}

declare -a JOBS=()
add_job() {
    local dataset=$1 partition_name=$2 partition_flags=$3 variant=$4 active_branches=$5 objective=$6 ce_weight=$7 feature_beta=$8
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${dataset}|${partition_name}|${partition_flags}|${variant}|${active_branches}|${objective}|${ce_weight}|${feature_beta}|${seed}")
    done
}

for dataset in cifar100 cifar10; do
    # A. Hold feature imitation and active exits fixed; vary only CE strength.
    add_job "$dataset" beta_0.5 "--partition noniid --beta 0.5" feature_only_all 1,2,3 feature_only 0.00 "$FEATURE_BETA"
    for weight in 0.10 0.25 0.50 1.00; do
        tag="$(printf '%.2f' "$weight" | tr '.' 'p')"
        add_job "$dataset" beta_0.5 "--partition noniid --beta 0.5" "cew${tag}_all_feature" 1,2,3 blend "$weight" "$FEATURE_BETA"
    done

    # B. Match total CE weight using mean reduction and remove feature loss,
    # so differing exits cannot be explained by a changing feature-loss set.
    add_job "$dataset" beta_0.5 "--partition noniid --beta 0.5" off_no_feature none blend 0.00 0.00
    for branches in 1 2 3 1,2,3; do
        tag="b$(echo "$branches" | tr ',' '_b')"
        add_job "$dataset" beta_0.5 "--partition noniid --beta 0.5" "ce_${tag}_no_feature" "$branches" blend 1.00 0.00
    done

    # C. Same feature and CE configuration across label-skew severity.
    for spec in "iid|--partition iid" "beta_0.3|--partition noniid --beta 0.3" "beta_0.1|--partition noniid --beta 0.1"; do
        IFS='|' read -r partition_name partition_flags <<< "$spec"
        add_job "$dataset" "$partition_name" "$partition_flags" feature_only_all 1,2,3 feature_only 0.00 "$FEATURE_BETA"
        add_job "$dataset" "$partition_name" "$partition_flags" ce_all_feature 1,2,3 blend 1.00 "$FEATURE_BETA"
    done
done

echo "========== Branch CE Mechanism Ablation =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seeds=${SEEDS[*]}"
echo "A=strength, B=depth/no-feature, C=IID/beta0.3/beta0.1 skew control"
echo "gradient_alignment=every ${GRADIENT_INTERVAL} rounds, ${GRADIENT_BATCHES} batch/client"
echo "log_root=${LOG_ROOT}, jobs=${#JOBS[@]}, skip_existing=${SKIP_EXISTING}"

run_queue() {
    local gpu_id=$1; shift
    local job dataset partition_name partition_flags variant active_branches objective ce_weight feature_beta seed
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset partition_name partition_flags variant active_branches objective ce_weight feature_beta seed <<< "$job"
        run_job "$gpu_id" "$dataset" "$partition_name" "$partition_flags" "$variant" "$active_branches" "$objective" "$ce_weight" "$feature_beta" "$seed"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done
for ((i = 0; i < ${#JOBS[@]}; i++)); do QUEUES[$((i % NUM_GPUS))]+="${JOBS[$i]}"$'\n'; done

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done
wait "${pids[@]}"
echo "Branch CE mechanism ablation complete (${#JOBS[@]} jobs)"
