#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Target ablation for the question: why is branch CE less harmful than branch
# KD on simple tasks, but KD preferable on complex tasks?
#
# All conditions retain the final teacher hard-label CE and the same BYOT
# architecture / active branches / feature imitation.  They alter only the
# branch classification target.  `KD%` in each terminal log is the selected
# target fraction for filtered-KD rows (not an efficiency measurement).

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
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.10}"
KD_CONF_THRESHOLD="${KD_CONF_THRESHOLD:-0.80}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_branch_target_ablation_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

run_job() {
    local gpu_id=$1 dataset=$2 variant=$3 objective=$4 alpha=$5 smoothing=$6 kd_filter=$7 target_mode=$8 seed=$9
    local setting="${dataset}_resnet18/beta_0.5/fedavg/seed${seed}"
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
        --partition noniid --beta 0.5 \
        --byot_active_branches 1,2,3 \
        --byot_branch_objective "$objective" --byot_alpha "$alpha" \
        --byot_beta "$FEATURE_BETA" \
        --byot_branch_ce_label_smoothing "$smoothing" \
        --byot_branch_kd_filter "$kd_filter" \
        --byot_branch_kd_target_mode "$target_mode" \
        --byot_branch_kd_conf_threshold "$KD_CONF_THRESHOLD" \
        > "${log_dir}/${variant}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${setting} | ${variant}"
}

declare -a JOBS=()
add_variant() {
    local dataset=$1 variant=$2 objective=$3 alpha=$4 smoothing=$5 kd_filter=$6 target_mode=${7:-full_teacher}
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${dataset}|${variant}|${objective}|${alpha}|${smoothing}|${kd_filter}|${target_mode}|${seed}")
    done
}

add_dataset_suite() {
    local dataset=$1
    # Reference conditions.
    add_variant "$dataset" feature_only feature_only 0.00 0.00 none
    add_variant "$dataset" ce_feature blend 0.00 0.00 none
    # Tests whether generic target softening, without teacher relations,
    # reproduces the KD effect.
    add_variant "$dataset" "lsce$(printf '%.2f' "$LABEL_SMOOTHING" | tr '.' 'p')_feature" blend 0.00 "$LABEL_SMOOTHING" none
    # Tests teacher relations with all targets versus only reliable targets.
    add_variant "$dataset" kd_feature blend 1.00 0.00 none
    # Preserves q_T(y|x) per sample but deletes the teacher's non-target
    # class allocation.  This isolates the contribution of class relations.
    add_variant "$dataset" kd_teacher_mass_uniform_feature blend 1.00 0.00 none teacher_mass_uniform
    add_variant "$dataset" kd_teacher_correct_feature blend 1.00 0.00 teacher_correct
    add_variant "$dataset" "kd_teacher_correct_conf$(printf '%.2f' "$KD_CONF_THRESHOLD" | tr '.' 'p')_feature" blend 1.00 0.00 teacher_correct_confident
}

# Queue C100 first: this is the primary CE-removal claim.  C10 is the
# boundary condition explaining why the claim must not be universal.
add_dataset_suite cifar100
add_dataset_suite cifar10

echo "========== Branch Target Ablation =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seeds=${SEEDS[*]}"
echo "datasets=cifar100,cifar10 | partition=beta_0.5 | feature_beta=${FEATURE_BETA}"
echo "label_smoothing=${LABEL_SMOOTHING}, kd_conf_threshold=${KD_CONF_THRESHOLD}"
echo "conditions=feature_only,ce,lsce,kd_all,kd_teacher_mass_uniform,kd_correct,kd_correct_confident"
echo "log_root=${LOG_ROOT}, jobs=${#JOBS[@]}, skip_existing=${SKIP_EXISTING}"

run_queue() {
    local gpu_id=$1
    shift
    local job dataset variant objective alpha smoothing kd_filter target_mode seed
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset variant objective alpha smoothing kd_filter target_mode seed <<< "$job"
        run_job "$gpu_id" "$dataset" "$variant" "$objective" "$alpha" "$smoothing" "$kd_filter" "$target_mode" "$seed"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done
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
wait "${pids[@]}"
echo "Branch target ablation complete (${#JOBS[@]} jobs)"
