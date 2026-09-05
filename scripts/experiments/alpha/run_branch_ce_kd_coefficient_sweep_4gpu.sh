#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "Set GPUS_OVERRIDE to at least one GPU id." >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x venv/bin/python ]; then
        PYTHON_BIN="venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

SEEDS=(${SEEDS_OVERRIDE:-0})
PARTITIONS=(${PARTITIONS_OVERRIDE:-beta_0.5 beta_0.1})
COEFFICIENTS=(${COEFFICIENTS_OVERRIDE:-0.0 0.2 0.4 0.6 0.8 1.0})
DATASET="${DATASET_OVERRIDE:-cifar100}"

case "$DATASET" in
    cifar10)
        NUM_CLASSES=10
        DEFAULT_LOG_ROOT="logs/alpha/logs_branch_ce_kd_coefficient_sweep_cifar10"
        ;;
    cifar100)
        NUM_CLASSES=100
        DEFAULT_LOG_ROOT="logs/alpha/logs_branch_ce_kd_coefficient_sweep"
        ;;
    *)
        echo "Unsupported DATASET_OVERRIDE: $DATASET (supported: cifar10, cifar100)" >&2
        exit 1
        ;;
esac

ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-0.5}"
LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

WANDB_ARGS=()
if [ "${USE_WANDB:-0}" = "1" ]; then
    WANDB_ARGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
fi

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

coefficient_tag() {
    printf '%s' "$1" | sed 's/\./p/g'
}

is_zero() {
    awk -v value="$1" 'BEGIN { exit !(value == 0.0) }'
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1 partition=$2 objective=$3 coefficient=$4 seed=$5 label=$6
    local coefficient_name run_name log_dir log_file
    local -a command partition_flags

    coefficient_name="$(coefficient_tag "$coefficient")"
    run_name="${label}_coef${coefficient_name}_seed${seed}"
    log_dir="${LOG_ROOT}/${partition}/fedavg/seed${seed}"
    log_file="${log_dir}/${run_name}.log"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${partition} | ${run_name}"
        return
    fi

    mapfile -t partition_flags < <(partition_args "$partition")
    command=(
        "$PYTHON_BIN" main.py
        --dataset "$DATASET" --datadir ./data --num_classes "$NUM_CLASSES"
        --n_clients 100 --sample_fraction 0.1
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$seed" --device "cuda:${gpu_id}"
        --logdir "$LOG_ROOT"
        --log_file_name "${partition}/fedavg/seed${seed}/${run_name}"
        --model resnet18_byot --alg fedbyot
        --kd_conf_threshold 0.0
        --byot_active_branches 1,2,3
        --byot_branch_loss_reduction sum
        --byot_branch_objective "$objective"
        --byot_alpha "$coefficient"
        --byot_beta "$FEATURE_BETA"
        --temperature "$KD_TEMPERATURE"
    )
    command+=("${partition_flags[@]}" "${WANDB_ARGS[@]}")

    echo "[GPU ${gpu_id}] start: ${partition} | ${label} | coefficient=${coefficient} | seed=${seed}"
    if [ "$DRY_RUN" = "1" ]; then
        printf '  %q' "${command[@]}"
        printf '\n'
        return
    fi
    "${command[@]}" > "${log_dir}/${run_name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${partition} | ${label} | coefficient=${coefficient} | seed=${seed}"
}

# At coefficient zero, CE-only and KD-only are exactly the same objective:
# main CE + feature imitation.  Store one shared run instead of duplicating it.
JOBS=()
for seed in "${SEEDS[@]}"; do
    for partition in "${PARTITIONS[@]}"; do
        zero_added=0
        for coefficient in "${COEFFICIENTS[@]}"; do
            if is_zero "$coefficient"; then
                if [ "$zero_added" -eq 0 ]; then
                    JOBS+=("${partition}|ce_only|${coefficient}|${seed}|shared_zero")
                    zero_added=1
                fi
                continue
            fi
            JOBS+=("${partition}|ce_only|${coefficient}|${seed}|ce_only")
            JOBS+=("${partition}|kd_only|${coefficient}|${seed}|kd_only")
        done
    done
done

declare -a QUEUES
for ((index=0; index<${#GPUS[@]}; index++)); do
    QUEUES[$index]=""
done
for ((index=0; index<${#JOBS[@]}; index++)); do
    gpu_index=$((index % ${#GPUS[@]}))
    QUEUES[$gpu_index]+="${JOBS[$index]}"$'\n'
done

echo "========== Independent Branch CE/KD Coefficient Sweep =========="
echo "dataset=${DATASET} | GPUs=${GPUS[*]} | seeds=${SEEDS[*]} | partitions=${PARTITIONS[*]}"
echo "coefficients=${COEFFICIENTS[*]} | jobs=${#JOBS[@]}"
echo "rounds=${ROUNDS} | local_epochs=${LOCAL_EPOCHS} | feature_beta=${FEATURE_BETA} | KD_T=${KD_TEMPERATURE}"
echo "log_root=${LOG_ROOT}"

run_queue() {
    local gpu_id=$1 queue=$2
    while IFS='|' read -r partition objective coefficient seed label; do
        [ -z "${partition:-}" ] && continue
        run_job "$gpu_id" "$partition" "$objective" "$coefficient" "$seed" "$label"
    done <<< "$queue"
}

pids=()
for ((index=0; index<${#GPUS[@]}; index++)); do
    if [ -n "${QUEUES[$index]}" ]; then
        run_queue "${GPUS[$index]}" "${QUEUES[$index]}" &
        pids+=("$!")
    fi
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
    echo "At least one CE/KD coefficient-sweep queue failed." >&2
    exit "$status"
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run complete."
else
    echo "Independent branch CE/KD coefficient sweep complete (${#JOBS[@]} runs)."
fi
