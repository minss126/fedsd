#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if [[ "${#GPUS[@]}" -ne 4 ]]; then
    echo "This launcher expects 4 GPUs; received: ${GPUS[*]}" >&2
    exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

DATASET="${DATASET:-cifar100}"
DATADIR="${DATADIR:-./data}"
ROUNDS="${ROUNDS:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
LR="${LR:-0.1}"
SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
PARTITION="${PARTITION:-iid}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-$((ROUNDS / 2))}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_layer_mismatch_update_steps}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/layer_mismatch/update_steps/${DATASET}_${PARTITION}_r${ROUNDS}}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
USE_WANDB="${USE_WANDB:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dxfl_layer_mismatch_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

partition_flags() {
    case "$PARTITION" in
        iid) echo "--partition iid" ;;
        beta_0.5) echo "--partition noniid --beta 0.5" ;;
        beta_0.3) echo "--partition noniid --beta 0.3" ;;
        beta_0.1) echo "--partition noniid --beta 0.1" ;;
        *) echo "Unknown PARTITION=${PARTITION}" >&2; exit 1 ;;
    esac
}

has_completed_log() {
    local log_file="$1"
    [[ -f "$log_file" ]] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id="$1" method="$2" seed="$3"
    local name="seed${seed}"
    local log_dir="${LOG_ROOT}/${DATASET}/${PARTITION}/${method}"
    local log_file="${log_dir}/${name}.log"
    local -a cmd partition_args_array wandb_flags
    mkdir -p "$log_dir"

    if [[ "$SKIP_EXISTING" == "1" ]] && has_completed_log "$log_file" \
        && [[ -f "${log_dir}/${name}.pkl" ]]; then
        echo "[skip] ${method} seed=${seed}"
        return
    fi

    read -r -a partition_args_array <<< "$(partition_flags)"
    wandb_flags=()
    if [[ "$USE_WANDB" == "1" ]]; then
        wandb_flags=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    fi

    cmd=(
        "$PYTHON_BIN" main.py
        --dataset "$DATASET" --datadir "$DATADIR"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers 0
        --round "$ROUNDS" --seed "$seed" --device "cuda:${gpu_id}"
        --logdir "$LOG_ROOT"
        --log_file_name "${DATASET}/${PARTITION}/${method}/${name}"
        --model resnet18_byot --log_update_step_size --log_update_step_common_ce
    )
    cmd+=("${partition_args_array[@]}")

    if [[ "$method" == "plain" ]]; then
        # Same BYOT-shaped architecture, but forward_teacher bypasses every
        # private exit. This makes the shared-path step norm directly paired.
        cmd+=(--alg fedavg)
    else
        cmd+=(
            --alg fedbyot
            --byot_active_branches "1,2,3"
            --byot_branch_loss_reduction sum
            --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
            --byot_alpha "$LAMBDA_MAX"
            --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00
            --byot_round_lambda_warmup "$WARMUP_ROUNDS"
            --temperature "$KD_TEMPERATURE"
            --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
            --byot_proxy_temperature "$PROXY_TEMPERATURE"
            --alpha_min_scale 0.0
            --byot_client_proxy teacher_label_prob
            --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00
            --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
            --byot_client_skew_proxy prediction_entropy
            --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.00
            --byot_client_skew_correction_mode soft_relax
            --byot_client_skew_soft_tau "$SOFT_TAU"
            --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
        )
    fi
    cmd+=("${wandb_flags[@]}")

    echo "[GPU ${gpu_id}] start ${method} seed=${seed}"
    "${cmd[@]}" > "${log_dir}/${name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete ${method} seed=${seed}"
}

jobs=()
# Adaptive jobs are slightly slower; placing them first balances the queues.
for method in adaptive plain; do
    for seed in "${SEEDS[@]}"; do
        jobs+=("${method}|${seed}")
    done
done

echo "========== Layer-mismatch update-step diagnostic =========="
echo "gpus=${GPUS[*]}"
echo "dataset=${DATASET}, partition=${PARTITION}, K=${NUM_CLIENTS}, C=${SAMPLE_FRACTION}"
echo "rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, batch=${BATCH_SIZE}, lr=${LR}"
echo "methods=plain teacher-only vs selected soft-b adaptive KD; seeds=${SEEDS[*]}"
echo "primary metric=sum of per-tensor L2 update norms on shared final path"
echo "also records=global L2, relative L2, gradient L2, all-model and block-wise values"
echo "common-objective control=Final-CE gradient norm at every local state"
echo "estimated 4-GPU time: R=100 about 2.5-5 hours; R=500 about 12-24 hours"

declare -a queues
for ((i=0; i<4; i++)); do queues[$i]=""; done
for ((i=0; i<${#jobs[@]}; i++)); do
    queues[$((i % 4))]+="${jobs[$i]}"$'\n'
done

pids=()
for ((i=0; i<4; i++)); do
    gpu_id="${GPUS[$i]}"
    (
        while IFS='|' read -r method seed; do
            [[ -z "$method" ]] && continue
            run_job "$gpu_id" "$method" "$seed"
        done <<< "${queues[$i]}"
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "At least one update-step run failed; inspect *_terminal.log." >&2
    exit 1
fi

plain_args=()
adaptive_args=()
for seed in "${SEEDS[@]}"; do
    plain_args+=(--plain "${LOG_ROOT}/${DATASET}/${PARTITION}/plain/seed${seed}.pkl")
    adaptive_args+=(--adaptive "${LOG_ROOT}/${DATASET}/${PARTITION}/adaptive/seed${seed}.pkl")
done
"$PYTHON_BIN" scripts/experiments/analysis/plot_layer_mismatch_update_steps.py \
    "${plain_args[@]}" "${adaptive_args[@]}" \
    --output-dir "$OUTPUT_DIR" --group shared_teacher --metric fedpart_l2sum
"$PYTHON_BIN" scripts/experiments/analysis/plot_layer_mismatch_update_steps.py \
    "${plain_args[@]}" "${adaptive_args[@]}" \
    --output-dir "$OUTPUT_DIR" --group shared_teacher --metric relative_l2
"$PYTHON_BIN" scripts/experiments/analysis/plot_layer_mismatch_update_steps.py \
    "${plain_args[@]}" "${adaptive_args[@]}" \
    --output-dir "$OUTPUT_DIR" --group shared_teacher --metric common_ce_gradient_l2

echo "Completed. Plots: ${OUTPUT_DIR}"
