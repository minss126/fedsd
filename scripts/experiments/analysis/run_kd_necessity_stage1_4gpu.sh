#!/usr/bin/env bash

# Stage 1: measure candidate branch-wise KD-necessity signals without changing
# the selected soft-b adaptive rule.  Six independent trajectories cover
# CIFAR-10/CIFAR-100 x IID/beta=0.5/beta=0.1.  At target rounds, all selected
# clients are evaluated on all of their local training samples both before the
# local update and after it/before FedAvg.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  scripts/experiments/analysis/run_kd_necessity_stage1_4gpu.sh

Common overrides:
  GPUS_OVERRIDE="0 1 2 3" SEEDS_OVERRIDE="0"
  ROUNDS=500 ANALYSIS_ROUNDS="50,100,250,500"
  PARTITIONS_OVERRIDE="iid beta_0.5 beta_0.1"
  KD_NECESSITY_MAX_BATCHES=0 KD_NECESSITY_CLIENT_COUNT=0
  USE_WANDB=0 SKIP_EXISTING=1 DRY_RUN=1

The defaults use all selected clients and every local batch.  This launcher
does not implement a new gate; it only diagnoses the existing adaptive method.
EOF
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if (( ${#GPUS[@]} != 4 )); then
    echo "This launcher requires exactly four GPU ids; received: ${GPUS[*]:-(none)}" >&2
    exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

SEEDS=(${SEEDS_OVERRIDE:-0})
DATASETS=(${DATASETS_OVERRIDE:-cifar10 cifar100})
PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.1})
ROUNDS="${ROUNDS:-500}"
ANALYSIS_ROUNDS="${ANALYSIS_ROUNDS:-50,100,250,500}"
FINAL_ANALYSIS_ROUND="$(awk -v spec="$ANALYSIS_ROUNDS" '
    BEGIN {
        count = split(spec, values, ","); maximum = 0;
        for (item_idx = 1; item_idx <= count; item_idx++) {
            value = values[item_idx] + 0;
            if (value > maximum) maximum = value;
        }
        print maximum;
    }')"
if (( FINAL_ANALYSIS_ROUND < 1 || FINAL_ANALYSIS_ROUND > ROUNDS )); then
    echo "ANALYSIS_ROUNDS must contain a round in [1, ${ROUNDS}]." >&2
    exit 1
fi
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"

FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.5}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
KD_NECESSITY_TEMPERATURE="${KD_NECESSITY_TEMPERATURE:-1.0}"
KD_NECESSITY_MAX_BATCHES="${KD_NECESSITY_MAX_BATCHES:-0}"
KD_NECESSITY_CLIENT_COUNT="${KD_NECESSITY_CLIENT_COUNT:-0}"

LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_kd_necessity_stage1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
WARMUP_ROUNDS="$(awk -v rounds="$ROUNDS" -v ratio="$WARMUP_RATIO" \
    'BEGIN { printf "%d", int(rounds * ratio + 0.5) }')"
MPLCONFIGDIR="${MPLCONFIGDIR:-${LOG_ROOT}/.matplotlib}"
export MPLCONFIGDIR
mkdir -p "$LOG_ROOT" "$MPLCONFIGDIR"

WANDB_FLAGS=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
fi

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

dataset_args() {
    case "$1" in
        cifar10) printf '%s\n' --dataset cifar10 --num_classes 10 ;;
        cifar100) printf '%s\n' --dataset cifar100 --num_classes 100 ;;
        *) echo "Unknown dataset: $1" >&2; return 1 ;;
    esac
}

has_complete_diagnostic() {
    local log_file="$1" csv_file="$2"
    [[ -f "$log_file" ]] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file" \
        && [[ -f "$csv_file" ]] \
        && awk -F, -v target="$FINAL_ANALYSIS_ROUND" \
            'NR > 1 && $7 == target && $9 == "client_post_local_pre_aggregation" {found=1} END {exit !found}' \
            "$csv_file"
}

run_job() {
    local gpu_id="$1" dataset="$2" partition="$3" seed="$4"
    local run_name log_name log_dir log_file analysis_dir
    local -a CMD DATASET_FLAGS PARTITION_FLAGS

    run_name="${dataset}_${partition}_seed${seed}_soft_b_kd_necessity_r${ROUNDS}"
    log_name="runs/${dataset}/${partition}/seed${seed}/${run_name}"
    log_dir="${LOG_ROOT}/runs/${dataset}/${partition}/seed${seed}"
    log_file="${LOG_ROOT}/${log_name}.log"
    analysis_dir="${LOG_ROOT}/diagnostics/${dataset}/${partition}/seed${seed}"
    mkdir -p "$log_dir" "$analysis_dir"

    if [[ "$SKIP_EXISTING" == "1" ]] \
        && has_complete_diagnostic "$log_file" "$analysis_dir/client_branch_metrics.csv"; then
        echo "[GPU ${gpu_id}] skip: ${dataset} | ${partition} | seed=${seed}"
        return 0
    fi

    mapfile -t DATASET_FLAGS < <(dataset_args "$dataset")
    mapfile -t PARTITION_FLAGS < <(partition_args "$partition")
    CMD=(
        "$PYTHON_BIN" main.py
        --datadir ./data --in_channels 3
        --model resnet18_byot --alg fedbyot
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$seed" --device "cuda:${gpu_id}"
        --logdir "$LOG_ROOT" --log_file_name "$log_name"
        --sequential_client_execution
        --paired_resnet_init --paired_execution_rng --preserve_byot_proxy_rng
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
        --byot_alpha "$LAMBDA_MAX"
        --byot_round_lambda_schedule linear --byot_round_lambda_min 0.0
        --byot_round_lambda_warmup "$WARMUP_ROUNDS"
        --alpha_min_scale 0.0
        --byot_client_proxy teacher_label_prob
        --byot_client_alpha_min 0.0 --byot_client_alpha_max 1.0
        --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
        --byot_client_skew_proxy prediction_entropy
        --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.0
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau "$SOFT_TAU"
        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
        --analyze_kd_necessity
        --kd_necessity_analysis_rounds "$ANALYSIS_ROUNDS"
        --kd_necessity_temperature "$KD_NECESSITY_TEMPERATURE"
        --kd_necessity_client_count "$KD_NECESSITY_CLIENT_COUNT"
        --kd_necessity_max_batches "$KD_NECESSITY_MAX_BATCHES"
        --kd_necessity_output_dir "$analysis_dir"
        --kd_necessity_overwrite
    )
    CMD+=("${DATASET_FLAGS[@]}" "${PARTITION_FLAGS[@]}" "${WANDB_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: ${dataset} | ${partition} | seed=${seed}"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run][GPU %s] ' "$gpu_id"
        printf '%q ' "${CMD[@]}"
        printf '\n'
        return 0
    fi
    if ! "${CMD[@]}" > "${log_dir}/${run_name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu_id}] failed: ${dataset} | ${partition} | seed=${seed}" >&2
        tail -40 "${log_dir}/${run_name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_complete_diagnostic "$log_file" "$analysis_dir/client_branch_metrics.csv"; then
        echo "[GPU ${gpu_id}] incomplete diagnostic: ${analysis_dir}" >&2
        return 1
    fi
    echo "[GPU ${gpu_id}] complete: ${dataset} | ${partition} | seed=${seed}"
}

JOBS=()
for seed in "${SEEDS[@]}"; do
    for partition in "${PARTITIONS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            JOBS+=("${dataset}|${partition}|${seed}")
        done
    done
done

declare -a QUEUES LOADS
for ((index = 0; index < 4; index++)); do
    QUEUES[$index]=""
    LOADS[$index]=0
done
for job in "${JOBS[@]}"; do
    target=0
    for ((index = 1; index < 4; index++)); do
        (( LOADS[index] < LOADS[target] )) && target=$index
    done
    QUEUES[$target]+="${job}"$'\n'
    LOADS[$target]=$((LOADS[$target] + 1))
done

echo "========== Stage-1 branch-wise KD-necessity diagnostic =========="
echo "gpus=${GPUS[*]}, jobs=${#JOBS[@]}, seeds=${SEEDS[*]}"
echo "datasets=${DATASETS[*]}, partitions=${PARTITIONS[*]}"
echo "R=${ROUNDS}, E=${LOCAL_EPOCHS}, target rounds=${ANALYSIS_ROUNDS}"
echo "measurement=all selected clients, local train distribution, pre-local + post-local/pre-aggregation"
echo "client cap=${KD_NECESSITY_CLIENT_COUNT} (0=all), batch cap=${KD_NECESSITY_MAX_BATCHES} (0=all)"
echo "adaptive rule=unchanged soft-b; this run only records diagnostic candidates"
echo "estimated 4-GPU wall time: about 4.5-6 hours for seed 0"
for ((index = 0; index < 4; index++)); do
    echo "GPU ${GPUS[$index]}: ${LOADS[$index]} job(s)"
done

run_queue() {
    local gpu_id="$1" queue="$2" failed=0
    while IFS='|' read -r dataset partition seed; do
        [[ -z "${dataset:-}" ]] && continue
        if ! run_job "$gpu_id" "$dataset" "$partition" "$seed"; then
            failed=1
        fi
    done <<< "$queue"
    return "$failed"
}

pids=()
for ((index = 0; index < 4; index++)); do
    if [[ -n "${QUEUES[$index]}" ]]; then
        run_queue "${GPUS[$index]}" "${QUEUES[$index]}" &
        pids+=("$!")
    fi
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if (( status != 0 )); then
    echo "At least one stage-1 KD-necessity job failed." >&2
    exit "$status"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete; summary generation skipped."
    exit 0
fi

"$PYTHON_BIN" scripts/experiments/analysis/analyze_kd_necessity_stage1.py \
    --input-root "$LOG_ROOT/diagnostics" \
    --output-dir "$LOG_ROOT/summary"

echo "Stage-1 diagnostic complete: ${LOG_ROOT}/summary/report.md"
