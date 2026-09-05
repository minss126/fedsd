#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "Set GPUS_OVERRIDE to one or more GPU ids." >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x venv/bin/python ]; then
        PYTHON_BIN="venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

ROUNDS="${ROUNDS:-500}"
ANALYSIS_ROUNDS="${ANALYSIS_ROUNDS:-50,100,250,300,400,500}"
SEEDS=(${SEEDS_OVERRIDE:-0})
LOCAL_EPOCHS=(${LOCAL_EPOCHS_OVERRIDE:-1 5 10})
PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.1})
METHODS=(${METHODS_OVERRIDE:-plain adaptive})

LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
N_CLIENTS="${N_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
REFERENCE_PER_CLASS="${REFERENCE_PER_CLASS:-10}"
REFERENCE_SEED="${REFERENCE_SEED:-1729}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.5}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_aggregation_induced_degradation_pilot}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${LOG_ROOT}/.matplotlib}"
export MPLCONFIGDIR
mkdir -p "$LOG_ROOT" "$MPLCONFIGDIR"

ADAPTIVE_WARMUP=$(awk -v rounds="$ROUNDS" -v ratio="$WARMUP_RATIO" \
    'BEGIN { printf "%d", int(rounds * ratio + 0.5) }')

WANDB_FLAGS=()
if [ "${USE_WANDB:-0}" = "1" ]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
fi

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

partition_label() {
    case "$1" in
        iid) echo "IID" ;;
        beta_0.1) echo "Gamma = 0.1" ;;
        *) echo "$1" ;;
    esac
}

has_completed_run() {
    local log_file=$1
    local model_csv=$2
    [ -f "$log_file" ] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file" \
        && [ -f "$model_csv" ] \
        && awk -F, -v target="$ROUNDS" 'NR > 1 && $5 == target { found=1 } END { exit !found }' "$model_csv"
}

run_job() {
    local gpu_id=$1 method=$2 partition=$3 epochs=$4 seed=$5
    local run_name log_name log_dir log_file analysis_dir label
    local -a CMD PARTITION_ARGS METHOD_ARGS

    run_name="${method}_${partition}_e${epochs}_seed${seed}"
    log_name="runs/${method}/${partition}/e${epochs}/seed${seed}/${run_name}"
    log_dir="${LOG_ROOT}/runs/${method}/${partition}/e${epochs}/seed${seed}"
    log_file="${LOG_ROOT}/${log_name}.log"
    analysis_dir="${LOG_ROOT}/analysis_outputs/${method}/${partition}/e${epochs}/seed${seed}"
    label="$(partition_label "$partition")"
    mkdir -p "$log_dir" "$analysis_dir"

    if [ "$SKIP_EXISTING" = "1" ] \
        && has_completed_run "$log_file" "$analysis_dir/model_level.csv"; then
        echo "[skip] $run_name"
        return
    fi

    mapfile -t PARTITION_ARGS < <(partition_args "$partition")
    if [ "$method" = "plain" ]; then
        METHOD_ARGS=(--model resnet18 --alg fedavg)
    elif [ "$method" = "adaptive" ]; then
        METHOD_ARGS=(
            --model resnet18_byot --alg fedbyot
            --byot_active_branches 1,2,3
            --byot_branch_loss_reduction sum
            --byot_branch_objective kd_only
            --byot_beta "$FEATURE_BETA"
            --temperature "$KD_TEMPERATURE"
            --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
            --byot_proxy_temperature "$PROXY_TEMPERATURE"
            --byot_alpha "$LAMBDA_MAX"
            --byot_round_lambda_schedule linear
            --byot_round_lambda_min 0.0
            --byot_round_lambda_warmup "$ADAPTIVE_WARMUP"
            --alpha_min_scale 0.0
            --byot_client_proxy teacher_label_prob
            --byot_client_alpha_min 0.0
            --byot_client_alpha_max 1.0
            --byot_client_alpha_mode multiply
            --byot_client_reliability_power 1.0
            --byot_client_skew_proxy prediction_entropy
            --byot_client_skew_power "$SKEW_POWER"
            --byot_client_skew_min_scale 0.0
            --byot_client_skew_correction_mode soft_relax
            --byot_client_skew_soft_tau "$SOFT_TAU"
            --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
        )
    else
        echo "Unknown method: $method" >&2
        return 1
    fi

    CMD=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data --num_classes 100
        --n_clients "$N_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$epochs" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$seed" --device "cuda:${gpu_id}"
        --logdir "$LOG_ROOT" --log_file_name "$log_name"
        --paired_resnet_init --paired_execution_rng
        --analyze_aggregation_damage
        --aggregation_damage_analysis_rounds "$ANALYSIS_ROUNDS"
        --aggregation_damage_reference_per_class "$REFERENCE_PER_CLASS"
        --aggregation_damage_reference_batch_size "$TEST_BATCH_SIZE"
        --aggregation_damage_reference_seed "$REFERENCE_SEED"
        --aggregation_damage_output_dir "$analysis_dir"
        --aggregation_damage_method_label "${method^}"
        --aggregation_damage_partition_label "$label"
        --aggregation_damage_overwrite
    )
    CMD+=("${PARTITION_ARGS[@]}" "${METHOD_ARGS[@]}" "${WANDB_FLAGS[@]}")

    echo "[GPU $gpu_id] start: $run_name | analysis=$ANALYSIS_ROUNDS"
    if [ "$DRY_RUN" = "1" ]; then
        printf '  %q' "${CMD[@]}"
        printf '\n'
        return
    fi
    "${CMD[@]}" > "${log_dir}/${run_name}_terminal.log" 2>&1
    echo "[GPU $gpu_id] complete: $run_name"
}

# Long/high-epoch adaptive jobs are assigned first.  Greedy static placement
# keeps heterogeneous Plain/Adaptive and E={1,5,10} workloads balanced for any
# number of GPUs supplied through GPUS_OVERRIDE.
JOBS=()
for seed in "${SEEDS[@]}"; do
    for epochs in 10 5 1; do
        if [[ ! " ${LOCAL_EPOCHS[*]} " =~ " ${epochs} " ]]; then
            continue
        fi
        for method in adaptive plain; do
            if [[ ! " ${METHODS[*]} " =~ " ${method} " ]]; then
                continue
            fi
            for partition in "${PARTITIONS[@]}"; do
                multiplier=10
                [ "$method" = "adaptive" ] && multiplier=14
                weight=$((epochs * multiplier))
                JOBS+=("${method}|${partition}|${epochs}|${seed}|${weight}")
            done
        done
    done
done

declare -a QUEUES LOADS
for ((index=0; index<${#GPUS[@]}; index++)); do
    QUEUES[$index]=""
    LOADS[$index]=0
done

for job in "${JOBS[@]}"; do
    weight="${job##*|}"
    best=0
    for ((index=1; index<${#GPUS[@]}; index++)); do
        if [ "${LOADS[$index]}" -lt "${LOADS[$best]}" ]; then
            best=$index
        fi
    done
    QUEUES[$best]+="${job}"$'\n'
    LOADS[$best]=$((LOADS[$best] + weight))
done

echo "========== Aggregation-induced Degradation Pilot =========="
echo "gpus=${GPUS[*]}, rounds=$ROUNDS, analysis_rounds=$ANALYSIS_ROUNDS"
echo "methods=${METHODS[*]}, partitions=${PARTITIONS[*]}, local_epochs=${LOCAL_EPOCHS[*]}, seeds=${SEEDS[*]}"
echo "reference_per_class=$REFERENCE_PER_CLASS, output=$LOG_ROOT"
for ((index=0; index<${#GPUS[@]}; index++)); do
    echo "queue gpu=${GPUS[$index]} estimated_weight=${LOADS[$index]}"
done

run_queue() {
    local gpu_id=$1 queue=$2
    while IFS='|' read -r method partition epochs seed weight; do
        [ -z "${method:-}" ] && continue
        run_job "$gpu_id" "$method" "$partition" "$epochs" "$seed"
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
    echo "At least one aggregation-damage pilot queue failed." >&2
    exit "$status"
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run complete; plot generation skipped."
    exit 0
fi

"$PYTHON_BIN" scripts/experiments/analysis/plot_aggregation_damage.py \
    --input-root "$LOG_ROOT/analysis_outputs" \
    --output-dir "$LOG_ROOT/summary"

echo "Pilot complete: $LOG_ROOT"
