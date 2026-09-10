#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPU ids provided. Set GPUS_OVERRIDE." >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then PYTHON_BIN="venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

# Seed-0 trend screen at the two heterogeneity endpoints: IID and severe
# non-IID (Dirichlet beta=0.1). Training and diagnostic KD temperatures are
# both fixed to T=1.0.
DATASETS=(${DATASETS_OVERRIDE:-cifar10 cifar100})
PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.1})
SEEDS=(${SEEDS_OVERRIDE:-0})
VARIANTS=(${VARIANTS_OVERRIDE:-ce_feature kd_feature})
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
TEMPERATURE="${TEMPERATURE:-1.0}"
PROBE_INTERVAL="${PROBE_INTERVAL:-50}"
PROBE_ROUNDS="${PROBE_ROUNDS_OVERRIDE:-}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-64}"
PROBE_CLIENTS="${PROBE_CLIENTS:-0}"
LOCAL_MAX_BATCHES="${LOCAL_MAX_BATCHES:-0}"
GLOBAL_MAX_BATCHES="${GLOBAL_MAX_BATCHES:-0}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_gradient_route_probe_t1_r500_min64}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

variant_args() {
    case "$1" in
        feature_only) echo "feature_only|0.00" ;;
        ce_feature)   echo "blend|0.00" ;;
        kd_feature)   echo "blend|1.00" ;;
        # Clearer aliases for runs whose FEATURE_BETA is zero.  They use the
        # same branch objectives as the historical *_feature variants but do
        # not imply that feature imitation is active.
        ce_only)      echo "blend|0.00" ;;
        kd_only)      echo "blend|1.00" ;;
        *) echo "Unknown variant: $1" >&2; return 1 ;;
    esac
}

partition_args() {
    case "$1" in
        iid)      echo "iid|0.5" ;;
        beta_0.5) echo "noniid|0.5" ;;
        beta_0.1) echo "noniid|0.1" ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

probe_schedule_args=()
if [ -n "$PROBE_ROUNDS" ]; then
    probe_schedule_args=(--gradient_route_probe_rounds "$PROBE_ROUNDS")
else
    probe_schedule_args=(--gradient_route_probe_interval "$PROBE_INTERVAL")
fi

run_job() {
    local gpu_id=$1 dataset=$2 partition_label=$3 variant=$4 seed=$5
    local objective alpha partition_mode beta setting log_dir result route_dir final_route
    IFS='|' read -r objective alpha <<< "$(variant_args "$variant")"
    IFS='|' read -r partition_mode beta <<< "$(partition_args "$partition_label")"
    setting="${dataset}_resnet18/${partition_label}/fedavg/seed${seed}"
    log_dir="${LOG_ROOT}/${setting}"
    result="${log_dir}/${variant}.pkl"
    route_dir="${log_dir}/${variant}_gradient_routes"
    final_route="${route_dir}/round_$(printf '%04d' "$ROUNDS").json"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$result" ] && [ -s "$final_route" ]; then
        echo "[GPU ${gpu_id}] skip: ${setting} | ${variant}"
        return 0
    fi

    echo "[GPU ${gpu_id}] start: ${setting} | ${variant} | T=${TEMPERATURE}"
    command=(
        "$PYTHON_BIN" main.py
        --dataset "$dataset" --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$seed" \
        --device "cuda:${gpu_id}" \
        --logdir "$LOG_ROOT" --log_file_name "${setting}/${variant}" \
        --model resnet18_byot --alg fedbyot \
        --partition "$partition_mode" --beta "$beta" \
        --min_require_size "$MIN_REQUIRE_SIZE" \
        --byot_active_branches 1,2,3 \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective "$objective" --byot_alpha "$alpha" \
        --byot_beta "$FEATURE_BETA" \
        --temperature "$TEMPERATURE" \
        --byot_branch_kd_teacher_temperature "$TEMPERATURE" \
        --byot_branch_kd_student_temperature "$TEMPERATURE" \
        --log_gradient_routes \
        "${probe_schedule_args[@]}" \
        --gradient_route_probe_batch_size "$PROBE_BATCH_SIZE" \
        --gradient_route_probe_client_count "$PROBE_CLIENTS" \
        --gradient_route_local_max_batches "$LOCAL_MAX_BATCHES" \
        --gradient_route_global_max_batches "$GLOBAL_MAX_BATCHES" \
        --gradient_route_temperature "$TEMPERATURE" \
        --gradient_route_branch_reduction sum \
        --gradient_route_output_dir "$route_dir"
    )
    if [ "$DRY_RUN" = "1" ]; then
        printf '[dry-run][GPU %s] ' "$gpu_id"
        printf '%q ' "${command[@]}"
        printf '\n'
        return 0
    fi
    "${command[@]}" > "${log_dir}/${variant}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${setting} | ${variant}"
}

declare -a JOBS=()
for dataset in "${DATASETS[@]}"; do
    for partition_label in "${PARTITIONS[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                JOBS+=("${dataset}|${partition_label}|${variant}|${seed}")
            done
        done
    done
done

echo "========== Independent CE/KD Gradient-Route Probe =========="
echo "gpus=${GPUS[*]}"
echo "datasets=${DATASETS[*]}, partitions=${PARTITIONS[*]}"
echo "variants=${VARIANTS[*]}, seeds=${SEEDS[*]}"
echo "rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, temperature=${TEMPERATURE}"
echo "partition min_require_size=${MIN_REQUIRE_SIZE}"
if [ -n "$PROBE_ROUNDS" ]; then
    echo "probe completed rounds=${PROBE_ROUNDS}"
else
    echo "probe interval=${PROBE_INTERVAL} completed rounds"
fi
echo "probe data=all participating clients/full local sets/full official test set"
echo "probe caps: clients=${PROBE_CLIENTS}(0=all), local_batches=${LOCAL_MAX_BATCHES}(0=all), global_batches=${GLOBAL_MAX_BATCHES}(0=all)"
echo "outputs=B1/B2/B3/All; aggregate + per-client + client summaries + full pairwise stats"
echo "estimated 4-GPU wall time=about 6-8 hours for the default 8 jobs"
echo "log_root=${LOG_ROOT}, jobs=${#JOBS[@]}, skip_existing=${SKIP_EXISTING}"

run_queue() {
    local gpu_id=$1
    shift
    local job dataset partition_label variant seed
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r dataset partition_label variant seed <<< "$job"
        run_job "$gpu_id" "$dataset" "$partition_label" "$variant" "$seed"
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

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
    echo "At least one gradient-route job failed; inspect *_terminal.log files." >&2
    exit "$status"
fi
echo "Gradient-route probe complete (${#JOBS[@]} jobs)"
