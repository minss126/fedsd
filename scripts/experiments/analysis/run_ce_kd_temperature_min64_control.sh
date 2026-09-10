#!/usr/bin/env bash

# Minimal control for the legacy CE-vs-KD discrepancy.
#
# Reuses the already valid IID CE/KD T=1 results and runs only:
#   1) IID      / KD-only / T=0.5
#   2) beta=0.1 / CE-only / T=1.0
#   3) beta=0.1 / KD-only / T=1.0
#   4) beta=0.1 / KD-only / T=0.5
#
# All non-IID jobs explicitly use the final min_require_size=64 protocol.
# Gradient-route logging is retained to match the existing T=1 analysis runs.
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
    if [ -x "venv/bin/python" ]; then
        PYTHON_BIN="venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"
SEED="${SEED:-0}"
PROBE_INTERVAL="${PROBE_INTERVAL:-50}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_ce_kd_temperature_min64_control_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$MIN_REQUIRE_SIZE" != "64" ]; then
    echo "This control requires MIN_REQUIRE_SIZE=64; got ${MIN_REQUIRE_SIZE}." >&2
    exit 1
fi

# partition | objective | alpha | temperature | run tag
JOBS=(
    "iid|blend|1.00|0.5|kd_only_t0p5"
    "beta_0.1|blend|0.00|1.0|ce_only_t1p0"
    "beta_0.1|blend|1.00|1.0|kd_only_t1p0"
    "beta_0.1|blend|1.00|0.5|kd_only_t0p5"
)

run_job() {
    local gpu_id=$1 job=$2
    local partition_label objective alpha temperature tag partition_mode beta
    IFS='|' read -r partition_label objective alpha temperature tag <<< "$job"

    case "$partition_label" in
        iid)
            partition_mode="iid"
            beta="0.5"
            ;;
        beta_0.1)
            partition_mode="noniid"
            beta="0.1"
            ;;
        *)
            echo "Unknown partition: ${partition_label}" >&2
            return 1
            ;;
    esac

    local setting="cifar100_resnet18/${partition_label}/fedavg/seed${SEED}"
    local log_dir="${LOG_ROOT}/${setting}"
    local result="${log_dir}/${tag}.pkl"
    local route_dir="${log_dir}/${tag}_gradient_routes"
    local final_route="${route_dir}/round_$(printf '%04d' "$ROUNDS").json"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$result" ] && [ -s "$final_route" ]; then
        echo "[GPU ${gpu_id}] skip: ${partition_label} | ${tag}"
        return 0
    fi

    local command=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data
        --model resnet18_byot --alg fedbyot
        --n_clients 100 --sample_fraction 0.1
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS"
        --lr "$LR" --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --seed "$SEED" --device "cuda:${gpu_id}"
        --partition "$partition_mode" --beta "$beta"
        --min_require_size "$MIN_REQUIRE_SIZE"
        --byot_active_branches 1,2,3
        --byot_branch_loss_reduction sum
        --byot_branch_objective "$objective" --byot_alpha "$alpha"
        --byot_beta 0.0
        --temperature "$temperature"
        --byot_branch_kd_teacher_temperature "$temperature"
        --byot_branch_kd_student_temperature "$temperature"
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_proxy_temperature 1.0
        --log_gradient_routes
        --gradient_route_probe_interval "$PROBE_INTERVAL"
        --gradient_route_probe_batch_size 64
        --gradient_route_probe_client_count 0
        --gradient_route_local_max_batches 0
        --gradient_route_global_max_batches 0
        --gradient_route_temperature "$temperature"
        --gradient_route_branch_reduction sum
        --gradient_route_output_dir "$route_dir"
        --logdir "$LOG_ROOT" --log_file_name "${setting}/${tag}"
    )

    echo "[GPU ${gpu_id}] start: ${partition_label} | ${tag} | min=64"
    if [ "$DRY_RUN" = "1" ]; then
        printf '[dry-run][GPU %s] ' "$gpu_id"
        printf '%q ' "${command[@]}"
        printf '\n'
        return 0
    fi

    "${command[@]}" > "${log_dir}/${tag}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${partition_label} | ${tag}"
}

run_queue() {
    local gpu_id=$1
    shift
    local job
    for job in "$@"; do
        [ -z "$job" ] && continue
        run_job "$gpu_id" "$job"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done
for ((i = 0; i < ${#JOBS[@]}; i++)); do
    QUEUES[$((i % NUM_GPUS))]+="${JOBS[$i]}"$'\n'
done

echo "========== CE/KD Temperature × Partition Control =========="
echo "GPUs=${GPUS[*]} | jobs=${#JOBS[@]} | seed=${SEED}"
echo "CIFAR-100 / ResNet18-BYOT / FedAvg / R=${ROUNDS} / E=${LOCAL_EPOCHS}"
echo "feature_beta=0 | min_require_size=64 | proxy_temperature=1"
echo "Estimated wall time: 4 GPUs ~3-3.5 h; 2 GPUs ~6-7 h"
echo "Log root: ${LOG_ROOT}"

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
    echo "At least one control run failed; inspect *_terminal.log." >&2
    exit "$status"
fi

echo "All CE/KD temperature controls complete."
