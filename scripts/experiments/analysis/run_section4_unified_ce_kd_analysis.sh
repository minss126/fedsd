#!/usr/bin/env bash

# Unified Section 4 evidence run.
# Each training trajectory writes final/branch accuracy, T=1 gradient routes,
# and post-local full logits for rare/frequent-client analysis.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

RUN_SET="${RUN_SET:-server4}"
GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPU ids provided. Set GPUS_OVERRIDE." >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x venv/bin/python ]; then PYTHON_BIN=venv/bin/python; else PYTHON_BIN=python3; fi
fi

SEED="${SEED_OVERRIDE:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
FEATURE_BETA="${FEATURE_BETA:-0.0}"
ENABLE_GRADIENT_ROUTES="${ENABLE_GRADIENT_ROUTES:-1}"
GRADIENT_ROUNDS="${GRADIENT_ROUNDS:-50,100,150,200,250,300,350,400,450,500}"
POSTLOCAL_ROUNDS="${POSTLOCAL_ROUNDS:-470,480,490}"
POSTLOCAL_SAMPLES_PER_CLASS="${POSTLOCAL_SAMPLES_PER_CLASS:-8}"
LOW_RATIO="${LOW_RATIO:-0.5}"
HIGH_RATIO="${HIGH_RATIO:-1.5}"
TEMPERATURE_TAG="$(printf '%s' "$TEMPERATURE" | tr '.' 'p')"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_section4_unified_no_feature_t${TEMPERATURE_TAG}_min64}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Four-GPU server handles IID and beta=0.3; two-GPU server handles beta=0.1.
# This creates exactly six trajectories (3 partitions x 2 objectives) per seed.
case "$RUN_SET" in
    server4)
        JOBS=(
            "iid|ce"
            "iid|kd"
            "beta_0.3|ce"
            "beta_0.3|kd"
        )
        ;;
    server2)
        JOBS=(
            "beta_0.1|ce"
            "beta_0.1|kd"
        )
        ;;
    all)
        JOBS=(
            "iid|ce"
            "iid|kd"
            "beta_0.3|ce"
            "beta_0.3|kd"
            "beta_0.1|ce"
            "beta_0.1|kd"
        )
        ;;
    *)
        echo "RUN_SET must be server4, server2, or all; got: $RUN_SET" >&2
        exit 2
        ;;
esac

partition_args() {
    case "$1" in
        iid)      echo "iid|0.5" ;;
        beta_0.3) echo "noniid|0.3" ;;
        beta_0.1) echo "noniid|0.1" ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

objective_args() {
    case "$1" in
        ce) echo "0.00|alpha0p00" ;;
        kd) echo "1.00|alpha1p00" ;;
        *) echo "Unknown objective: $1" >&2; return 1 ;;
    esac
}

run_job() {
    local gpu_id=$1 partition_label=$2 objective_label=$3
    local partition_mode beta alpha alpha_tag setting stem log_dir
    local result_path route_dir final_route logits_dir final_logits
    IFS='|' read -r partition_mode beta <<< "$(partition_args "$partition_label")"
    IFS='|' read -r alpha alpha_tag <<< "$(objective_args "$objective_label")"

    setting="cifar100_resnet18/${partition_label}/fedavg"
    stem="${alpha_tag}_seed${SEED}_client_pretrain_branch_freq"
    log_dir="${LOG_ROOT}/${setting}"
    result_path="${log_dir}/${stem}.pkl"
    route_dir="${log_dir}/${stem}_gradient_routes"
    final_route="${route_dir}/round_$(printf '%04d' "$ROUNDS").json"
    logits_dir="${log_dir}/${stem}_full_logits"
    final_logits="${logits_dir}/round_$(printf '%04d' "${POSTLOCAL_ROUNDS##*,}").pt"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && [ -s "$result_path" ] && [ -s "$final_logits" ]; then
        if [ "$ENABLE_GRADIENT_ROUTES" != "1" ] || [ -s "$final_route" ]; then
            echo "[GPU ${gpu_id}] skip complete: ${partition_label} | ${objective_label}"
            return 0
        fi
    fi

    command=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data --num_classes 100
        --model resnet18_byot --alg fedbyot
        --partition "$partition_mode" --beta "$beta"
        --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients 100 --sample_fraction 0.1
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS"
        --lr "$LR" --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --seed "$SEED"
        --device "cuda:${gpu_id}"
        --logdir "$LOG_ROOT" --log_file_name "${setting}/${stem}"
        --byot_active_branches 1,2,3
        --byot_branch_loss_reduction sum
        --byot_branch_objective blend --byot_alpha "$alpha"
        --byot_beta "$FEATURE_BETA"
        --temperature "$TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$TEMPERATURE"
        --byot_branch_kd_student_temperature "$TEMPERATURE"
        --log_postlocal_branch_distribution_stats
        --postlocal_ref_probe_rounds "$POSTLOCAL_ROUNDS"
        --postlocal_ref_samples_per_class "$POSTLOCAL_SAMPLES_PER_CLASS"
        --train_branch_freq_low_ratio "$LOW_RATIO"
        --train_branch_freq_high_ratio "$HIGH_RATIO"
        --save_postlocal_full_logits
    )
    if [ "$ENABLE_GRADIENT_ROUTES" = "1" ]; then
        command+=(
            --log_gradient_routes
            --gradient_route_probe_rounds "$GRADIENT_ROUNDS"
            --gradient_route_probe_batch_size 64
            --gradient_route_probe_client_count 0
            --gradient_route_local_max_batches 0
            --gradient_route_global_max_batches 0
            --gradient_route_temperature "$TEMPERATURE"
            --gradient_route_branch_reduction sum
            --gradient_route_output_dir "$route_dir"
        )
    fi

    echo "[GPU ${gpu_id}] start: ${partition_label} | ${objective_label} | seed=${SEED}"
    if [ "$DRY_RUN" = "1" ]; then
        printf '[dry-run] '
        printf '%q ' "${command[@]}"
        printf '\n'
        return 0
    fi
    "${command[@]}" > "${log_dir}/${stem}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${partition_label} | ${objective_label}"
}

run_queue() {
    local gpu_id=$1
    shift
    local job partition_label objective_label
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r partition_label objective_label <<< "$job"
        run_job "$gpu_id" "$partition_label" "$objective_label"
    done
}

echo "========== Unified Section 4 CE/KD analysis =========="
echo "run_set=${RUN_SET} | GPUs=${GPUS[*]} | jobs=${#JOBS[@]} | seed=${SEED}"
echo "CIFAR-100 / ResNet18-BYOT / FedAvg / R=${ROUNDS} / E=${LOCAL_EPOCHS} / participation=0.1"
echo "CE alpha=0 vs KD alpha=1 | T=${TEMPERATURE} | feature_beta=${FEATURE_BETA} | min_size=${MIN_REQUIRE_SIZE}"
if [ "$ENABLE_GRADIENT_ROUTES" = "1" ]; then
    echo "gradient completed rounds=${GRADIENT_ROUNDS}"
else
    echo "gradient routes=disabled (T=0.5 is the empirical accuracy/logit control)"
fi
echo "post-local full logits completed rounds=${POSTLOCAL_ROUNDS} | samples/class=${POSTLOCAL_SAMPLES_PER_CLASS}"
echo "frequency groups: rare < ${LOW_RATIO}x expected, frequent > ${HIGH_RATIO}x expected"
echo "canonical protocol: no paired_resnet_init, no paired_execution_rng, no preserve_byot_proxy_rng"
echo "log_root=${LOG_ROOT} | skip_existing=${SKIP_EXISTING}"

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done
for ((i = 0; i < ${#JOBS[@]}; i++)); do
    gpu_slot=$((i % NUM_GPUS))
    QUEUES[$gpu_slot]+="${JOBS[$i]}"$'\n'
done

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    [ -z "${QUEUES[$i]}" ] && continue
    mapfile -t queue_jobs <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
    echo "At least one run failed; inspect *_terminal.log." >&2
    exit "$status"
fi
echo "Unified Section 4 analysis complete (${#JOBS[@]} jobs)."
