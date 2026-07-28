#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1\" bash $0"
    echo
    echo "Compares p=2 adaptive-lambda signal alternatives with lambda warmup 0 -> 2."
    echo "Default environments: beta_0.5 beta_0.1."
    echo
    echo "Optional overrides:"
    echo "  ENVS_OVERRIDE=\"iid beta_0.5 beta_0.1\""
    echo "  METHODS_OVERRIDE=\"adaptive_entropy_soft_tau0p80 adaptive_label_entropy_p2 adaptive_label_js_global_p2 adaptive_prediction_js_global_p2 adaptive_teacher_correctness_p2\""
    echo "  LOG_ROOT=logs/lambda/adaptive/logs_adaptive_signal_alternatives"
    echo "  SKIP_EXISTING=0 USE_WANDB=0"
    exit 0
fi

# Every method uses:
#   lambda_t = linear warmup(0 -> LAMBDA_MAX, WARMUP_ROUNDS)
#   lambda_k = lambda_t * r_k * s_k
#
# r_k is normally mean teacher label probability.  The final method replaces
# it with teacher correctness.  s_k is a p=2 client-skew scale.
#
# Methods:
#   adaptive_entropy_p2:              r_k * H(mean teacher prediction)^2
#   adaptive_entropy_soft_tau0p80:    r_k * soft_relax(H(mean prediction)^2)
#   adaptive_label_entropy_p2:        r_k * H(local true-label distribution)^2
#   adaptive_label_js_global_p2:      r_k * (1 - JS(local labels, global labels))^2
#   adaptive_prediction_js_global_p2: r_k * (1 - JS(mean prediction, global labels))^2
#   adaptive_teacher_correctness_p2:  correctness_k * H(mean prediction)^2

GPUS=(${GPUS_OVERRIDE:-0 1})
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

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LAMBDA_MAX="${LAMBDA_MAX:-2.00}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-250}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_adaptive_signal_alternatives}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EXTRA_FLAGS="${EXTRA_FLAGS:---log_client_group_lambda}"

# beta=0.5 tests the mild-skew regime targeted by soft_relax; beta=0.1 tests
# whether a candidate retains the original adaptive method's severe-skew gain.
ENVS=(${ENVS_OVERRIDE:-beta_0.5 beta_0.1})
# adaptive_entropy_p2 is the already-completed existing-adaptive baseline in
# logs_lambda_granularity_fair_warmup, so it remains selectable via
# METHODS_OVERRIDE but is not re-run by default.
METHODS=(${METHODS_OVERRIDE:-adaptive_entropy_soft_tau0p80 adaptive_label_entropy_p2 adaptive_label_js_global_p2 adaptive_prediction_js_global_p2 adaptive_teacher_correctness_p2})

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

env_flags() {
    case "$1" in
        iid) echo "--partition iid" ;;
        beta_0.5) echo "--partition noniid --beta 0.5" ;;
        beta_0.1) echo "--partition noniid --beta 0.1" ;;
        *) echo "Unknown environment: $1" >&2; exit 1 ;;
    esac
}

common_flags() {
    echo "--model resnet18_byot --alg fedbyot --byot_alpha ${LAMBDA_MAX} --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00 --byot_round_lambda_warmup ${WARMUP_ROUNDS} --alpha_min_scale 0.0"
}

client_reliability_flags() {
    local proxy=$1
    echo "--byot_client_proxy ${proxy} --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00 --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0"
}

skew_flags() {
    local proxy=$1
    local correction_mode=${2:-multiply}
    echo "--byot_client_skew_proxy ${proxy} --byot_client_skew_min_scale 0.00 --byot_client_skew_power 2.0 --byot_client_skew_correction_mode ${correction_mode}"
}

method_flags() {
    case "$1" in
        adaptive_entropy_p2)
            echo "$(common_flags) $(client_reliability_flags teacher_label_prob) $(skew_flags prediction_entropy)"
            ;;
        adaptive_entropy_soft_tau0p80)
            echo "$(common_flags) $(client_reliability_flags teacher_label_prob) $(skew_flags prediction_entropy soft_relax) --byot_client_skew_soft_tau 0.80 --byot_client_skew_soft_temperature 0.05"
            ;;
        adaptive_label_entropy_p2)
            echo "$(common_flags) $(client_reliability_flags teacher_label_prob) $(skew_flags label_entropy)"
            ;;
        adaptive_label_js_global_p2)
            echo "$(common_flags) $(client_reliability_flags teacher_label_prob) $(skew_flags label_js_global) --byot_client_skew_label_smoothing 1.0"
            ;;
        adaptive_prediction_js_global_p2)
            echo "$(common_flags) $(client_reliability_flags teacher_label_prob) $(skew_flags prediction_js_global)"
            ;;
        adaptive_teacher_correctness_p2)
            echo "$(common_flags) $(client_reliability_flags teacher_correctness) $(skew_flags prediction_entropy)"
            ;;
        *)
            echo "Unknown method: $1" >&2
            exit 1
            ;;
    esac
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1
    local env_name=$2
    local method_name=$3
    local log_dir="${LOG_ROOT}/${env_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local partition_args
    local selected_args

    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | ${method_name}"
        return
    fi

    partition_args="$(env_flags "$env_name")"
    selected_args="$(method_flags "$method_name")"
    echo "[GPU ${gpu_id}] start: ${env_name} | ${method_name}"

    "$PYTHON_BIN" main.py \
        --dataset cifar100 --datadir ./data --num_classes 100 \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED" \
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT" \
        --log_file_name "${env_name}/fedavg/${method_name}" \
        --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA" \
        --temperature "$TEMP_VAL" \
        ${selected_args} ${partition_args} ${EXTRA_FLAGS} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${env_name} | ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local job env_name method_name
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r env_name method_name <<< "$job"
        run_job "$gpu_id" "$env_name" "$method_name"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for env_name in "${ENVS[@]}"; do
    for method_name in "${METHODS[@]}"; do
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${env_name}|${method_name}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== Adaptive Lambda Signal Alternatives =========="
echo "gpus=${GPUS[*]}, jobs=${job_count}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "envs=${ENVS[*]}"
echo "methods=${METHODS[*]}"
echo "lambda_schedule=linear 0 -> ${LAMBDA_MAX}, warmup_rounds=${WARMUP_ROUNDS}"
echo "log_root=${LOG_ROOT}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done

wait "${pids[@]}"
echo "Adaptive lambda signal-alternative experiments complete (${job_count} jobs)"
