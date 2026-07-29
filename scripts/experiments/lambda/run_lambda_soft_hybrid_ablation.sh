#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1\" bash $0"
    echo
    echo "Runs two lambda ablations under the common 0 -> 2 / 250-round warmup."
    echo "Default environments: beta_0.5 beta_0.1"
    echo
    echo "Methods:"
    echo "  label_entropy_soft_tau0p80: client-wise r_k * soft_relax(h_k^2)"
    echo "  sample_labelprob_soft_predskew: sample-wise p_i * soft_relax(b_k^2)"
    exit 0
fi

# Definitions:
#   p_i = p_T(y_i | x_i), r_k = mean_i p_i
#   b_k = normalized entropy of mean teacher prediction distribution
#   h_k = normalized entropy of the local true-label distribution
#
# label_entropy_soft_tau0p80 (client-wise):
#   lambda_k = lambda_t * r_k * soft_relax(h_k^2)
#
# sample_labelprob_soft_predskew (sample + client ablation):
#   lambda_{k,i} = lambda_t * p_i * soft_relax(b_k^2)
#
# soft_relax keeps the p=2 scale for low reliability and smoothly moves it
# toward 1 above tau: (1-g)*b^2 + g, g=sigmoid((b-tau)/temperature).

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
SOFT_TAU="${SOFT_TAU:-0.80}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_lambda_soft_hybrid_ablation}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ENVS=(${ENVS_OVERRIDE:-beta_0.5 beta_0.1})
METHODS=(${METHODS_OVERRIDE:-label_entropy_soft_tau0p80 sample_labelprob_soft_predskew})

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

client_label_probability_flags() {
    echo "--byot_client_proxy teacher_label_prob --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00 --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0"
}

soft_skew_flags() {
    local proxy=$1
    echo "--byot_client_skew_proxy ${proxy} --byot_client_skew_power 2.0 --byot_client_skew_min_scale 0.00 --byot_client_skew_correction_mode soft_relax --byot_client_skew_soft_tau ${SOFT_TAU} --byot_client_skew_soft_temperature ${SOFT_TEMPERATURE}"
}

method_flags() {
    case "$1" in
        label_entropy_soft_tau0p80)
            echo "$(common_flags) $(client_label_probability_flags) $(soft_skew_flags label_entropy)"
            ;;
        sample_labelprob_soft_predskew)
            echo "$(common_flags) --byot_sample_proxy teacher_label_prob $(soft_skew_flags prediction_entropy)"
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
    local gpu_id=$1 env_name=$2 method_name=$3
    local log_dir="${LOG_ROOT}/${env_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local partition_args selected_args

    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | ${method_name}"
        return
    fi

    partition_args="$(env_flags "$env_name")"
    selected_args="$(method_flags "$method_name")"
    echo "[GPU ${gpu_id}] start: ${env_name} | ${method_name}"

    "$PYTHON_BIN" main.py \
        --dataset cifar100 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED" \
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT" \
        --log_file_name "${env_name}/fedavg/${method_name}" \
        --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA" \
        --temperature "$TEMP_VAL" \
        ${selected_args} ${partition_args} ${WANDB_FLAGS} \
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
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done

job_count=0
for env_name in "${ENVS[@]}"; do
    for method_name in "${METHODS[@]}"; do
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${env_name}|${method_name}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== Lambda Soft Hybrid Ablation =========="
echo "gpus=${GPUS[*]}, jobs=${job_count}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "envs=${ENVS[*]}"
echo "methods=${METHODS[*]}"
echo "lambda_schedule=linear 0 -> ${LAMBDA_MAX}, warmup_rounds=${WARMUP_ROUNDS}"
echo "soft_relax=tau ${SOFT_TAU}, temperature ${SOFT_TEMPERATURE}"
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
echo "Lambda soft-hybrid ablation complete (${job_count} jobs)"
