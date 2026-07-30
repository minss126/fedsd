#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1\" bash $0"
    echo
    echo "Tunes the client-wise soft prediction-entropy adaptive method on a"
    echo "compact global grid.  One setting is selected by mean last-30 accuracy"
    echo "over the default dev environments (beta_0.5 and beta_0.1)."
    echo
    echo "Grid defaults:"
    echo "  LAMBDA_MAX_VALUES=\"1.00 2.00 3.00\""
    echo "  WARMUP_VALUES=\"125 250\""
    echo "  TAU_VALUES=\"0.75 0.80 0.85\""
    echo "  SOFT_TEMPERATURE=0.05, SKEW_POWER=2.0"
    echo
    echo "Optional overrides:"
    echo "  ENVS_OVERRIDE=\"beta_0.5 beta_0.1\""
    echo "  SEED=0 USE_WANDB=0 SKIP_EXISTING=1"
    exit 0
fi

# Proposed method (client-wise only):
#   lambda_k = lambda_t * r_k * s_soft(b_k)
#   r_k = mean_i p_T(y_i | x_i)
#   b_k = H(mean_i p_T(. | x_i)) / log(C)
#   s_soft(b) = (1-g)*b^p + g, g = sigmoid((b-tau)/temperature)
#
# This script deliberately uses one common hyperparameter setting for every
# partition.  Do not choose a separate best setting for each beta.

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
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_soft_adaptive_hparam_grid}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ENVS=(${ENVS_OVERRIDE:-beta_0.5 beta_0.1})
LAMBDA_MAX_VALUES=(${LAMBDA_MAX_VALUES:-1.00 2.00 3.00})
WARMUP_VALUES=(${WARMUP_VALUES:-125 250})
TAU_VALUES=(${TAU_VALUES:-0.75 0.80 0.85})

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
        beta_0.3) echo "--partition noniid --beta 0.3" ;;
        beta_0.1) echo "--partition noniid --beta 0.1" ;;
        *) echo "Unknown environment: $1" >&2; exit 1 ;;
    esac
}

method_name() {
    local lambda_max=$1 warmup=$2 tau=$3
    local lambda_tag=${lambda_max/./p}
    local tau_tag=${tau/./p}
    echo "soft_predentropy_lmax${lambda_tag}_warm${warmup}_tau${tau_tag}"
}

method_flags() {
    local lambda_max=$1 warmup=$2 tau=$3
    echo "--model resnet18_byot --alg fedbyot --byot_alpha ${lambda_max} --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00 --byot_round_lambda_warmup ${warmup} --alpha_min_scale 0.0 --byot_client_proxy teacher_label_prob --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00 --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0 --byot_client_skew_proxy prediction_entropy --byot_client_skew_power ${SKEW_POWER} --byot_client_skew_min_scale 0.00 --byot_client_skew_correction_mode soft_relax --byot_client_skew_soft_tau ${tau} --byot_client_skew_soft_temperature ${SOFT_TEMPERATURE}"
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1 env_name=$2 lambda_max=$3 warmup=$4 tau=$5
    local name log_dir log_file partition_args selected_args
    name="$(method_name "$lambda_max" "$warmup" "$tau")"
    log_dir="${LOG_ROOT}/${env_name}/fedavg"
    log_file="${log_dir}/${name}.log"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | ${name}"
        return
    fi

    partition_args="$(env_flags "$env_name")"
    selected_args="$(method_flags "$lambda_max" "$warmup" "$tau")"
    echo "[GPU ${gpu_id}] start: ${env_name} | ${name}"

    "$PYTHON_BIN" main.py \
        --dataset cifar100 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED" \
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT" \
        --log_file_name "${env_name}/fedavg/${name}" \
        --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA" \
        --temperature "$TEMP_VAL" \
        ${selected_args} ${partition_args} ${WANDB_FLAGS} \
        > "${log_dir}/${name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${env_name} | ${name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local job env_name lambda_max warmup tau
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r env_name lambda_max warmup tau <<< "$job"
        run_job "$gpu_id" "$env_name" "$lambda_max" "$warmup" "$tau"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done

job_count=0
for lambda_max in "${LAMBDA_MAX_VALUES[@]}"; do
    for warmup in "${WARMUP_VALUES[@]}"; do
        for tau in "${TAU_VALUES[@]}"; do
            for env_name in "${ENVS[@]}"; do
                gpu_idx=$((job_count % NUM_GPUS))
                QUEUES[$gpu_idx]+="${env_name}|${lambda_max}|${warmup}|${tau}"$'\n'
                job_count=$((job_count + 1))
            done
        done
    done
done

echo "========== Soft Adaptive Hyperparameter Grid =========="
echo "gpus=${GPUS[*]}, jobs=${job_count}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "dev_envs=${ENVS[*]}"
echo "lambda_max=${LAMBDA_MAX_VALUES[*]}"
echo "warmup=${WARMUP_VALUES[*]}"
echo "tau=${TAU_VALUES[*]}, soft_temperature=${SOFT_TEMPERATURE}, skew_power=${SKEW_POWER}"
echo "selection=one global setting by mean last-30 accuracy across dev_envs"
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
echo "Soft adaptive hyperparameter grid complete (${job_count} jobs)"
