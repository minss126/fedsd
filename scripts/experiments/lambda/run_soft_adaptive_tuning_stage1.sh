#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Stage 1 of the final adaptive-lambda tuning.  We retain the selected
# client-wise soft prediction-entropy method and screen only the two
# parameters whose interaction is most consequential for the KD loss:
#
#   lambda_k = lambda_t * r_k * [(1-g_k) b_k^2 + g_k]
#   g_k = sigmoid((b_k - tau) / T_soft)
#
# Fixed here: T_proxy=1, tau=0.85, T_soft=0.05, p=2, warm-up=250.
# Tuned here: T_KD x lambda_max, on beta=0.5 and beta=0.1.
#
# This script is invoked by the server-specific 4-GPU and 2-GPU wrappers.

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: TUNE_KD_TEMPERATURES=\"0.50\" GPUS_OVERRIDE=\"0 1\" bash $0"
    echo
    echo "Defaults: beta=0.5/0.1; lambda_max={1,2,3}; tau=0.85;"
    echo "          T_proxy=1; KD and branch temperatures are jointly tuned."
    echo "The 4-GPU/2-GPU wrappers provide complementary T_KD values."
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1})
NUM_GPUS=${#GPUS[@]}
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-0}"
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "No GPU ids provided. Set GPUS_OVERRIDE." >&2
    exit 1
fi
if [ "$EXPECTED_GPU_COUNT" -gt 0 ] && [ "$NUM_GPUS" -ne "$EXPECTED_GPU_COUNT" ]; then
    echo "This launcher requires ${EXPECTED_GPU_COUNT} GPUs; received ${NUM_GPUS}: ${GPUS[*]}" >&2
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
FEATURE_BETA="${FEATURE_BETA:-0.01}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_WARMUP="${LAMBDA_WARMUP:-250}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_soft_adaptive_tuning_stage1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ENVS=(${ENVS_OVERRIDE:-beta_0.5 beta_0.1})
KD_TEMPERATURE_VALUES=(${TUNE_KD_TEMPERATURES:-0.50})
LAMBDA_MAX_VALUES=(${LAMBDA_MAX_VALUES:-1.00 2.00 3.00})

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
    local kd_temp=$1 lambda_max=$2
    local kd_tag=${kd_temp/./p}
    local lambda_tag=${lambda_max/./p}
    echo "soft_b_tkd${kd_tag}_lmax${lambda_tag}_warm${LAMBDA_WARMUP}_tau${SOFT_TAU/./p}"
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1 env_name=$2 kd_temp=$3 lambda_max=$4
    local name log_dir log_file partition_args
    name="$(method_name "$kd_temp" "$lambda_max")"
    log_dir="${LOG_ROOT}/${env_name}/fedavg"
    log_file="${log_dir}/${name}.log"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | ${name}"
        return
    fi

    partition_args="$(env_flags "$env_name")"
    echo "[GPU ${gpu_id}] start: ${env_name} | ${name}"

    "$PYTHON_BIN" main.py \
        --dataset cifar100 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED" \
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT" \
        --log_file_name "${env_name}/fedavg/${name}" \
        --model resnet18_byot --alg fedbyot \
        --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA" \
        --byot_alpha "$lambda_max" \
        --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00 \
        --byot_round_lambda_warmup "$LAMBDA_WARMUP" \
        --temperature "$kd_temp" \
        --byot_branch_kd_teacher_temperature "$kd_temp" \
        --byot_branch_kd_student_temperature "$kd_temp" \
        --byot_proxy_temperature "$PROXY_TEMPERATURE" \
        --alpha_min_scale 0.0 \
        --byot_client_proxy teacher_label_prob \
        --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00 \
        --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0 \
        --byot_client_skew_proxy prediction_entropy \
        --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.00 \
        --byot_client_skew_correction_mode soft_relax \
        --byot_client_skew_soft_tau "$SOFT_TAU" \
        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE" \
        ${partition_args} ${WANDB_FLAGS} \
        > "${log_dir}/${name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${env_name} | ${name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local job env_name kd_temp lambda_max
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r env_name kd_temp lambda_max <<< "$job"
        run_job "$gpu_id" "$env_name" "$kd_temp" "$lambda_max"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done

job_count=0
for kd_temp in "${KD_TEMPERATURE_VALUES[@]}"; do
    for lambda_max in "${LAMBDA_MAX_VALUES[@]}"; do
        for env_name in "${ENVS[@]}"; do
            gpu_idx=$((job_count % NUM_GPUS))
            QUEUES[$gpu_idx]+="${env_name}|${kd_temp}|${lambda_max}"$'\n'
            job_count=$((job_count + 1))
        done
    done
done

echo "========== Soft-b Stage-1 Tuning =========="
echo "gpus=${GPUS[*]}, jobs=${job_count}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "envs=${ENVS[*]}, T_KD=${KD_TEMPERATURE_VALUES[*]}, lambda_max=${LAMBDA_MAX_VALUES[*]}"
echo "fixed: T_proxy=${PROXY_TEMPERATURE}, tau=${SOFT_TAU}, T_soft=${SOFT_TEMPERATURE}, p=${SKEW_POWER}, warmup=${LAMBDA_WARMUP}"
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
echo "Soft-b stage-1 tuning complete (${job_count} jobs)"
