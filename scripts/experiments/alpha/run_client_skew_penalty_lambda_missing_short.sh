#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Re-run the missing client-skew adaptive lambda jobs with short W&B job_type
# names. W&B rejects job_type strings longer than 64 chars, so keep method
# basenames compact.

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

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LAMBDA_MAX="${LAMBDA_MAX:-3.00}"
WARMUP="${WARMUP:-250}"
SKEW_MIN_SCALE="${SKEW_MIN_SCALE:-0.00}"
SKEW_POWER="${SKEW_POWER:-1.00}"
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_client_skew_penalty_lambda_pilot}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Primary missing/important runs:
#   - beta_0.1 entropy-skew: needed for severe non-IID conclusion.
#   - concentration-skew for all partitions: previous long names failed.
JOBS=(
    "beta_0.1|teacher_label_prob_entropy_branch_js|linear|label_entropy|rc_ent_b01_w${WARMUP}"
    "iid|teacher_label_prob_entropy_branch_js|linear|max_concentration|rc_conc_iid_w${WARMUP}"
    "beta_0.5|teacher_label_prob_entropy_branch_js|linear|max_concentration|rc_conc_b05_w${WARMUP}"
    "beta_0.3|teacher_label_prob_entropy_branch_js|linear|max_concentration|rc_conc_b03_w${WARMUP}"
    "beta_0.1|teacher_label_prob_entropy_branch_js|linear|max_concentration|rc_conc_b01_w${WARMUP}"
)

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

env_flags() {
    local env_name=$1
    case "$env_name" in
        iid)
            echo "--partition iid"
            ;;
        beta_0.1)
            echo "--partition noniid --beta 0.1"
            ;;
        beta_0.3)
            echo "--partition noniid --beta 0.3"
            ;;
        beta_0.5)
            echo "--partition noniid --beta 0.5"
            ;;
        *)
            echo "Unknown env: ${env_name}" >&2
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
    local proxy=$3
    local schedule=$4
    local skew_proxy=$5
    local method_name=$6
    local log_dir="${LOG_ROOT}/${env_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local flags
    local schedule_flags=""

    flags="$(env_flags "$env_name")"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${env_name} | ${method_name}"
        return
    fi

    if [ "$schedule" != "none" ]; then
        schedule_flags="--byot_round_lambda_schedule ${schedule} --byot_round_lambda_min 0.00 --byot_round_lambda_warmup ${WARMUP}"
    fi

    echo "[GPU ${gpu_id}] start: ${env_name} | ${method_name} | proxy=${proxy} | skew=${skew_proxy}"

    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${LR}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/fedavg/${method_name}" \
        --model resnet18_byot --alg fedbyot \
        --kd_conf_threshold 0.0 \
        --byot_active_branches "1,2,3" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only \
        --byot_alpha "${LAMBDA_MAX}" \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        --byot_client_proxy "${proxy}" \
        --byot_client_alpha_min 0.00 \
        --byot_client_alpha_max 1.00 \
        --byot_client_alpha_mode multiply \
        --byot_client_skew_proxy "${skew_proxy}" \
        --byot_client_skew_min_scale "${SKEW_MIN_SCALE}" \
        --byot_client_skew_power "${SKEW_POWER}" \
        ${schedule_flags} \
        ${flags} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${env_name} | ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")

    for job in "${jobs[@]}"; do
        [ -z "$job" ] && continue
        IFS='|' read -r env_name proxy schedule skew_proxy method_name <<< "$job"
        run_job "$gpu_id" "$env_name" "$proxy" "$schedule" "$skew_proxy" "$method_name"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for job in "${JOBS[@]}"; do
    gpu_idx=$((job_count % NUM_GPUS))
    QUEUES[$gpu_idx]+="${job}"$'\n'
    job_count=$((job_count + 1))
done

echo "========== Client Skew Penalty Missing Jobs =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, seed=${SEED}, jobs=${job_count}"
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
echo "Client skew penalty missing jobs complete (${job_count} jobs)"
