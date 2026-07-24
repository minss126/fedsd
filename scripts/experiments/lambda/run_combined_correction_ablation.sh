#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1\" bash $0"
    echo
    echo "Optional overrides:"
    echo "  ENVS_OVERRIDE=\"iid beta_0.5 beta_0.3 beta_0.1 noniid_grouping\""
    echo "  METHODS_OVERRIDE=\"combined_norm0p75_p2 combined_residual0p5_c0p75_p2 combined_floor0p5_p2\""
    echo "  LOG_ROOT=\"logs/lambda/selective/logs_combined_correction_ablation\""
    echo "  USE_WANDB=0"
    exit 0
fi

# Ablation for corrected combined sample/client lambda control.
#
# Existing role-split combined used:
#   lambda_{k,i,t} = lambda_round(t) * p_T(y_i|x_i) * b_k^p
#
# These candidates keep the sample-level label-probability term but make the
# client-level correction milder and closer to a redistribution around lambda=1.

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
BASE_LAMBDA="${BASE_LAMBDA:-1.00}"
PARTITION_GROUPS="${PARTITION_GROUPS:-8}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/selective/logs_combined_correction_ablation}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EXTRA_FLAGS="${EXTRA_FLAGS:---log_client_group_lambda}"

ENVS=(${ENVS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1 noniid_grouping})
METHODS=(${METHODS_OVERRIDE:-combined_norm0p75_p2 combined_residual0p5_c0p75_p2 combined_floor0p5_p2})

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
        beta_0.1) echo "--partition noniid --beta 0.1" ;;
        beta_0.3) echo "--partition noniid --beta 0.3" ;;
        beta_0.5) echo "--partition noniid --beta 0.5" ;;
        noniid_grouping) echo "--partition noniid_grouping --partition_groups ${PARTITION_GROUPS}" ;;
        *) echo "Unknown env: $1" >&2; exit 1 ;;
    esac
}

sample_label_prob_flags() {
    echo "--byot_sample_proxy teacher_label_prob --alpha_min_scale 0.0"
}

client_entropy_flags() {
    local power=$1
    echo "--byot_client_skew_proxy prediction_entropy --byot_client_skew_power ${power} --byot_client_skew_min_scale 0.00"
}

method_flags() {
    case "$1" in
        combined_norm0p75_p2)
            # lambda_{k,i}=lambda*p_T(y_i|x_i)*(b_k^2/0.75)
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${BASE_LAMBDA} $(sample_label_prob_flags) $(client_entropy_flags 2.0) --byot_client_skew_correction_mode normalize --byot_client_skew_norm_value 0.75 --byot_client_skew_max_scale 2.00"
            ;;
        combined_norm0p90_p2)
            # More conservative normalization for high-entropy/IID-like settings.
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${BASE_LAMBDA} $(sample_label_prob_flags) $(client_entropy_flags 2.0) --byot_client_skew_correction_mode normalize --byot_client_skew_norm_value 0.90 --byot_client_skew_max_scale 2.00"
            ;;
        combined_residual0p5_c0p75_p2)
            # lambda_{k,i}=lambda*p_T(y_i|x_i)*(1+0.5*(b_k^2-0.75))
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${BASE_LAMBDA} $(sample_label_prob_flags) $(client_entropy_flags 2.0) --byot_client_skew_correction_mode residual --byot_client_skew_center 0.75 --byot_client_skew_gamma 0.50 --byot_client_skew_min_scale 0.50 --byot_client_skew_max_scale 1.50"
            ;;
        combined_residual1p0_c0p75_p2)
            # Stronger residual correction.
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${BASE_LAMBDA} $(sample_label_prob_flags) $(client_entropy_flags 2.0) --byot_client_skew_correction_mode residual --byot_client_skew_center 0.75 --byot_client_skew_gamma 1.00 --byot_client_skew_min_scale 0.25 --byot_client_skew_max_scale 1.75"
            ;;
        combined_floor0p5_p2)
            # lambda_{k,i}=lambda*p_T(y_i|x_i)*(0.5+0.5*b_k^2)
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${BASE_LAMBDA} $(sample_label_prob_flags) $(client_entropy_flags 2.0) --byot_client_skew_correction_mode multiply --byot_client_skew_min_scale 0.50"
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
    "${PYTHON_BIN}" main.py \
        --dataset cifar100 --datadir ./data --num_classes 100 \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${LR}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${env_name}/fedavg/${method_name}" \
        --byot_active_branches "1,2,3" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        ${selected_args} ${partition_args} ${EXTRA_FLAGS} ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${env_name} | ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")
    local job env_name method_name

    for job in "${jobs[@]}"; do
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

echo "========== Corrected Combined Lambda Ablation =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}, jobs=${job_count}"
echo "envs=${ENVS[*]}"
echo "methods=${METHODS[*]}"
echo "base_lambda=${BASE_LAMBDA}"
echo "log_root=${LOG_ROOT}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}, extra_flags=${EXTRA_FLAGS}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done

wait "${pids[@]}"
echo "Corrected combined lambda ablation complete (${job_count} jobs)"
