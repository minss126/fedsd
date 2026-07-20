#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Selective-only vs. selective+adaptive lambda experiments.
#
# Main comparison:
#   selective only:
#       lambda_{k,i} = lambda_fixed * sample_score_{k,i}
#   selective + adaptive:
#       lambda_{k,i} = lambda_round(t) * client_score_k * sample_score_{k,i}
#
# The default target is noniid_grouping, because that setting exposes mixed
# head/tail clients in the same federation.

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
SELECTIVE_LAMBDA="${SELECTIVE_LAMBDA:-1.00}"
LAMBDA_MAX="${LAMBDA_MAX:-2.00}"
WARMUP="${WARMUP:-250}"
PARTITION_GROUPS="${PARTITION_GROUPS:-8}"
LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_selective_adaptive_grouping}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EXTRA_FLAGS="${EXTRA_FLAGS:---log_client_group_lambda}"

ENVS=(${ENVS_OVERRIDE:-noniid_grouping})
METHODS=(${METHODS_OVERRIDE:-selective_only_label_prob selective_adaptive_client_entropy__sample_label_prob_p1 selective_adaptive_client_entropy__sample_label_prob_p2 adaptive_client_entropy_only_p1 adaptive_client_entropy_only_p2})

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

round_schedule_flags() {
    echo "--byot_round_lambda_schedule linear --byot_round_lambda_min 0.00 --byot_round_lambda_warmup ${WARMUP}"
}

client_signal_flags() {
    local proxy=$1
    local power=$2
    echo "--byot_client_proxy ${proxy} --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00 --byot_client_alpha_mode multiply --byot_client_reliability_power ${power} --alpha_min_scale 0.0"
}

sample_signal_flags() {
    local proxy=$1
    echo "--byot_sample_proxy ${proxy} --alpha_min_scale 0.0"
}

method_flags() {
    case "$1" in
        selective_only_label_prob)
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${SELECTIVE_LAMBDA} $(sample_signal_flags teacher_label_prob)"
            ;;
        selective_only_label_prob_entropy)
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${SELECTIVE_LAMBDA} $(sample_signal_flags teacher_label_prob_entropy)"
            ;;
        selective_adaptive_label_prob_entropy__sample_label_prob)
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${LAMBDA_MAX} $(round_schedule_flags) $(client_signal_flags teacher_label_prob_entropy 1.0) $(sample_signal_flags teacher_label_prob)"
            ;;
        selective_adaptive_label_prob_entropy__sample_label_prob_entropy)
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${LAMBDA_MAX} $(round_schedule_flags) $(client_signal_flags teacher_label_prob_entropy 1.0) $(sample_signal_flags teacher_label_prob_entropy)"
            ;;
        selective_adaptive_client_entropy__sample_label_prob)
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${LAMBDA_MAX} $(round_schedule_flags) $(client_signal_flags teacher_label_prob 1.0) --byot_client_skew_proxy prediction_entropy --byot_client_skew_min_scale 0.00 --byot_client_skew_power 1.0 $(sample_signal_flags teacher_label_prob)"
            ;;
        selective_adaptive_client_entropy__sample_label_prob_p1)
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${LAMBDA_MAX} $(round_schedule_flags) --byot_client_skew_proxy prediction_entropy --byot_client_skew_min_scale 0.00 --byot_client_skew_power 1.0 $(sample_signal_flags teacher_label_prob)"
            ;;
        selective_adaptive_client_entropy__sample_label_prob_p2)
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${LAMBDA_MAX} $(round_schedule_flags) --byot_client_skew_proxy prediction_entropy --byot_client_skew_min_scale 0.00 --byot_client_skew_power 2.0 $(sample_signal_flags teacher_label_prob)"
            ;;
        selective_adaptive_client_entropy__sample_label_prob_entropy_p1)
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${LAMBDA_MAX} $(round_schedule_flags) --byot_client_skew_proxy prediction_entropy --byot_client_skew_min_scale 0.00 --byot_client_skew_power 1.0 $(sample_signal_flags teacher_label_prob_entropy)"
            ;;
        adaptive_client_entropy_only_p1)
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${LAMBDA_MAX} $(round_schedule_flags) --byot_client_skew_proxy prediction_entropy --byot_client_skew_min_scale 0.00 --byot_client_skew_power 1.0"
            ;;
        adaptive_client_entropy_only_p2)
            echo "--model resnet18_byot --alg fedbyot --byot_alpha ${LAMBDA_MAX} $(round_schedule_flags) --byot_client_skew_proxy prediction_entropy --byot_client_skew_min_scale 0.00 --byot_client_skew_power 2.0"
            ;;
        ce_fallback_conf0p60)
            echo "--model resnet18_byot --alg fedbyot_selective_ce_fallback --byot_alpha ${SELECTIVE_LAMBDA} --kd_conf_threshold 0.60"
            ;;
        ce_fallback_label_prob)
            echo "--model resnet18_byot --alg fedbyot_selective_ce_fallback --byot_alpha ${SELECTIVE_LAMBDA} $(sample_signal_flags teacher_label_prob)"
            ;;
        ce_fallback_adaptive_label_prob_entropy__sample_label_prob)
            echo "--model resnet18_byot --alg fedbyot_selective_ce_fallback --byot_alpha ${LAMBDA_MAX} $(round_schedule_flags) $(client_signal_flags teacher_label_prob_entropy 1.0) $(sample_signal_flags teacher_label_prob)"
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

echo "========== Selective / Adaptive Lambda Grouping =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}, jobs=${job_count}"
echo "envs=${ENVS[*]}"
echo "methods=${METHODS[*]}"
echo "selective_lambda=${SELECTIVE_LAMBDA}, lambda_max=${LAMBDA_MAX}, warmup=${WARMUP}"
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
echo "Selective/adaptive grouping experiments complete (${job_count} jobs)"
