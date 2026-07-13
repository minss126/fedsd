#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Fixed single-branch KD-only lambda sweep.
#
# Goal:
#   After observing that a single active branch is often competitive with
#   or better than using all branches, fix one branch and sweep lambda_kd
#   more densely.
#
# Default:
#   dataset/backbone: CIFAR-100 / ResNet18-BYOT
#   base FL: FedAvg
#   active branch: B2
#   objective: kd_only
#   partitions: iid, beta_0.5, beta_0.3, beta_0.1
#   lambda_kd: 0, 0.1, 0.3, 0.7, 1, 3, 10
#
# Override examples:
#   BRANCH_OVERRIDE="1:B1" bash scripts/experiments/branch/run_fixed_single_branch_lambda_sweep.sh
#   GPUS_OVERRIDE="0 1" LAMBDAS_OVERRIDE="0.30:0p300 1.00:1p000 3.00:3p000" bash ...

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
        PYTHON_BIN="python"
    fi
fi

SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CIFAR_LR="${CIFAR_LR:-0.1}"
TEMP_VAL="${TEMP_VAL:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LOG_ROOT="${LOG_ROOT:-logs/branch/logs_fixed_single_branch_lambda_sweep}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SETTING="${SETTING_OVERRIDE:-cifar100_resnet18}"
PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})
BRANCH_PAIR="${BRANCH_OVERRIDE:-2:B2}"
LAMBDAS=(${LAMBDAS_OVERRIDE:-0.00:0p000 0.10:0p100 0.30:0p300 0.70:0p700 1.00:1p000 3.00:3p000 10.00:10p000})

BRANCH_ID="${BRANCH_PAIR%%:*}"
BRANCH_TAG="${BRANCH_PAIR##*:}"

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

dataset_flags() {
    local setting=$1
    case "$setting" in
        cifar100_resnet18)
            echo "--dataset cifar100 --datadir ./data --in_channels 3 --num_classes 100"
            ;;
        cifar10_resnet18)
            echo "--dataset cifar10 --datadir ./data --in_channels 3 --num_classes 10"
            ;;
        *)
            echo "Unknown setting: ${setting}" >&2
            exit 1
            ;;
    esac
}

model_flags() {
    local setting=$1
    case "$setting" in
        cifar100_resnet18|cifar10_resnet18)
            echo "--model resnet18_byot --alg fedbyot"
            ;;
        *)
            echo "Unknown setting: ${setting}" >&2
            exit 1
            ;;
    esac
}

partition_flags() {
    local partition_name=$1
    case "$partition_name" in
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
            echo "Unknown partition: ${partition_name}" >&2
            exit 1
            ;;
    esac
}

run_job() {
    local gpu_id=$1
    local partition_name=$2
    local lambda_val=$3
    local lambda_tag=$4

    local method_name="kd_only_${BRANCH_TAG}_lambda${lambda_tag}"
    local log_dir="${LOG_ROOT}/${SETTING}/${partition_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local data_args
    local model_args
    local partition_args

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${SETTING} | ${partition_name} | ${method_name}"
        return
    fi

    data_args="$(dataset_flags "$SETTING")"
    model_args="$(model_flags "$SETTING")"
    partition_args="$(partition_flags "$partition_name")"

    echo "[GPU ${gpu_id}] start: ${SETTING} | ${partition_name} | ${method_name} | active=${BRANCH_ID}"

    "${PYTHON_BIN}" main.py \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${CIFAR_LR}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${SETTING}/${partition_name}/fedavg/${method_name}" \
        ${data_args} ${partition_args} ${model_args} \
        --kd_conf_threshold 0.0 \
        --byot_active_branches "${BRANCH_ID}" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only \
        --byot_alpha "${lambda_val}" \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${SETTING} | ${partition_name} | ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")

    for job in "${jobs[@]}"; do
        [ -z "$job" ] && continue
        IFS='|' read -r partition_name lambda_val lambda_tag <<< "$job"
        run_job "$gpu_id" "$partition_name" "$lambda_val" "$lambda_tag"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for partition_name in "${PARTITIONS[@]}"; do
    for lambda_pair in "${LAMBDAS[@]}"; do
        lambda_val="${lambda_pair%%:*}"
        lambda_tag="${lambda_pair##*:}"
        gpu_idx=$((job_count % NUM_GPUS))
        QUEUES[$gpu_idx]+="${partition_name}|${lambda_val}|${lambda_tag}"$'\n'
        job_count=$((job_count + 1))
    done
done

echo "========== Fixed Single-Branch KD Lambda Sweep =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "setting=${SETTING}"
echo "partitions=${PARTITIONS[*]}"
echo "branch=${BRANCH_ID}:${BRANCH_TAG}"
echo "lambdas=${LAMBDAS[*]}"
echo "feature_beta=${FEATURE_BETA}, temp=${TEMP_VAL}, log_root=${LOG_ROOT}"
echo "skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"
echo "jobs=${job_count}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done

wait "${pids[@]}"
echo "Fixed single-branch KD lambda sweep complete (${job_count} jobs)"
