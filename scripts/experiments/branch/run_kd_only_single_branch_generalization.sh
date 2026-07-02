#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# KD-only single-branch generalization sweep.
#
# This removes branch CE and keeps only one BYOT branch loss active:
#   L = CE_teacher + lambda_kd KD_branch_i + beta_feat L_feat_i
#
# Default pilot matrix:
#   dataset/backbone: CIFAR-100 / ResNet18-BYOT
#   partitions: iid, beta_0.5, beta_0.1
#   active branch: B1, B2, B3
#   lambda_kd: 0, 0.1, 3
#
# Total default jobs: 1 setting x 3 partitions x 3 branches x 3 lambdas = 27.
# Jobs are distributed over the provided GPU ids; each GPU runs its queue
# sequentially, while queues run in parallel.
#
# To expand later:
#   SETTINGS_OVERRIDE="cifar100_resnet18 cifar100_mobilenet cifar10_resnet18" \
#   LAMBDAS_OVERRIDE="0.00:0p000 0.01:0p010 0.10:0p100 0.30:0p300 1.00:1p000 3.00:3p000 10.00:10p000" \
#   bash scripts/experiments/branch/run_kd_only_single_branch_generalization.sh

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
TEMP_VAL="${TEMP_VAL:-0.5}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
LOG_ROOT="${LOG_ROOT:-logs_kd_only_single_branch_generalization}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SETTINGS=(${SETTINGS_OVERRIDE:-cifar100_resnet18})
PARTITIONS=(${PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.1})
BRANCHES=(${BRANCHES_OVERRIDE:-1:B1 2:B2 3:B3})
LAMBDAS=(${LAMBDAS_OVERRIDE:-0.00:0p000 0.10:0p100 3.00:3p000})

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
        cifar10_resnet18|cifar10_mobilenet)
            echo "--dataset cifar10 --datadir ./data"
            ;;
        cifar100_resnet18|cifar100_mobilenet)
            echo "--dataset cifar100 --datadir ./data"
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
        cifar10_resnet18|cifar100_resnet18)
            echo "--model resnet18_byot --alg fedbyot"
            ;;
        cifar10_mobilenet|cifar100_mobilenet)
            echo "--model mobilenet_byot --last_fc --alg fedbyot"
            ;;
        *)
            echo "Unknown setting: ${setting}" >&2
            exit 1
            ;;
    esac
}

dataset_lr() {
    local setting=$1
    case "$setting" in
        cifar10_resnet18|cifar10_mobilenet|cifar100_resnet18|cifar100_mobilenet)
            echo "${CIFAR_LR:-0.1}"
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
    local setting=$2
    local partition_name=$3
    local branch_id=$4
    local branch_tag=$5
    local lambda_val=$6
    local lambda_tag=$7

    local method_name="kd_only_${branch_tag}_lambda${lambda_tag}"
    local log_dir="${LOG_ROOT}/${setting}/${partition_name}/fedavg"
    local log_file="${log_dir}/${method_name}.log"
    local data_args
    local model_args
    local partition_args
    local learning_rate

    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] ${setting} | ${partition_name} | ${method_name}"
        return
    fi

    data_args="$(dataset_flags "$setting")"
    model_args="$(model_flags "$setting")"
    partition_args="$(partition_flags "$partition_name")"
    learning_rate="$(dataset_lr "$setting")"

    echo "[GPU ${gpu_id}] start: ${setting} | ${partition_name} | ${method_name} | active=${branch_id}"

    "${PYTHON_BIN}" main.py \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "${LOCAL_EPOCHS}" --lr "${learning_rate}" --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --round "${ROUNDS}" --seed "${SEED}" \
        --device "cuda:${gpu_id}" \
        --logdir "${LOG_ROOT}" \
        --log_file_name "${setting}/${partition_name}/fedavg/${method_name}" \
        ${data_args} ${partition_args} ${model_args} \
        --kd_conf_threshold 0.0 \
        --byot_active_branches "${branch_id}" \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective kd_only \
        --byot_alpha "${lambda_val}" \
        --byot_beta "${FEATURE_BETA}" \
        --temperature "${TEMP_VAL}" \
        ${WANDB_FLAGS} \
        > "${log_dir}/${method_name}_terminal.log" 2>&1

    echo "[GPU ${gpu_id}] complete: ${setting} | ${partition_name} | ${method_name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local jobs=("$@")

    for job in "${jobs[@]}"; do
        [ -z "$job" ] && continue
        IFS='|' read -r setting partition_name branch_id branch_tag lambda_val lambda_tag <<< "$job"
        run_job "$gpu_id" "$setting" "$partition_name" "$branch_id" "$branch_tag" "$lambda_val" "$lambda_tag"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do
    QUEUES[$i]=""
done

job_count=0
for setting in "${SETTINGS[@]}"; do
    for partition_name in "${PARTITIONS[@]}"; do
        for branch_pair in "${BRANCHES[@]}"; do
            branch_id="${branch_pair%%:*}"
            branch_tag="${branch_pair##*:}"
            for lambda_pair in "${LAMBDAS[@]}"; do
                lambda_val="${lambda_pair%%:*}"
                lambda_tag="${lambda_pair##*:}"
                gpu_idx=$((job_count % NUM_GPUS))
                QUEUES[$gpu_idx]+="${setting}|${partition_name}|${branch_id}|${branch_tag}|${lambda_val}|${lambda_tag}"$'\n'
                job_count=$((job_count + 1))
            done
        done
    done
done

echo "========== KD-Only Single-Branch Generalization Sweep =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "settings=${SETTINGS[*]}"
echo "partitions=${PARTITIONS[*]}"
echo "branches=${BRANCHES[*]}"
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
echo "KD-only single-branch generalization sweep complete (${job_count} jobs)"
