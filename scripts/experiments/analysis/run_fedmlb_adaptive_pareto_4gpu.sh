#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-4}"
NUM_GPUS="${#GPUS[@]}"
if [[ "$NUM_GPUS" -ne "$EXPECTED_GPU_COUNT" ]]; then
    echo "This launcher expects exactly ${EXPECTED_GPU_COUNT} GPUs; received: ${GPUS[*]}" >&2
    exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

DATASETS=(${DATASETS_OVERRIDE:-cifar100})
SEEDS=(${SEEDS_OVERRIDE:-0 1 2})
PARTITION="${PARTITION:-iid}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-0.1}"
LR_GAMMA="${LR_GAMMA:-0.998}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
MOMENTUM="${MOMENTUM:-0.9}"

# Official FedMLB objective coefficients.
FEDMLB_MAIN_CE="${FEDMLB_MAIN_CE:-1.0}"
FEDMLB_HYBRID_CE="${FEDMLB_HYBRID_CE:-1.0}"
FEDMLB_HYBRID_KD="${FEDMLB_HYBRID_KD:-1.0}"
FEDMLB_TEMPERATURE="${FEDMLB_TEMPERATURE:-1.0}"
FEDMLB_GRAD_CLIP="${FEDMLB_GRAD_CLIP:-10.0}"

# Selected reliability-aware adaptive KD configuration.
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-$((ROUNDS / 2))}"

LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_fedmlb_adaptive_pareto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis/fedmlb_adaptive_pareto}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
USE_WANDB="${USE_WANDB:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dxfl_fedmlb_pareto_matplotlib}"
mkdir -p "$MPLCONFIGDIR" "$LOG_ROOT" "$OUTPUT_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_names=()
    for gpu_id in "${GPUS[@]}"; do
        if gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$gpu_id" 2>/dev/null)"; then
            gpu_names+=("${gpu_name%%$'\n'*}")
        else
            gpu_names=()
            break
        fi
    done
    if [[ "${#gpu_names[@]}" -eq "$NUM_GPUS" ]]; then
        unique_gpu_count="$(printf '%s\n' "${gpu_names[@]}" | sort -u | wc -l)"
    else
        unique_gpu_count=0
    fi
    if [[ "$unique_gpu_count" -gt 1 ]]; then
        echo "WARNING: selected GPUs are heterogeneous: ${gpu_names[*]}" >&2
        echo "Accuracy remains comparable, but wall-clock Pareto curves are hardware-confounded." >&2
    fi
fi

dataset_flags() {
    case "$1" in
        cifar10) echo "--dataset cifar10 --datadir ${CIFAR_DATADIR:-./data}" ;;
        cifar100) echo "--dataset cifar100 --datadir ${CIFAR_DATADIR:-./data}" ;;
        tinyimagenet) echo "--dataset tinyimagenet --datadir ${TINYIMAGENET_DATADIR:-./data/tiny-imagenet-200}" ;;
        *) echo "Unknown dataset: $1" >&2; return 1 ;;
    esac
}

partition_flags() {
    case "$PARTITION" in
        iid) echo "--partition iid" ;;
        beta_0.5) echo "--partition noniid --beta 0.5" ;;
        beta_0.3) echo "--partition noniid --beta 0.3" ;;
        beta_0.1) echo "--partition noniid --beta 0.1" ;;
        *) echo "Unknown PARTITION=${PARTITION}" >&2; return 1 ;;
    esac
}

target_accuracies() {
    case "$1" in
        cifar10) echo "80,85,90" ;;
        cifar100) echo "60,65,68,70" ;;
        tinyimagenet) echo "35,40,45" ;;
    esac
}

has_completed_result() {
    local log_file="$1" result_file="$2"
    [[ -f "$log_file" && -f "$result_file" ]] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id="$1" dataset="$2" method="$3" seed="$4"
    local log_dir="${LOG_ROOT}/${dataset}/${PARTITION}/${method}"
    local name="seed${seed}"
    local log_file="${log_dir}/${name}.log"
    local result_file="${log_dir}/${name}.pkl"
    local -a cmd data_args partition_args wandb_args
    mkdir -p "$log_dir"

    if [[ "$SKIP_EXISTING" == "1" ]] \
        && has_completed_result "$log_file" "$result_file"; then
        echo "[skip] ${dataset} ${method} seed=${seed}"
        return
    fi

    read -r -a data_args <<< "$(dataset_flags "$dataset")"
    read -r -a partition_args <<< "$(partition_flags)"
    wandb_args=()
    if [[ "$USE_WANDB" == "1" ]]; then
        wandb_args=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
        if [[ -n "${WANDB_ENTITY:-}" ]]; then
            wandb_args+=(--wandb_entity "$WANDB_ENTITY")
        fi
    fi

    cmd=(
        "$PYTHON_BIN" main.py
        "${data_args[@]}" "${partition_args[@]}"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$seed" --device "cuda:${gpu_id}"
        --optimizer sgd --lr "$LR" --momentum "$MOMENTUM" --reg "$WEIGHT_DECAY"
        --scheduler round --schedule_round 1 --lr_gamma "$LR_GAMMA"
        --paired_resnet_init --paired_execution_rng
        --logdir "$LOG_ROOT"
        --log_file_name "${dataset}/${PARTITION}/${method}/${name}"
    )

    if [[ "$method" == "fedmlb" ]]; then
        cmd+=(
            --model resnet18 --alg fedmlb
            --fedmlb_main_ce_weight "$FEDMLB_MAIN_CE"
            --fedmlb_hybrid_ce_weight "$FEDMLB_HYBRID_CE"
            --fedmlb_hybrid_kd_weight "$FEDMLB_HYBRID_KD"
            --fedmlb_temperature "$FEDMLB_TEMPERATURE"
            --fedmlb_grad_clip "$FEDMLB_GRAD_CLIP"
            --fedmlb_select_level -1
        )
    elif [[ "$method" == "adaptive" ]]; then
        cmd+=(
            --model resnet18_byot --alg fedbyot
            --byot_active_branches "1,2,3"
            --byot_branch_loss_reduction sum
            --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
            --byot_alpha "$LAMBDA_MAX"
            --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00
            --byot_round_lambda_warmup "$WARMUP_ROUNDS"
            --temperature "$KD_TEMPERATURE"
            --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
            --byot_proxy_temperature "$PROXY_TEMPERATURE"
            --preserve_byot_proxy_rng
            --alpha_min_scale 0.0
            --byot_client_proxy teacher_label_prob
            --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00
            --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
            --byot_client_skew_proxy prediction_entropy
            --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.00
            --byot_client_skew_correction_mode soft_relax
            --byot_client_skew_soft_tau "$SOFT_TAU"
            --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
        )
    else
        echo "Unknown method: ${method}" >&2
        return 1
    fi
    cmd+=("${wandb_args[@]}")

    echo "[GPU ${gpu_id}] start ${dataset} ${method} seed=${seed}"
    "${cmd[@]}" > "${log_dir}/${name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete ${dataset} ${method} seed=${seed}"
}

run_queue() {
    local gpu_id="$1" queue="$2"
    local dataset method seed
    while IFS='|' read -r dataset method seed; do
        [[ -z "$dataset" ]] && continue
        run_job "$gpu_id" "$dataset" "$method" "$seed"
    done <<< "$queue"
}

echo "========== FedMLB vs adaptive accuracy-cost comparison =========="
echo "gpus=${GPUS[*]}"
echo "datasets=${DATASETS[*]}, partition=${PARTITION}, seeds=${SEEDS[*]}"
echo "K=${NUM_CLIENTS}, C=${SAMPLE_FRACTION}, rounds=${ROUNDS}, E=${LOCAL_EPOCHS}"
echo "batch=${BATCH_SIZE}, lr=${LR}, scheduler=exponential(${LR_GAMMA}/round)"
echo "paired initialization=identical shared stem/layer1-4/fc for each seed"
echo "FedMLB=5 local-prefix/frozen-global-suffix paths, CE+KL weights=${FEDMLB_MAIN_CE}/${FEDMLB_HYBRID_CE}/${FEDMLB_HYBRID_KD}"
echo "Adaptive=selected reliability-aware soft-b KD"
echo "outputs=accuracy-vs-round, accuracy-vs-GPU-hours, accuracy-vs-communication"
if [[ "$NUM_GPUS" -eq 2 ]]; then
    echo "estimated clean CIFAR-100 R=500 2-GPU wall time: about 30-60 hours"
else
    echo "estimated clean CIFAR-100 R=500 4-GPU wall time: about 18-36 hours"
fi

# Place all expensive FedMLB jobs first, then continue the same round-robin
# cursor for adaptive jobs.  With two GPUs and three seeds this yields
# [F0,F2,A1] versus [F1,A0,A2], which is substantially better balanced than
# assigning one method per GPU.
queues=()
for ((index = 0; index < NUM_GPUS; index++)); do
    queues[$index]=""
done
job_index=0
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu_index=$((job_index % NUM_GPUS))
        queues[$gpu_index]+="${dataset}|fedmlb|${seed}"$'\n'
        job_index=$((job_index + 1))
    done
    for seed in "${SEEDS[@]}"; do
        gpu_index=$((job_index % NUM_GPUS))
        queues[$gpu_index]+="${dataset}|adaptive|${seed}"$'\n'
        job_index=$((job_index + 1))
    done
done

pids=()
for ((index = 0; index < NUM_GPUS; index++)); do
    if [[ -n "${queues[$index]}" ]]; then
        run_queue "${GPUS[$index]}" "${queues[$index]}" &
        pids+=("$!")
    fi
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "At least one comparison run failed; inspect *_terminal.log." >&2
    exit 1
fi

for dataset in "${DATASETS[@]}"; do
    fedmlb_args=()
    adaptive_args=()
    for seed in "${SEEDS[@]}"; do
        fedmlb_args+=(--fedmlb "${LOG_ROOT}/${dataset}/${PARTITION}/fedmlb/seed${seed}.pkl")
        adaptive_args+=(--adaptive "${LOG_ROOT}/${dataset}/${PARTITION}/adaptive/seed${seed}.pkl")
    done
    "$PYTHON_BIN" scripts/experiments/analysis/plot_fedmlb_adaptive_pareto.py \
        "${fedmlb_args[@]}" "${adaptive_args[@]}" \
        --output-dir "${OUTPUT_ROOT}/${dataset}_${PARTITION}_r${ROUNDS}" \
        --target-accuracies "$(target_accuracies "$dataset")"
done

echo "Completed. Results: ${OUTPUT_ROOT}"
