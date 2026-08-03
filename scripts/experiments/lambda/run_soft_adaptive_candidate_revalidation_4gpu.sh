#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Multi-seed re-ranking after the seed-0 screening grid.
#
# Fixed method definition:
#   lambda_k = lambda_t * r_k * [(1-g_k)b_k^2 + g_k]
#   T_proxy=1, tau=0.85, T_soft=0.05, p=2, warm-up=250.
#
# Candidates retained from seed 0:
#   (T_KD, lambda_max) = (1.00, 1.00), (0.50, 2.00), (0.50, 3.00)
#
# New evaluation matrix:
#   seed={1,2} x candidate={3} x beta={0.5,0.1} = 12 runs.
# On four GPUs each GPU receives two sequential runs; expected wall time is
# roughly 12-14 hours using the observed CIFAR-100 runtime.

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1 2 3\" bash $0"
    echo
    echo "Runs seed 1/2 revalidation for the three retained soft-b candidates."
    echo "Override candidates with: CANDIDATES_OVERRIDE=\"1.00,1.00 0.50,2.00\""
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -ne 4 ]; then
    echo "This launcher requires exactly four GPU ids; received: ${GPUS[*]}" >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then
        PYTHON_BIN="venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

SEEDS=(${SEEDS_OVERRIDE:-1 2})
ENVS=(${ENVS_OVERRIDE:-beta_0.5 beta_0.1})
CANDIDATES=(${CANDIDATES_OVERRIDE:-1.00,1.00 0.50,2.00 0.50,3.00})

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
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_soft_adaptive_candidate_revalidation}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

WANDB_FLAGS=""
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT:-dxfl}"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

env_flags() {
    case "$1" in
        beta_0.5) echo "--partition noniid --beta 0.5" ;;
        beta_0.1) echo "--partition noniid --beta 0.1" ;;
        *) echo "Unknown environment: $1" >&2; exit 1 ;;
    esac
}

method_name() {
    local kd_temp=$1 lambda_max=$2
    echo "soft_b_tkd$(value_tag "$kd_temp")_lmax$(value_tag "$lambda_max")_warm${LAMBDA_WARMUP}_tau$(value_tag "$SOFT_TAU")"
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu_id=$1 seed=$2 env_name=$3 kd_temp=$4 lambda_max=$5
    local name log_dir log_file partition_args
    name="$(method_name "$kd_temp" "$lambda_max")"
    log_dir="${LOG_ROOT}/seed${seed}/${env_name}/fedavg"
    log_file="${log_dir}/${name}.log"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] seed${seed} | ${env_name} | ${name}"
        return
    fi

    partition_args="$(env_flags "$env_name")"
    echo "[GPU ${gpu_id}] start: seed${seed} | ${env_name} | ${name}"

    "$PYTHON_BIN" main.py \
        --dataset cifar100 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$seed" \
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT" \
        --log_file_name "seed${seed}/${env_name}/fedavg/${name}" \
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

    echo "[GPU ${gpu_id}] complete: seed${seed} | ${env_name} | ${name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local job seed env_name kd_temp lambda_max
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r seed env_name kd_temp lambda_max <<< "$job"
        run_job "$gpu_id" "$seed" "$env_name" "$kd_temp" "$lambda_max"
    done
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done

job_count=0
for seed in "${SEEDS[@]}"; do
    for candidate in "${CANDIDATES[@]}"; do
        IFS=',' read -r kd_temp lambda_max <<< "$candidate"
        for env_name in "${ENVS[@]}"; do
            gpu_idx=$((job_count % NUM_GPUS))
            QUEUES[$gpu_idx]+="${seed}|${env_name}|${kd_temp}|${lambda_max}"$'\n'
            job_count=$((job_count + 1))
        done
    done
done

echo "========== Soft-b Candidate Revalidation =========="
echo "gpus=${GPUS[*]}, jobs=${job_count}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}"
echo "seeds=${SEEDS[*]}, envs=${ENVS[*]}"
echo "candidates (T_KD,lambda_max)=${CANDIDATES[*]}"
echo "fixed: T_proxy=${PROXY_TEMPERATURE}, tau=${SOFT_TAU}, T_soft=${SOFT_TEMPERATURE}, p=${SKEW_POWER}, warmup=${LAMBDA_WARMUP}"
echo "log_root=${LOG_ROOT}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    mapfile -t queue_jobs <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
    pids+=("$!")
done

wait "${pids[@]}"
echo "Soft-b candidate revalidation complete (${job_count} jobs)"
