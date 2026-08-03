#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# One-shot follow-up for the currently leading stage-1 candidate:
#   1) run tau={0.80, 0.90}; tau=0.85 is reused from stage 1;
#   2) select tau by mean last-30 accuracy over beta={0.5,0.1};
#   3) run final base validation: adaptive on IID/beta=0.3 and fixed
#      lambda=1 (constant from round 0) on IID/beta={0.5,0.3,0.1}.
#
# This intentionally does not wait for the separate 2-GPU server.  If that
# server later selects a different (T_KD, lambda_max) pair, re-run this
# pipeline with KD_TEMPERATURE and LAMBDA_MAX overridden for that candidate.

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1 2 3\" bash $0"
    echo
    echo "Expected wall time: about 13-14 hours from launch."
    echo "Requires the matching stage-1 logs under LOG_ROOT."
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -ne 4 ]; then
    echo "This pipeline requires exactly four GPU ids; received: ${GPUS[*]}" >&2
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
KD_TEMPERATURE="${KD_TEMPERATURE:-1.00}"
LAMBDA_MAX="${LAMBDA_MAX:-1.00}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_WARMUP="${LAMBDA_WARMUP:-250}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
TAU_VALUES=(${TAU_VALUES:-0.80 0.85 0.90})
NEW_TAU_VALUES=(${NEW_TAU_VALUES:-0.80 0.90})
DEV_ENVS=(beta_0.5 beta_0.1)
FINAL_ADAPTIVE_ENVS=(iid beta_0.3)
FINAL_FIXED_ENVS=(iid beta_0.5 beta_0.3 beta_0.1)
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_soft_adaptive_tuning_stage1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SELECTOR="scripts/experiments/lambda/select_soft_adaptive_tuning.py"

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

kd_tag="$(value_tag "$KD_TEMPERATURE")"
lambda_tag="$(value_tag "$LAMBDA_MAX")"

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

tau_tag() {
    value_tag "$1"
}

adaptive_name() {
    local tau=$1
    echo "soft_b_tkd${kd_tag}_lmax${lambda_tag}_warm${LAMBDA_WARMUP}_tau$(tau_tag "$tau")"
}

fixed_name() {
    echo "fixed_lambda1_tkd${kd_tag}"
}

has_completed_log() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

base_args() {
    echo "--dataset cifar100 --datadir ./data --n_clients 100 --sample_fraction 0.1 --epochs ${LOCAL_EPOCHS} --lr ${LR} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --round ${ROUNDS} --seed ${SEED} --model resnet18_byot --alg fedbyot --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum --byot_branch_objective kd_only --byot_beta ${FEATURE_BETA} --temperature ${KD_TEMPERATURE} --byot_branch_kd_teacher_temperature ${KD_TEMPERATURE} --byot_branch_kd_student_temperature ${KD_TEMPERATURE} --byot_proxy_temperature ${PROXY_TEMPERATURE}"
}

run_adaptive_job() {
    local gpu_id=$1 env_name=$2 tau=$3
    local name log_dir log_file partition_args args
    name="$(adaptive_name "$tau")"
    log_dir="${LOG_ROOT}/${env_name}/fedavg"
    log_file="${log_dir}/${name}.log"
    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] adaptive | ${env_name} | ${name}"
        return
    fi
    partition_args="$(env_flags "$env_name")"
    args="$(base_args) --byot_alpha ${LAMBDA_MAX} --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00 --byot_round_lambda_warmup ${LAMBDA_WARMUP} --alpha_min_scale 0.0 --byot_client_proxy teacher_label_prob --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00 --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0 --byot_client_skew_proxy prediction_entropy --byot_client_skew_power ${SKEW_POWER} --byot_client_skew_min_scale 0.00 --byot_client_skew_correction_mode soft_relax --byot_client_skew_soft_tau ${tau} --byot_client_skew_soft_temperature ${SOFT_TEMPERATURE}"
    echo "[GPU ${gpu_id}] start: adaptive | ${env_name} | ${name}"
    "$PYTHON_BIN" main.py ${args} ${partition_args} ${WANDB_FLAGS} \
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT" \
        --log_file_name "${env_name}/fedavg/${name}" \
        > "${log_dir}/${name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: adaptive | ${env_name} | ${name}"
}

run_fixed_job() {
    local gpu_id=$1 env_name=$2
    local name log_dir log_file partition_args args
    name="$(fixed_name)"
    log_dir="${LOG_ROOT}/${env_name}/fedavg"
    log_file="${log_dir}/${name}.log"
    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file"; then
        echo "[skip] fixed | ${env_name} | ${name}"
        return
    fi
    partition_args="$(env_flags "$env_name")"
    args="$(base_args) --byot_alpha 1.00"
    echo "[GPU ${gpu_id}] start: fixed | ${env_name} | ${name}"
    "$PYTHON_BIN" main.py ${args} ${partition_args} ${WANDB_FLAGS} \
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT" \
        --log_file_name "${env_name}/fedavg/${name}" \
        > "${log_dir}/${name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: fixed | ${env_name} | ${name}"
}

run_queue() {
    local gpu_id=$1
    shift
    local job kind env_name tau
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r kind env_name tau <<< "$job"
        case "$kind" in
            adaptive) run_adaptive_job "$gpu_id" "$env_name" "$tau" ;;
            fixed) run_fixed_job "$gpu_id" "$env_name" ;;
            *) echo "Unknown job kind: $kind" >&2; return 1 ;;
        esac
    done
}

launch_jobs() {
    local -a jobs=("$@")
    local -a queues pids
    local i job gpu_idx
    for ((i = 0; i < NUM_GPUS; i++)); do queues[$i]=""; done
    for i in "${!jobs[@]}"; do
        job="${jobs[$i]}"
        gpu_idx=$((i % NUM_GPUS))
        queues[$gpu_idx]+="${job}"$'\n'
    done
    pids=()
    for ((i = 0; i < NUM_GPUS; i++)); do
        if [ -n "${queues[$i]}" ]; then
            mapfile -t queue_jobs <<< "${queues[$i]}"
            run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
            pids+=("$!")
        fi
    done
    wait "${pids[@]}"
}

echo "========== Tau Selection + Final Base Validation =========="
echo "candidate: T_KD=${KD_TEMPERATURE}, lambda_max=${LAMBDA_MAX}"
echo "tau candidates: ${TAU_VALUES[*]} (new runs: ${NEW_TAU_VALUES[*]})"
echo "gpus: ${GPUS[*]}, log_root: ${LOG_ROOT}"

tau_jobs=()
for tau in "${NEW_TAU_VALUES[@]}"; do
    for env_name in "${DEV_ENVS[@]}"; do
        tau_jobs+=("adaptive|${env_name}|${tau}")
    done
done
launch_jobs "${tau_jobs[@]}"

selection_file="${LOG_ROOT}/selection_tkd${kd_tag}_lmax${lambda_tag}.json"
echo "Tau runs finished. Selecting tau for the current 4-GPU candidate."
"$PYTHON_BIN" "$SELECTOR" \
    --log-root "$LOG_ROOT" \
    --kd-temperature "$KD_TEMPERATURE" --lambda-max "$LAMBDA_MAX" \
    --taus "${TAU_VALUES[@]}" \
    --stage1-kd-values "$KD_TEMPERATURE" \
    --stage1-lambda-values "$LAMBDA_MAX" \
    --stage1-tau 0.85 --warmup "$LAMBDA_WARMUP" \
    --rounds "$ROUNDS" --window 30 \
    --output "$selection_file"

selected_tau="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_tau"])' "$selection_file")"
echo "Selected tau=${selected_tau}; starting final base validation."

final_jobs=()
for env_name in "${FINAL_FIXED_ENVS[@]}"; do
    final_jobs+=("fixed|${env_name}|")
done
for env_name in "${FINAL_ADAPTIVE_ENVS[@]}"; do
    final_jobs+=("adaptive|${env_name}|${selected_tau}")
done
launch_jobs "${final_jobs[@]}"

echo "Tau selection and final base validation complete."
