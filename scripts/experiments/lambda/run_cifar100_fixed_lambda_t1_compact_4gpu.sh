#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Compact fixed-lambda sweep corresponding to the paper's "Role of lambda"
# analysis, rerun with teacher/student KD temperatures fixed to 1.0.
#
# This launcher intentionally runs only lambda={0.1,0.3,3,5}:
#   - lambda=0 is temperature-independent and can reuse the old feature-only
#     result;
#   - lambda=1 is produced by run_cifar100_fixed_lambda1_no_warmup_4gpu.sh.
#
# Every lambda below is constant from round 0 through round 499.  No round
# schedule or warm-up flags are passed.

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-4}"
if [ "${#GPUS[@]}" -ne "$EXPECTED_GPU_COUNT" ]; then
    echo "This launcher requires exactly ${EXPECTED_GPU_COUNT} GPU ids; received: ${GPUS[*]}" >&2
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
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/analysis/logs_cifar100_fixed_lambda_t1_compact}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# This experiment is specifically the T_KD=1 replacement for the old T=0.5
# fixed-lambda table.  Reject an inherited override instead of silently
# producing a mislabeled run.
case "$KD_TEMPERATURE" in
    1|1.0|1.00) KD_TEMPERATURE="1.00" ;;
    *)
        echo "This launcher requires KD_TEMPERATURE=1.0; received ${KD_TEMPERATURE}." >&2
        exit 1
        ;;
esac

ENVS=(${ENVS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})
LAMBDAS=(${LAMBDAS_OVERRIDE:-0.10:0p10 0.30:0p30 3.00:3p00 5.00:5p00})

WANDB_FLAGS=()
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
fi

CODE_FILES=(main.py train.py data_utils.py fl_utils.py utils.py models/resnet_byot.py)
INITIAL_CODE_FINGERPRINT="$(sha256sum "${CODE_FILES[@]}" | sha256sum | awk '{print $1}')"

assert_code_unchanged() {
    local current
    current="$(sha256sum "${CODE_FILES[@]}" | sha256sum | awk '{print $1}')"
    if [ "$current" != "$INITIAL_CODE_FINGERPRINT" ]; then
        echo "Source changed during the fixed-lambda sweep; refusing to mix code versions." >&2
        return 1
    fi
}

env_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown environment: $1" >&2; return 1 ;;
    esac
}

has_completed_result() {
    local result=$1
    [ -s "$result" ] || return 1
    "$PYTHON_BIN" -c '
import pickle
import sys

path, expected_rounds = sys.argv[1], int(sys.argv[2])
try:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    values = payload.get("acc_global", [])
    ok = isinstance(values, (list, tuple)) and len(values) >= expected_rounds
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
' "$result" "$ROUNDS"
}

run_job() {
    local gpu_id=$1 env_name=$2 lambda_value=$3 lambda_tag=$4
    local name log_dir result
    local -a PARTITION_ARGS CMD

    name="fixed_lambda${lambda_tag}_tkd1p00"
    log_dir="${LOG_ROOT}/${env_name}/fedavg"
    result="${log_dir}/${name}.pkl"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_result "$result"; then
        echo "[GPU ${gpu_id}] skip: ${env_name} | lambda=${lambda_value}"
        return
    fi

    assert_code_unchanged
    mapfile -t PARTITION_ARGS < <(env_args "$env_name")
    CMD=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data
        --n_clients 100 --sample_fraction 0.1
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED"
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT"
        --log_file_name "${env_name}/fedavg/${name}"
        --model resnet18_byot --alg fedbyot
        --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
        --byot_alpha "$lambda_value"
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_branch_kd_target_mode full_teacher
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
    )
    CMD+=("${PARTITION_ARGS[@]}")
    CMD+=("${WANDB_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: ${env_name} | fixed lambda=${lambda_value} | no warm-up"
    if ! "${CMD[@]}" > "${log_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu_id}] failed: ${env_name} | lambda=${lambda_value}" >&2
        return 1
    fi
    if ! has_completed_result "$result"; then
        echo "[GPU ${gpu_id}] invalid result: ${result}" >&2
        return 1
    fi
    echo "[GPU ${gpu_id}] complete: ${env_name} | fixed lambda=${lambda_value}"
}

declare -a JOBS=()
for env_name in "${ENVS[@]}"; do
    for lambda_pair in "${LAMBDAS[@]}"; do
        JOBS+=("${env_name}|${lambda_pair%%:*}|${lambda_pair##*:}")
    done
done

echo "========== C100 compact fixed-lambda sweep at T_KD=1 =========="
echo "gpus=${GPUS[*]}, envs=${ENVS[*]}, lambdas=${LAMBDAS[*]}"
echo "rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "T_KD=${KD_TEMPERATURE}, T_proxy=${PROXY_TEMPERATURE}, feature_beta=${FEATURE_BETA}"
echo "schedule=none, jobs=${#JOBS[@]}, log_root=${LOG_ROOT}"
echo "lambda=0: reuse old feature-only result"
echo "lambda=1: reuse logs/lambda/adaptive/logs_cifar100_fixed_lambda1_no_warmup"

mkdir -p "$LOG_ROOT"
{
    echo "code_fingerprint=${INITIAL_CODE_FINGERPRINT}"
    echo "git_revision=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "kd_temperature=${KD_TEMPERATURE}"
    echo "schedule=none"
    echo "lambdas=${LAMBDAS[*]}"
} > "${LOG_ROOT}/run_manifest.txt"

run_queue() {
    local gpu_id=$1
    shift
    local job env_name lambda_value lambda_tag
    local failed=0
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r env_name lambda_value lambda_tag <<< "$job"
        if ! run_job "$gpu_id" "$env_name" "$lambda_value" "$lambda_tag"; then
            failed=1
        fi
    done
    return "$failed"
}

declare -a QUEUES
for ((i = 0; i < ${#GPUS[@]}; i++)); do QUEUES[$i]=""; done
for ((i = 0; i < ${#JOBS[@]}; i++)); do
    QUEUES[$((i % ${#GPUS[@]}))]+="${JOBS[$i]}"$'\n'
done

pids=()
for ((i = 0; i < ${#GPUS[@]}; i++)); do
    mapfile -t queue_jobs <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
if [ "$failed" -ne 0 ]; then
    echo "One or more compact fixed-lambda jobs failed." >&2
    exit 1
fi

echo "C100 compact fixed-lambda T=1 sweep complete (${#JOBS[@]} jobs)."
