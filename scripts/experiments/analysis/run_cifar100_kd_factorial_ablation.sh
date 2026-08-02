#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# C100 factorial ablation for the residual KD mechanism question.
#
# Fixed throughout: FedAvg, Dirichlet beta=0.5, all BYOT branches, feature
# imitation, teacher KD temperature T_t=0.5, and alpha=1.  The 2 x 2 x 2
# factors are:
#   (A) target mass: per-sample teacher q_T(y|x) vs its batch mean;
#   (B) student temperature: T_s=0.5 vs 1.0;
#   (C) KL multiplier: native T_s^2 vs T_s (unit leading gradient factor).
# Both targets spread non-target mass uniformly, so their only difference is
# sample-adaptive teacher target mass.  This intentionally does not compare
# the full teacher distribution again; that was measured in target ablation.

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

SEEDS=(${SEEDS_OVERRIDE:-0 1})
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.5}"
LOG_ROOT="${LOG_ROOT:-logs/analysis/logs_cifar100_kd_factorial_r500}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Refuse to mix Python source versions in one factorial comparison.  Running
# Python processes keep the source loaded at launch, so editing either file
# mid-run would otherwise reproduce an old-code/new-code mixture.
CODE_FILES=(main.py train.py data_utils.py fl_utils.py utils.py models/resnet_byot.py)
INITIAL_CODE_FINGERPRINT="$(sha256sum "${CODE_FILES[@]}" | sha256sum | awk '{print $1}')"

assert_code_unchanged() {
    local current_fingerprint
    current_fingerprint="$(sha256sum "${CODE_FILES[@]}" | sha256sum | awk '{print $1}')"
    if [ "$current_fingerprint" != "$INITIAL_CODE_FINGERPRINT" ]; then
        echo "Source changed during the factorial run; refusing to mix code versions." >&2
        echo "initial=${INITIAL_CODE_FINGERPRINT}, current=${current_fingerprint}" >&2
        return 1
    fi
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
    accuracies = payload.get("acc_global", [])
    ok = isinstance(accuracies, (list, tuple)) and len(accuracies) >= expected_rounds
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
' "$result" "$ROUNDS"
}

run_job() {
    local gpu_id=$1 variant=$2 target_mode=$3 student_temperature=$4 scale_mode=$5 seed=$6
    local setting="cifar100_resnet18/beta_0.5/fedavg/seed${seed}"
    local log_dir="${LOG_ROOT}/${setting}"
    local result="${log_dir}/${variant}.pkl"
    mkdir -p "$log_dir"
    if [ "$SKIP_EXISTING" = "1" ] && has_completed_result "$result"; then
        echo "[GPU ${gpu_id}] skip: ${setting} | ${variant}"
        return
    fi

    assert_code_unchanged

    echo "[GPU ${gpu_id}] start: ${setting} | ${variant}"
    "$PYTHON_BIN" main.py \
        --dataset cifar100 --datadir ./data \
        --n_clients 100 --sample_fraction 0.1 \
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$seed" \
        --device "cuda:${gpu_id}" \
        --logdir "$LOG_ROOT" --log_file_name "${setting}/${variant}" \
        --model resnet18_byot --alg fedbyot \
        --partition noniid --beta 0.5 \
        --byot_active_branches 1,2,3 \
        --byot_branch_loss_reduction sum \
        --byot_branch_objective blend --byot_alpha 1.00 \
        --byot_beta "$FEATURE_BETA" \
        --temperature "$TEACHER_TEMPERATURE" \
        --byot_branch_kd_teacher_temperature "$TEACHER_TEMPERATURE" \
        --byot_branch_kd_student_temperature "$student_temperature" \
        --byot_branch_kd_loss_scale_mode "$scale_mode" \
        --byot_branch_kd_target_mode "$target_mode" \
        > "${log_dir}/${variant}_terminal.log" 2>&1

    if ! has_completed_result "$result"; then
        echo "[GPU ${gpu_id}] invalid or incomplete result: ${result}" >&2
        return 1
    fi
    echo "[GPU ${gpu_id}] complete: ${setting} | ${variant}"
}

declare -a JOBS=()
add_variant() {
    local variant=$1 target_mode=$2 student_temperature=$3 scale_mode=$4
    local seed
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${variant}|${target_mode}|${student_temperature}|${scale_mode}|${seed}")
    done
}

for target in adaptive batchmean; do
    if [ "$target" = "adaptive" ]; then
        target_mode="teacher_mass_uniform"
    else
        target_mode="teacher_mass_uniform_batchmean"
    fi
    for student_temperature in 0.5 1.0; do
        student_tag="t${student_temperature/./p}"
        for scale_mode in native_t2 gradient_prefactor_one; do
            if [ "$scale_mode" = "native_t2" ]; then
                scale_tag="native"
            else
                scale_tag="gradnorm"
            fi
            add_variant "${target}_${student_tag}_${scale_tag}" "$target_mode" "$student_temperature" "$scale_mode"
        done
    done
done

echo "========== C100 branch-KD factorial ablation =========="
echo "gpus=${GPUS[*]}, rounds=${ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seeds=${SEEDS[*]}"
echo "partition=beta_0.5, feature_beta=${FEATURE_BETA}, teacher_temperature=${TEACHER_TEMPERATURE}"
echo "factors=target(adaptive,batchmean) x student_temperature(0.5,1.0) x scale(native,gradnorm)"
echo "log_root=${LOG_ROOT}, jobs=${#JOBS[@]}, skip_existing=${SKIP_EXISTING}"
echo "code_fingerprint=${INITIAL_CODE_FINGERPRINT}"

mkdir -p "$LOG_ROOT"
MANIFEST_PATH="${LOG_ROOT}/run_manifest.txt"
if [ -f "$MANIFEST_PATH" ]; then
    recorded_fingerprint="$(awk -F= '$1 == "code_fingerprint" {print $2}' "$MANIFEST_PATH")"
    if [ -z "$recorded_fingerprint" ] || [ "$recorded_fingerprint" != "$INITIAL_CODE_FINGERPRINT" ]; then
        echo "Existing log root was created by a different source version: ${LOG_ROOT}" >&2
        echo "Use a new LOG_ROOT instead of mixing results." >&2
        exit 1
    fi
else
    {
        echo "code_fingerprint=${INITIAL_CODE_FINGERPRINT}"
        echo "git_revision=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
        echo "rounds=${ROUNDS}"
        echo "local_epochs=${LOCAL_EPOCHS}"
        echo "seeds=${SEEDS[*]}"
    } > "$MANIFEST_PATH"
fi

run_queue() {
    local gpu_id=$1
    shift
    local job variant target_mode student_temperature scale_mode seed
    local queue_failed=0
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r variant target_mode student_temperature scale_mode seed <<< "$job"
        if ! run_job "$gpu_id" "$variant" "$target_mode" "$student_temperature" "$scale_mode" "$seed"; then
            echo "[GPU ${gpu_id}] failed: ${variant} | seed=${seed}" >&2
            queue_failed=1
        fi
    done
    return "$queue_failed"
}

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done
for ((i = 0; i < ${#JOBS[@]}; i++)); do
    QUEUES[$((i % NUM_GPUS))]+="${JOBS[$i]}"$'\n'
done

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    if [ -n "${QUEUES[$i]}" ]; then
        mapfile -t queue_jobs <<< "${QUEUES[$i]}"
        run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
        pids+=("$!")
    fi
done
queue_failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        queue_failed=1
    fi
done
if [ "$queue_failed" -ne 0 ]; then
    echo "One or more factorial jobs failed; inspect *_terminal.log files." >&2
    exit 1
fi
echo "C100 branch-KD factorial ablation complete (${#JOBS[@]} jobs)"
