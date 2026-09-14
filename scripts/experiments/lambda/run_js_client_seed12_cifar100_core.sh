#!/usr/bin/env bash

# Multi-seed validation of the final canonical JS-client adaptive method.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

RUN_SET="${RUN_SET:?Set RUN_SET to server4 or server2}"
read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3}"
(( ${#GPUS[@]} > 0 )) || { echo "GPUS_OVERRIDE is empty." >&2; exit 1; }

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN=venv/bin/python
else
    PYTHON_BIN=python3
fi

LOG_ROOT="${LOG_ROOT:-logs/lambda/final/logs_js_client_seed12_cifar100_core}"
ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-250}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

[[ "$ROUNDS" == 500 ]] || { echo "Final protocol requires ROUNDS=500." >&2; exit 2; }
[[ "$LOCAL_EPOCHS" == 5 ]] || { echo "Final protocol requires LOCAL_EPOCHS=5." >&2; exit 2; }
[[ "$WARMUP_ROUNDS" == 250 ]] || { echo "Final protocol requires WARMUP_ROUNDS=250." >&2; exit 2; }
[[ "$MIN_REQUIRE_SIZE" == 64 ]] || { echo "Final protocol requires MIN_REQUIRE_SIZE=64." >&2; exit 2; }

# Six jobs total across both servers. Each physical GPU receives one job.
case "$RUN_SET" in
    server4)
        JOBS=(
            "iid|1"
            "iid|2"
            "beta_0.3|1"
            "beta_0.3|2"
        )
        ;;
    server2)
        JOBS=(
            "beta_0.1|1"
            "beta_0.1|2"
        )
        ;;
    *) echo "Unknown RUN_SET=${RUN_SET}; expected server4 or server2." >&2; exit 2 ;;
esac

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

has_completed_run() {
    local log_file=$1 pkl_file=$2
    [[ -s "$pkl_file" ]] && [[ -f "$log_file" ]] \
        && grep -q "Round $((ROUNDS - 1)) result" "$log_file"
}

run_job() {
    local gpu=$1 job=$2 partition seed name rel_dir log_file pkl_file
    local -a part_flags cmd
    IFS='|' read -r partition seed <<< "$job"
    mapfile -t part_flags < <(partition_args "$partition")

    name="cifar100_${partition}_js_client_lmax1p00_tau0p85_tkd1p00_canonical_nofeat_seed${seed}_r500"
    rel_dir="cifar100/${partition}/seed${seed}/adaptive"
    log_file="${LOG_ROOT}/${rel_dir}/${name}.log"
    pkl_file="${LOG_ROOT}/${rel_dir}/${name}.pkl"
    mkdir -p "${LOG_ROOT}/${rel_dir}"

    if [[ "$SKIP_EXISTING" == 1 ]] && has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] skip: ${partition} | seed=${seed}"
        return 0
    fi

    cmd=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data --in_channels 3 --num_classes 100
        "${part_flags[@]}" --min_require_size 64
        --n_clients 100 --sample_fraction 0.1
        --round 500 --epochs 5
        --optimizer sgd --lr 0.1 --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size 64 --test_batch_size 512 --num_workers 0
        --seed "$seed" --device "cuda:${gpu}" --sequential_client_execution
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta 0.0
        --byot_teacher_source local --byot_alpha 1.0 --alpha_min_scale 0.0
        --temperature 1.0
        --byot_branch_kd_teacher_temperature 1.0
        --byot_branch_kd_student_temperature 1.0
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_proxy_temperature 1.0
        --byot_round_lambda_schedule linear
        --byot_round_lambda_min 0.0 --byot_round_lambda_warmup 250
        --byot_client_proxy teacher_label_prob
        --byot_client_alpha_min 0.0 --byot_client_alpha_max 1.0
        --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
        --byot_client_skew_proxy prediction_entropy
        --byot_client_skew_power 2.0 --byot_client_skew_min_scale 0.0
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau 0.85
        --byot_client_skew_soft_temperature 0.05
        --byot_branch_need_proxy js_client
        --byot_branch_need_gain 1.0 --byot_branch_need_min_gate 0.0
        --byot_branch_need_temperature 1.0
        --logdir "$LOG_ROOT" --log_file_name "${rel_dir}/${name}"
    )

    echo "[GPU ${gpu}] start: ${partition} | JS-client | seed=${seed}"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run][GPU %s] ' "$gpu"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    if ! "${cmd[@]}" > "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${partition} | seed=${seed}" >&2
        tail -40 "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file"; then
        echo "[GPU ${gpu}] incomplete: ${partition} | seed=${seed}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${partition} | JS-client | seed=${seed}"
}

echo "========== Final JS-client seed 1/2 validation =========="
echo "run_set=${RUN_SET} | GPUs=${GPUS[*]} | jobs=${#JOBS[@]}"
echo "CIFAR-100 / ResNet18-BYOT / FedAvg / IID,beta=.3,beta=.1"
echo "R=500 | E=5 | participation=.1 | min_size=64 | feature_beta=0"
echo "KD/proxy T=1 | lambda_max=1 | warm-up=250 | soft tau=.85 | skew power=2"
echo "JS-client gain=1 | canonical execution flags absent"
echo "log_root=${LOG_ROOT} | skip_existing=${SKIP_EXISTING}"

declare -a QUEUES
for ((i=0; i<${#GPUS[@]}; i++)); do QUEUES[$i]=''; done
for ((i=0; i<${#JOBS[@]}; i++)); do
    QUEUES[$((i % ${#GPUS[@]}))]+="${JOBS[$i]}"$'\n'
done

run_queue() {
    local gpu_index=$1 gpu="${GPUS[$1]}" failed=0 job
    while IFS= read -r job; do
        [[ -n "$job" ]] || continue
        run_job "$gpu" "$job" || failed=1
    done <<< "${QUEUES[$gpu_index]}"
    return "$failed"
}

pids=()
for ((i=0; i<${#GPUS[@]}; i++)); do
    [[ -n "${QUEUES[$i]}" ]] || continue
    run_queue "$i" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if [[ "$status" != 0 ]]; then
    echo "At least one adaptive validation run failed; inspect *_terminal.log." >&2
    exit "$status"
fi
echo "Final JS-client seed validation complete (${#JOBS[@]} jobs)."
