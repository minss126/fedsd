#!/usr/bin/env bash

# Complete the publication fixed-lambda curve under the final canonical
# feature-free protocol.  Existing lambda={.1,.3,.5} runs live in the same
# log root; this runner adds only lambda={0,1,3}.
#
# CIFAR-100 / ResNet18-BYOT / FedAvg / R=500 / E=5 / C=.1 / seed=0
# IID, beta=.3, beta=.1 / KD-only / T_KD=1 / feature loss=0 / min-size=64
# Sequential client execution is retained to match the final performance runs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1}"
if (( ${#GPUS[@]} != 2 )); then
    echo "Provide exactly two GPU ids through GPUS_OVERRIDE." >&2
    exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

SEED="${SEED:-0}"
ROUNDS=500
LOCAL_EPOCHS=5
SAMPLE_FRACTION=0.1
MIN_REQUIRE_SIZE=64
KD_TEMPERATURE=1.0
LOG_ROOT="${LOG_ROOT:-logs/lambda/analysis/logs_fixed_lambda_canonical_no_feature_t1_recheck}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

PARTITIONS=(iid beta_0.3 beta_0.1)
LAMBDAS=(0.0 1.0 3.0)

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

pkl_complete() {
    local path="$1"
    [[ -s "$path" ]] || return 1
    "$PYTHON_BIN" -c '
import pickle, sys
try:
    with open(sys.argv[1], "rb") as stream:
        payload = pickle.load(stream)
    expected = int(sys.argv[2])
    candidates = (
        payload.get("acc_global", []),
        payload.get("branch_acc", []),
        payload.get("test_loss", []),
    )
    ok = any(isinstance(values, (list, tuple)) and len(values) >= expected
             for values in candidates)
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
' "$path" "$ROUNDS"
}

# job format: partition|lambda
JOBS=()
for partition in "${PARTITIONS[@]}"; do
    for lambda_value in "${LAMBDAS[@]}"; do
        JOBS+=("${partition}|${lambda_value}")
    done
done

job_weight() {
    local partition="${1%%|*}"
    case "$partition" in
        iid) printf '113' ;;
        beta_0.3) printf '117' ;;
        beta_0.1) printf '49' ;;
    esac
}

declare -a QUEUES LOADS
for ((i=0; i<2; i++)); do QUEUES[$i]=''; LOADS[$i]=0; done
for job in "${JOBS[@]}"; do
    target=0
    (( LOADS[1] < LOADS[0] )) && target=1
    QUEUES[$target]+="${job}"$'\n'
    weight="$(job_weight "$job")"
    LOADS[$target]=$((LOADS[$target] + weight))
done

run_job() {
    local gpu="$1" job="$2" partition lambda_value tag name rel_dir pkl_file terminal
    local -a part_flags cmd
    IFS='|' read -r partition lambda_value <<< "$job"
    mapfile -t part_flags < <(partition_args "$partition")
    tag="$(value_tag "$lambda_value")"
    name="fixed_lambda${tag}_tkd1p00_canonical_nofeat_seed${SEED}_r${ROUNDS}"
    rel_dir="cifar100/${partition}/seed${SEED}/lambda${tag}"
    pkl_file="${LOG_ROOT}/${rel_dir}/${name}.pkl"
    terminal="${LOG_ROOT}/${rel_dir}/${name}_terminal.log"

    if [[ "$SKIP_EXISTING" == 1 ]] && pkl_complete "$pkl_file"; then
        echo "[GPU ${gpu}] skip: ${partition} | lambda=${lambda_value}"
        return 0
    fi

    mkdir -p "${LOG_ROOT}/${rel_dir}"
    cmd=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data --in_channels 3 --num_classes 100
        "${part_flags[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients 100 --sample_fraction "$SAMPLE_FRACTION"
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS"
        --optimizer sgd --lr 0.1 --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size 64 --test_batch_size 512 --num_workers 0
        --seed "$SEED" --device "cuda:${gpu}" --sequential_client_execution
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta 0.0
        --byot_teacher_source local --byot_alpha "$lambda_value"
        --alpha_min_scale 0.0 --byot_branch_need_proxy none
        --temperature "$KD_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_proxy_temperature 1.0
        --logdir "$LOG_ROOT" --log_file_name "${rel_dir}/${name}"
    )

    echo "[GPU ${gpu}] start: ${partition} | fixed lambda=${lambda_value}"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run] '
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    if ! "${cmd[@]}" > "$terminal" 2>&1; then
        echo "[GPU ${gpu}] failed: ${partition} | lambda=${lambda_value}" >&2
        tail -40 "$terminal" >&2 || true
        return 1
    fi
    if ! pkl_complete "$pkl_file"; then
        echo "[GPU ${gpu}] incomplete output: ${pkl_file}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${partition} | fixed lambda=${lambda_value}"
}

run_queue() {
    local index="$1" gpu="${GPUS[$1]}" job failed=0
    while IFS= read -r job; do
        [[ -n "$job" ]] || continue
        run_job "$gpu" "$job" || failed=1
    done <<< "${QUEUES[$index]}"
    return "$failed"
}

echo "========== Fixed-lambda canonical completion =========="
echo "GPUs=${GPUS[*]} | jobs=${#JOBS[@]} | estimated_queue_minutes=${LOADS[*]}"
echo "lambdas=0,1,3 | partitions=IID,beta0.3,beta0.1 | seed=${SEED}"
echo "CIFAR-100/ResNet18-BYOT/FedAvg/R500/E5/C0.1/T1/no-feature/min64"
echo "log_root=${LOG_ROOT} | skip_existing=${SKIP_EXISTING}"

if [[ "$DRY_RUN" == 1 ]]; then
    run_queue 0
    run_queue 1
    echo "Dry run complete."
    exit 0
fi

run_queue 0 & pid0=$!
run_queue 1 & pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if (( status != 0 )); then
    echo "At least one fixed-lambda run failed; inspect *_terminal.log." >&2
    exit "$status"
fi
echo "Fixed-lambda canonical completion finished."
