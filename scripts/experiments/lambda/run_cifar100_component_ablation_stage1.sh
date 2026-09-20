#!/usr/bin/env bash

# Leave-one-component-out ablation for the final canonical adaptive method.
#
# Final method:
#   CIFAR-100 / ResNet18-BYOT / FedAvg / KD-only / no feature loss
#   T_KD=1 / T_proxy=1 / lambda_max=1 / tau=.85 / JS-client
#   canonical execution (no paired/preserved RNG controls)
#
# Default new jobs (8):
#   partitions = IID, beta=.1
#   ablations  = w/o warm-up, reliability, bias correction, JS-client
#
# The full adaptive result is already produced by the publication core matrix
# and is therefore not rerun by default. Include `full` in METHODS_OVERRIDE if
# a local full run/reference check is desired. Completed PKLs are skipped, so
# interrupted queues can be relaunched safely (the interrupted cell restarts).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage (recommended after the ResNet50 queue on the 2-GPU server):
  GPUS_OVERRIDE="0 1" \
    bash scripts/experiments/lambda/run_cifar100_component_ablation_stage1.sh

Defaults:
  PARTITIONS_OVERRIDE="iid beta_0.1"
  METHODS_OVERRIDE="wo_warmup wo_reliability wo_bias wo_js_client"
  SEEDS_OVERRIDE="0"

Optional:
  METHODS_OVERRIDE="wo_warmup wo_reliability wo_bias wo_js_client full"
  SEEDS_OVERRIDE="0 1 2"
  DRY_RUN=1 GPUS_OVERRIDE="0 1" bash scripts/experiments/lambda/run_cifar100_component_ablation_stage1.sh
EOF
    exit 0
fi

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1}"
(( ${#GPUS[@]} > 0 )) || { echo "GPUS_OVERRIDE is empty." >&2; exit 2; }
NUM_GPUS=${#GPUS[@]}

if [[ -n "${PYTHON_BIN:-}" ]]; then :
elif [[ -x venv/bin/python ]]; then PYTHON_BIN=venv/bin/python
else PYTHON_BIN=python3
fi

read -r -a PARTITIONS <<< "${PARTITIONS_OVERRIDE:-iid beta_0.1}"
read -r -a METHODS <<< "${METHODS_OVERRIDE:-wo_warmup wo_reliability wo_bias wo_js_client}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-0}"

ROUNDS="${ROUNDS:-500}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"

FEATURE_BETA=0.0
KD_TEMPERATURE=1.0
PROXY_TEMPERATURE=1.0
LAMBDA_MAX=1.0
LAMBDA_WARMUP="${LAMBDA_WARMUP:-$((ROUNDS / 2))}"
SKEW_POWER=2.0
SOFT_TAU=0.85
SOFT_TEMPERATURE=0.05
JS_GAIN=1.0

LOG_ROOT="${LOG_ROOT:-logs/lambda/final/logs_cifar100_js_client_component_ablation}"
FULL_REFERENCE_ROOT="${FULL_REFERENCE_ROOT:-logs/lambda/final/logs_publication_core_matrix}"
REUSE_FULL_REFERENCE="${REUSE_FULL_REFERENCE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
USE_WANDB="${USE_WANDB:-0}"

[[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || { echo "ROUNDS must be positive." >&2; exit 2; }
[[ "$LAMBDA_WARMUP" =~ ^[0-9]+$ ]] || { echo "LAMBDA_WARMUP must be non-negative." >&2; exit 2; }

partition_flags() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

validate_method() {
    case "$1" in
        wo_warmup|wo_reliability|wo_bias|wo_js_client|full) ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

pkl_complete() {
    local path="$1" expected="$2"
    [[ -s "$path" ]] || return 1
    "$PYTHON_BIN" -c '
import pickle, sys
try:
    with open(sys.argv[1], "rb") as handle:
        payload = pickle.load(handle)
    expected = int(sys.argv[2])
    complete = any(
        isinstance(payload.get(key), (list, tuple)) and len(payload[key]) >= expected
        for key in ("acc_global", "branch_acc", "test_loss")
    )
except Exception:
    complete = False
raise SystemExit(0 if complete else 1)
' "$path" "$expected"
}

run_name() {
    local partition="$1" method="$2" seed="$3"
    printf 'cifar100_resnet18_fedavg_%s_ablation_%s_tkd1p00_nofeat_canonical_seed%s_r%s' \
        "$partition" "$method" "$seed" "$ROUNDS"
}

current_path() {
    local partition="$1" method="$2" seed="$3"
    printf '%s/%s/seed%s/%s/%s.pkl' \
        "$LOG_ROOT" "$partition" "$seed" "$method" \
        "$(run_name "$partition" "$method" "$seed")"
}

full_reference_path() {
    local partition="$1" seed="$2"
    local name="cifar100_resnet18_fedavg_${partition}_js_client_lmax1p00_tau0p85_tkd1p00_nofeat_canonical_seed${seed}_r${ROUNDS}"
    printf '%s/resnet18/cifar100/fedavg/%s/seed%s/adaptive/%s.pkl' \
        "$FULL_REFERENCE_ROOT" "$partition" "$seed" "$name"
}

append_warmup() {
    CMD+=(
        --byot_round_lambda_schedule linear
        --byot_round_lambda_min 0.0
        --byot_round_lambda_warmup "$LAMBDA_WARMUP"
    )
}

append_reliability() {
    CMD+=(
        --byot_client_proxy teacher_label_prob
        --byot_client_alpha_min 0.0 --byot_client_alpha_max 1.0
        --byot_client_alpha_mode multiply
        --byot_client_reliability_power 1.0
    )
}

append_bias_correction() {
    CMD+=(
        --byot_client_skew_proxy prediction_entropy
        --byot_client_skew_power "$SKEW_POWER"
        --byot_client_skew_min_scale 0.0
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau "$SOFT_TAU"
        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
    )
}

append_js_client() {
    CMD+=(
        --byot_branch_need_proxy js_client
        --byot_branch_need_gain "$JS_GAIN"
        --byot_branch_need_min_gate 0.0
        --byot_branch_need_temperature "$PROXY_TEMPERATURE"
    )
}

job_needed() {
    local partition="$1" method="$2" seed="$3" output reference
    output="$(current_path "$partition" "$method" "$seed")"
    if [[ "$SKIP_EXISTING" == 1 ]] && pkl_complete "$output" "$ROUNDS"; then
        echo "[skip-current] $partition | $method | seed$seed" >&2
        return 1
    fi
    if [[ "$method" == full && "$REUSE_FULL_REFERENCE" == 1 ]]; then
        reference="$(full_reference_path "$partition" "$seed")"
        if pkl_complete "$reference" "$ROUNDS"; then
            echo "[reuse-full] $partition | seed$seed: $reference" >&2
            return 1
        fi
    fi
    return 0
}

run_job() {
    local gpu="$1" partition="$2" method="$3" seed="$4"
    local name rel output terminal
    local -a PARTITION_ARGS CMD

    validate_method "$method"
    mapfile -t PARTITION_ARGS < <(partition_flags "$partition")
    name="$(run_name "$partition" "$method" "$seed")"
    rel="${partition}/seed${seed}/${method}"
    output="${LOG_ROOT}/${rel}/${name}.pkl"
    terminal="${LOG_ROOT}/${rel}/${name}_terminal.log"
    mkdir -p "${LOG_ROOT}/${rel}"

    CMD=(
        "$PYTHON_BIN" main.py
        --dataset cifar100 --datadir ./data --in_channels 3 --num_classes 100
        "${PARTITION_ARGS[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS"
        --optimizer sgd --lr "$LR" --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --seed "$seed"
        --device "cuda:${gpu}" --sequential_client_execution
        --logdir "$LOG_ROOT" --log_file_name "${rel}/${name}"
        --model resnet18_byot --alg fedbyot
        --byot_active_branches 1,2,3
        --byot_branch_loss_reduction sum --byot_branch_objective kd_only
        --byot_beta "$FEATURE_BETA" --byot_teacher_source local
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
        --byot_alpha "$LAMBDA_MAX" --alpha_min_scale 0.0
    )

    [[ "$method" == wo_warmup ]] || append_warmup
    [[ "$method" == wo_reliability ]] || append_reliability
    [[ "$method" == wo_bias ]] || append_bias_correction
    [[ "$method" == wo_js_client ]] || append_js_client

    if [[ "$USE_WANDB" == 1 ]]; then
        CMD+=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
        [[ -n "${WANDB_ENTITY:-}" ]] && CMD+=(--wandb_entity "$WANDB_ENTITY")
    fi

    echo "[GPU $gpu] start: $partition | $method | seed$seed | R=$ROUNDS warm=$LAMBDA_WARMUP"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run] '; printf '%q ' "${CMD[@]}"; printf '\n'
        return 0
    fi
    if ! "${CMD[@]}" > "$terminal" 2>&1; then
        echo "[GPU $gpu] failed: $partition | $method | seed$seed; see $terminal" >&2
        tail -40 "$terminal" >&2 || true
        return 1
    fi
    if ! pkl_complete "$output" "$ROUNDS"; then
        echo "[GPU $gpu] incomplete output: $output" >&2
        return 1
    fi
    echo "[GPU $gpu] complete: $partition | $method | seed$seed"
}

for partition in "${PARTITIONS[@]}"; do partition_flags "$partition" >/dev/null; done
for method in "${METHODS[@]}"; do validate_method "$method"; done

declare -a JOBS=()
for partition in "${PARTITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for method in "${METHODS[@]}"; do
            if job_needed "$partition" "$method" "$seed"; then
                JOBS+=("$partition|$method|$seed")
            fi
        done
    done
done

echo "========== Final JS-client component ablation =========="
echo "GPUs=${GPUS[*]} | jobs=${#JOBS[@]}"
echo "CIFAR-100 / ResNet18-BYOT / FedAvg / R=$ROUNDS / E=$LOCAL_EPOCHS / C=$SAMPLE_FRACTION"
echo "partitions=${PARTITIONS[*]} | methods=${METHODS[*]} | seeds=${SEEDS[*]}"
echo "KD-only | T_KD=1 | T_proxy=1 | feature_beta=0 | min_require_size=$MIN_REQUIRE_SIZE"
echo "lambda_max=1 | warm=$LAMBDA_WARMUP | tau=.85 | skew_power=2 | JS-client gain=1"
echo "canonical execution: paired/preserved RNG controls are absent"
echo "log_root=$LOG_ROOT | completed PKLs are skipped"

if (( ${#JOBS[@]} == 0 )); then
    echo "No new jobs are required."
    exit 0
fi

worker() {
    local gpu="$1" slot="$2" i partition method seed failed=0
    for ((i=slot; i<${#JOBS[@]}; i+=NUM_GPUS)); do
        IFS='|' read -r partition method seed <<< "${JOBS[$i]}"
        run_job "$gpu" "$partition" "$method" "$seed" || failed=1
    done
    return "$failed"
}

declare -a PIDS=()
for i in "${!GPUS[@]}"; do
    worker "${GPUS[$i]}" "$i" &
    PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "At least one ablation run failed." >&2; exit 1; }
echo "Final JS-client component ablation complete."
