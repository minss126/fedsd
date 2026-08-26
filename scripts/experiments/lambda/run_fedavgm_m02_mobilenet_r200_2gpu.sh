#!/usr/bin/env bash

# Re-run two cells of the final extension matrix with revised settings:
#   1) FedAvgM: server momentum 0.9 -> 0.2
#   2) MobileNetV2 on TinyImageNet/ImageNet100-64: 100 -> 200 rounds
#
# Every cell retains the final comparison methods:
#   plain, fixed lambda=0.3, final soft-b adaptive lambda.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  GPUS_OVERRIDE="0 1" \
    bash scripts/experiments/lambda/run_fedavgm_m02_mobilenet_r200_2gpu.sh

Default matrix:
  FedAvgM    : CIFAR-100, TinyImageNet, ImageNet100-64; all 4 partitions
  MobileNetV2: TinyImageNet, ImageNet100-64; all 4 partitions; 200 rounds
  Methods    : plain, fixed lambda=0.3, adaptive

Useful overrides:
  AXES_OVERRIDE="fedavgm"
  AXES_OVERRIDE="mobilenet"
  FEDAVGM_DATASETS_OVERRIDE="cifar100 tinyimagenet imagenet100_64"
  MOBILENET_DATASETS_OVERRIDE="tinyimagenet imagenet100_64"
  FEDAVGM_PARTITIONS_OVERRIDE="iid beta_0.3"
  MOBILENET_PARTITIONS_OVERRIDE="iid beta_0.3"
  MOBILENET_ROUNDS=200
  DRY_RUN=1
EOF
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1})
if (( ${#GPUS[@]} != 2 )); then
    echo "This launcher expects exactly two GPU ids; received: ${GPUS[*]}" >&2
    exit 1
fi
NUM_GPUS=${#GPUS[@]}

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

AXES=(${AXES_OVERRIDE:-fedavgm mobilenet})
METHODS=(${METHODS_OVERRIDE:-plain fixed_lambda0p30 adaptive})
FEDAVGM_DATASETS=(${FEDAVGM_DATASETS_OVERRIDE:-cifar100 tinyimagenet imagenet100_64})
MOBILENET_DATASETS=(${MOBILENET_DATASETS_OVERRIDE:-tinyimagenet imagenet100_64})
FEDAVGM_PARTITIONS=(${FEDAVGM_PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})
MOBILENET_PARTITIONS=(${MOBILENET_PARTITIONS_OVERRIDE:-iid beta_0.5 beta_0.3 beta_0.1})

SEED="${SEED:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
FIXED_LAMBDA="${FIXED_LAMBDA:-0.3}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
FEDAVGM_MOMENTUM="${FEDAVGM_MOMENTUM:-0.2}"
MOBILENET_ROUNDS="${MOBILENET_ROUNDS:-200}"

TINYIMAGENET_DATADIR="${TINYIMAGENET_DATADIR:-./data/tiny-imagenet-200}"
if [[ -n "${IMAGENET100_DATADIR:-}" ]]; then
    :
elif [[ -d /data/imagenet100_resized_64_png/train ]]; then
    IMAGENET100_DATADIR="/data/imagenet100_resized_64_png"
elif [[ -d "$HOME/data/imagenet100_resized_64_png/train" ]]; then
    IMAGENET100_DATADIR="$HOME/data/imagenet100_resized_64_png"
else
    IMAGENET100_DATADIR="/data/imagenet100_resized_64_png"
fi

LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_fedavgm_m02_mobilenet_r200}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

WANDB_FLAGS=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
fi

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

configure_dataset() {
    DATASET="$1"
    case "$DATASET" in
        cifar100)
            DATA_DIR="./data"
            NUM_CLASSES=100
            BASE_ROUNDS=500
            LR=0.1
            NUM_WORKERS=0
            ;;
        tinyimagenet)
            DATA_DIR="$TINYIMAGENET_DATADIR"
            NUM_CLASSES=200
            BASE_ROUNDS=100
            LR=0.01
            NUM_WORKERS=2
            ;;
        imagenet100_64)
            DATA_DIR="$IMAGENET100_DATADIR"
            NUM_CLASSES=100
            BASE_ROUNDS=100
            LR=0.01
            NUM_WORKERS=2
            ;;
        *) echo "Unknown dataset: $DATASET" >&2; return 1 ;;
    esac
}

validate_dataset() {
    local dataset="$1"
    configure_dataset "$dataset"
    case "$dataset" in
        tinyimagenet|imagenet100_64)
            if [[ ! -d "$DATA_DIR/train" || ! -d "$DATA_DIR/val" ]]; then
                echo "Dataset directories are missing for ${dataset}: ${DATA_DIR}/{train,val}" >&2
                return 1
            fi
            ;;
    esac
}

partition_flags() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

configure_axis() {
    local axis="$1" dataset="$2"
    configure_dataset "$dataset"
    case "$axis" in
        fedavgm)
            ROUNDS="$BASE_ROUNDS"
            WARMUP_ROUNDS=$((ROUNDS / 2))
            PLAIN_MODEL="resnet18"
            BYOT_MODEL="resnet18_byot"
            ;;
        mobilenet)
            if [[ "$dataset" == "cifar100" ]]; then
                echo "MobileNetV2 extension is intended for the former 100-round datasets." >&2
                return 1
            fi
            ROUNDS="$MOBILENET_ROUNDS"
            WARMUP_ROUNDS=$((ROUNDS / 2))
            PLAIN_MODEL="mobilenet"
            BYOT_MODEL="mobilenet_byot"
            ;;
        *) echo "Unknown axis: $axis" >&2; return 1 ;;
    esac
}

method_name() {
    local axis="$1" method="$2" suffix
    if [[ "$axis" == "fedavgm" ]]; then
        suffix="mom$(value_tag "$FEDAVGM_MOMENTUM")_r${ROUNDS}"
    else
        suffix="r${ROUNDS}"
    fi
    case "$method" in
        plain) printf 'plain_%s' "$suffix" ;;
        fixed_lambda0p30)
            printf 'fixed_lambda%s_tkd%s_%s' \
                "$(value_tag "$FIXED_LAMBDA")" "$(value_tag "$KD_TEMPERATURE")" "$suffix"
            ;;
        adaptive)
            printf 'soft_b_tkd%s_lmax%s_warm%s_tau%s_%s' \
                "$(value_tag "$KD_TEMPERATURE")" "$(value_tag "$LAMBDA_MAX")" \
                "$WARMUP_ROUNDS" "$(value_tag "$SOFT_TAU")" "$suffix"
            ;;
        *) echo "Unknown method: $method" >&2; return 1 ;;
    esac
}

pkl_complete() {
    local path="$1" expected_rounds="$2"
    [[ -s "$path" ]] || return 1
    "$PYTHON_BIN" -c '
import pickle, sys
try:
    with open(sys.argv[1], "rb") as handle:
        payload = pickle.load(handle)
    values = payload.get("acc_global", [])
    complete = isinstance(values, (list, tuple)) and len(values) >= int(sys.argv[2])
except Exception:
    complete = False
raise SystemExit(0 if complete else 1)
' "$path" "$expected_rounds"
}

job_output() {
    local axis="$1" dataset="$2" partition="$3" method="$4" name
    configure_axis "$axis" "$dataset"
    name="$(method_name "$axis" "$method")"
    printf '%s/%s/%s/%s/%s.pkl' "$LOG_ROOT" "$axis" "$dataset" "$partition" "$name"
}

job_weight() {
    local axis="$1" dataset="$2" method="$3"
    # Relative runtime estimates derived from the completed extension logs.
    if [[ "$axis" == "mobilenet" ]]; then
        case "${dataset}:${method}" in
            tinyimagenet:plain) echo 90 ;;
            tinyimagenet:fixed_lambda0p30) echo 130 ;;
            tinyimagenet:adaptive) echo 160 ;;
            imagenet100_64:plain) echo 210 ;;
            imagenet100_64:fixed_lambda0p30) echo 320 ;;
            imagenet100_64:adaptive) echo 370 ;;
        esac
    else
        case "${dataset}:${method}" in
            cifar100:plain) echo 35 ;;
            cifar100:fixed_lambda0p30) echo 50 ;;
            cifar100:adaptive) echo 60 ;;
            tinyimagenet:plain) echo 135 ;;
            tinyimagenet:fixed_lambda0p30) echo 190 ;;
            tinyimagenet:adaptive) echo 230 ;;
            imagenet100_64:plain) echo 165 ;;
            imagenet100_64:fixed_lambda0p30) echo 230 ;;
            imagenet100_64:adaptive) echo 275 ;;
        esac
    fi
}

append_adaptive_flags() {
    CMD+=(
        --byot_alpha "$LAMBDA_MAX"
        --byot_round_lambda_schedule linear
        --byot_round_lambda_min 0.00
        --byot_round_lambda_warmup "$WARMUP_ROUNDS"
        --alpha_min_scale 0.0
        --byot_client_proxy teacher_label_prob
        --byot_client_alpha_min 0.00
        --byot_client_alpha_max 1.00
        --byot_client_alpha_mode multiply
        --byot_client_reliability_power 1.0
        --byot_client_skew_proxy prediction_entropy
        --byot_client_skew_power "$SKEW_POWER"
        --byot_client_skew_min_scale 0.00
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau "$SOFT_TAU"
        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
    )
}

run_job() {
    local gpu_id="$1" axis="$2" dataset="$3" partition="$4" method="$5"
    local name log_dir output
    local -a PARTITION_FLAGS CMD

    configure_axis "$axis" "$dataset"
    name="$(method_name "$axis" "$method")"
    log_dir="${LOG_ROOT}/${axis}/${dataset}/${partition}"
    output="${log_dir}/${name}.pkl"
    mkdir -p "$log_dir"
    mapfile -t PARTITION_FLAGS < <(partition_flags "$partition")

    CMD=(
        "$PYTHON_BIN" main.py
        --dataset "$dataset" --datadir "$DATA_DIR"
        --in_channels 3 --num_classes "$NUM_CLASSES"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --epochs "$LOCAL_EPOCHS" --lr "$LR"
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED"
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT"
        --log_file_name "${axis}/${dataset}/${partition}/${name}"
    )
    CMD+=("${PARTITION_FLAGS[@]}")

    if [[ "$method" == "plain" ]]; then
        CMD+=(--model "$PLAIN_MODEL")
        if [[ "$axis" == "fedavgm" ]]; then
            CMD+=(--alg fedavgM --server_momentum "$FEDAVGM_MOMENTUM")
        else
            CMD+=(--alg fedavg)
        fi
    else
        CMD+=(
            --model "$BYOT_MODEL" --alg fedbyot
            --byot_active_branches "1,2,3"
            --byot_branch_loss_reduction sum
            --byot_branch_objective kd_only
            --byot_beta "$FEATURE_BETA"
            --temperature "$KD_TEMPERATURE"
            --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
            --byot_proxy_temperature "$PROXY_TEMPERATURE"
        )
        [[ "$axis" == "fedavgm" ]] && CMD+=(--server_momentum "$FEDAVGM_MOMENTUM")
        if [[ "$method" == "fixed_lambda0p30" ]]; then
            CMD+=(--byot_alpha "$FIXED_LAMBDA")
        else
            append_adaptive_flags
        fi
    fi
    CMD+=("${WANDB_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: ${axis} | ${dataset} | ${partition} | ${method} | R=${ROUNDS}, warm=${WARMUP_ROUNDS}"
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    if ! "${CMD[@]}" > "${log_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu_id}] failed: ${axis} | ${dataset} | ${partition} | ${method}" >&2
        tail -30 "${log_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! pkl_complete "$output" "$ROUNDS"; then
        echo "[GPU ${gpu_id}] incomplete result: ${output}" >&2
        return 1
    fi
    echo "[GPU ${gpu_id}] complete: ${axis} | ${dataset} | ${partition} | ${method}"
}

run_queue() {
    local gpu_id="$1"
    shift
    local job axis dataset partition method
    for job in "$@"; do
        [[ -n "$job" ]] || continue
        IFS='|' read -r axis dataset partition method <<< "$job"
        run_job "$gpu_id" "$axis" "$dataset" "$partition" "$method"
    done
}

RAW_JOBS=()
for axis in "${AXES[@]}"; do
    case "$axis" in
        fedavgm)
            for dataset in "${FEDAVGM_DATASETS[@]}"; do
                validate_dataset "$dataset"
                for partition in "${FEDAVGM_PARTITIONS[@]}"; do
                    for method in "${METHODS[@]}"; do
                        configure_axis "$axis" "$dataset"
                        output="$(job_output "$axis" "$dataset" "$partition" "$method")"
                        if [[ "$SKIP_EXISTING" == "1" ]] && pkl_complete "$output" "$ROUNDS"; then
                            echo "[skip] ${axis} | ${dataset} | ${partition} | ${method}"
                            continue
                        fi
                        RAW_JOBS+=("$(job_weight "$axis" "$dataset" "$method")|${axis}|${dataset}|${partition}|${method}")
                    done
                done
            done
            ;;
        mobilenet)
            for dataset in "${MOBILENET_DATASETS[@]}"; do
                validate_dataset "$dataset"
                for partition in "${MOBILENET_PARTITIONS[@]}"; do
                    for method in "${METHODS[@]}"; do
                        configure_axis "$axis" "$dataset"
                        output="$(job_output "$axis" "$dataset" "$partition" "$method")"
                        if [[ "$SKIP_EXISTING" == "1" ]] && pkl_complete "$output" "$ROUNDS"; then
                            echo "[skip] ${axis} | ${dataset} | ${partition} | ${method}"
                            continue
                        fi
                        RAW_JOBS+=("$(job_weight "$axis" "$dataset" "$method")|${axis}|${dataset}|${partition}|${method}")
                    done
                done
            done
            ;;
        *) echo "Unknown axis: $axis" >&2; exit 1 ;;
    esac
done

mapfile -t SORTED_JOBS < <(printf '%s\n' "${RAW_JOBS[@]}" | sort -t'|' -k1,1nr)
declare -a QUEUES=("" "")
declare -a QUEUE_WEIGHTS=(0 0)
for weighted_job in "${SORTED_JOBS[@]}"; do
    [[ -n "$weighted_job" ]] || continue
    IFS='|' read -r weight axis dataset partition method <<< "$weighted_job"
    if (( QUEUE_WEIGHTS[0] <= QUEUE_WEIGHTS[1] )); then gpu_idx=0; else gpu_idx=1; fi
    QUEUES[$gpu_idx]+="${axis}|${dataset}|${partition}|${method}"$'\n'
    QUEUE_WEIGHTS[$gpu_idx]=$((QUEUE_WEIGHTS[$gpu_idx] + weight))
done

echo "========== FedAvgM m=0.2 + MobileNetV2 R=${MOBILENET_ROUNDS} =========="
echo "gpus=${GPUS[*]}, new_jobs=${#SORTED_JOBS[@]}, seed=${SEED}"
echo "axes=${AXES[*]}, methods=${METHODS[*]}"
echo "fedavgm: datasets=${FEDAVGM_DATASETS[*]}, partitions=${FEDAVGM_PARTITIONS[*]}, momentum=${FEDAVGM_MOMENTUM}"
echo "mobilenet: datasets=${MOBILENET_DATASETS[*]}, partitions=${MOBILENET_PARTITIONS[*]}, rounds=${MOBILENET_ROUNDS}, warmup=$((MOBILENET_ROUNDS / 2))"
echo "queue_weights=${QUEUE_WEIGHTS[*]}, log_root=${LOG_ROOT}, dry_run=${DRY_RUN}"

if (( ${#SORTED_JOBS[@]} == 0 )); then
    echo "No new jobs are required."
    exit 0
fi

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    mapfile -t queue_jobs <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if (( status != 0 )); then
    echo "One or more extension jobs failed." >&2
    exit "$status"
fi

echo "FedAvgM/MobileNet extension re-run complete (${#SORTED_JOBS[@]} jobs)."
