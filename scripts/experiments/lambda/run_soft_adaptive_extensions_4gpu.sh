#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# One-axis-at-a-time generalization screen for the selected soft-b adaptive
# lambda method.  The base method is fixed before every extension:
#   T_KD=1, T_proxy=1, lambda_max=1, tau=0.85, T_soft=0.05,
#   skew power=2, warm-up=250, kd_only BYOT branches.
#
# Every extension compares fixed lambda=1 and adaptive lambda under the same
# selected configuration on beta={0.5,0.1}.  Fixed lambda=1 is constant from
# round 0; only the adaptive method receives the selected lambda warm-up.
# The optional `plain` method is the matching non-BYOT baseline: no branches,
# no KD, and the underlying FL algorithm (FedAvg/FedProx/MOON) is used directly.
# This is a single-seed screening matrix; do not retune adaptive parameters
# per extension.
#
# Default extensions (20 jobs):
#   tinyimagenet, imagenet100_64, mobilenet, fedprox, moon.
# Each GPU receives five sequential jobs.  ImageNet100-64 is optional via
# EXTENSIONS_OVERRIDE if runtime becomes tight.

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: GPUS_OVERRIDE=\"0 1 2 3\" bash $0"
    echo
    echo "Default extensions: tinyimagenet imagenet100_64 mobilenet fedprox moon"
    echo "Exclude ImageNet100-64: EXTENSIONS_OVERRIDE=\"tinyimagenet mobilenet fedprox moon\" bash $0"
    echo "Default matrix: fixed lambda=1 vs. selected adaptive, beta={0.5,0.1}, seed=0."
    echo "Optional methods: METHODS_OVERRIDE=\"plain fixed_lambda1 soft_b_adaptive\""
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Set GPUS_OVERRIDE to one or more GPU ids." >&2
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
TINYIMAGENET_ROUNDS="${TINYIMAGENET_ROUNDS:-100}"
IMAGENET100_ROUNDS="${IMAGENET100_ROUNDS:-$ROUNDS}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.00}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.00}"
# Adaptive warm-up is expressed as a fraction of each setting's communication
# horizon.  An explicit LAMBDA_WARMUP remains available only as a legacy
# override for reproducing earlier runs.
LAMBDA_WARMUP_RATIO="${LAMBDA_WARMUP_RATIO:-0.5}"
LAMBDA_WARMUP_OVERRIDE="${LAMBDA_WARMUP:-}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
FEDPROX_MU="${FEDPROX_MU:-0.01}"
MOON_MU="${MOON_MU:-0.01}"
MOON_TEMPERATURE="${MOON_TEMPERATURE:-0.5}"
IMAGENET100_DATADIR="${IMAGENET100_DATADIR:-/data/imagenet100_resized_64_png}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_soft_adaptive_extensions_no_warmup_fixed}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ENVS=(${ENVS_OVERRIDE:-beta_0.5 beta_0.1})
METHODS=(${METHODS_OVERRIDE:-fixed_lambda1 soft_b_adaptive})
EXTENSIONS=(${EXTENSIONS_OVERRIDE:-tinyimagenet imagenet100_64 mobilenet fedprox moon})

WANDB_FLAGS=()
if [ "${USE_WANDB:-1}" = "1" ]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
fi

value_tag() {
    local formatted
    printf -v formatted '%.2f' "$1"
    printf '%s' "${formatted/./p}"
}

env_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown environment: $1" >&2; exit 1 ;;
    esac
}

configure_extension() {
    local extension=$1
    EXT_LABEL="$extension"
    DATASET="cifar100"
    DATA_DIR="./data"
    MODEL="resnet18_byot"
    LR="0.1"
    NUM_WORKERS="0"
    LOSS_TEMPERATURE="$KD_TEMPERATURE"
    EXT_ROUNDS="$ROUNDS"
    FL_ARGS=()

    case "$extension" in
        tinyimagenet)
            DATASET="tinyimagenet"
            DATA_DIR="./data/tiny-imagenet-200"
            LR="0.01"
            NUM_WORKERS="2"
            EXT_ROUNDS="$TINYIMAGENET_ROUNDS"
            ;;
        imagenet100_64)
            DATASET="imagenet100_64"
            DATA_DIR="$IMAGENET100_DATADIR"
            LR="0.01"
            NUM_WORKERS="2"
            EXT_ROUNDS="$IMAGENET100_ROUNDS"
            ;;
        mobilenet)
            MODEL="mobilenet_byot"
            ;;
        fedprox)
            FL_ARGS=(--use_fedprox --mu "$FEDPROX_MU")
            ;;
        moon)
            # Branch-KD temperatures remain explicitly fixed at 1.0 below.
            # --temperature is used only by MOON's contrastive regularizer.
            LOSS_TEMPERATURE="$MOON_TEMPERATURE"
            FL_ARGS=(--use_moon --mu "$MOON_MU")
            ;;
        *)
            echo "Unknown extension: $extension" >&2
            exit 1
            ;;
    esac

    if [ -n "$LAMBDA_WARMUP_OVERRIDE" ]; then
        EXT_LAMBDA_WARMUP="$LAMBDA_WARMUP_OVERRIDE"
    else
        EXT_LAMBDA_WARMUP=$(awk -v rounds="$EXT_ROUNDS" -v ratio="$LAMBDA_WARMUP_RATIO" 'BEGIN { printf "%d", int(rounds * ratio + 0.5) }')
    fi
}

method_name() {
    case "$1" in
        plain)
            echo "plain"
            ;;
        fixed_lambda0p10)
            echo "fixed_lambda0p10_tkd$(value_tag "$KD_TEMPERATURE")"
            ;;
        fixed_lambda0p30)
            echo "fixed_lambda0p30_tkd$(value_tag "$KD_TEMPERATURE")"
            ;;
        fixed_lambda1)
            echo "fixed_lambda1_tkd$(value_tag "$KD_TEMPERATURE")"
            ;;
        soft_b_adaptive)
            echo "soft_b_tkd$(value_tag "$KD_TEMPERATURE")_lmax$(value_tag "$LAMBDA_MAX")_warm${EXT_LAMBDA_WARMUP}_tau$(value_tag "$SOFT_TAU")"
            ;;
        *)
            echo "Unknown method: $1" >&2
            exit 1
            ;;
    esac
}

has_completed_log() {
    local log_file=$1
    local expected_rounds=$2
    [ -f "$log_file" ] && grep -q "Round $((expected_rounds - 1)) result" "$log_file"
}

append_method_args() {
    local method=$1
    case "$method" in
        plain)
            ;;
        fixed_lambda0p10)
            CMD+=(--byot_alpha 0.10)
            ;;
        fixed_lambda0p30)
            CMD+=(--byot_alpha 0.30)
            ;;
        fixed_lambda1)
            CMD+=(--byot_alpha 1.00)
            ;;
        soft_b_adaptive)
            CMD+=(
                --byot_alpha "$LAMBDA_MAX"
                --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00
                --byot_round_lambda_warmup "$EXT_LAMBDA_WARMUP"
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
            ;;
        *)
            echo "Unknown method: $method" >&2
            exit 1
            ;;
    esac
}

plain_model() {
    case "$MODEL" in
        resnet18_byot) echo "resnet18" ;;
        mobilenet_byot) echo "mobilenet" ;;
        *)
            echo "No plain-model mapping for ${MODEL}" >&2
            exit 1
            ;;
    esac
}

append_plain_algorithm_args() {
    case "$EXT_LABEL" in
        fedprox)
            CMD+=(--alg fedprox --mu "$FEDPROX_MU")
            ;;
        moon)
            CMD+=(--alg moon --mu "$MOON_MU" --temperature "$MOON_TEMPERATURE")
            ;;
        *)
            CMD+=(--alg fedavg)
            ;;
    esac
}

run_job() {
    local gpu_id=$1 extension=$2 env_name=$3 method=$4
    local name log_dir log_file
    local -a PARTITION_ARGS

    configure_extension "$extension"
    name="$(method_name "$method")_r${EXT_ROUNDS}"
    log_dir="${LOG_ROOT}/${EXT_LABEL}/${env_name}/fedavg"
    log_file="${log_dir}/${name}.log"
    mkdir -p "$log_dir"

    if [ "$SKIP_EXISTING" = "1" ] && has_completed_log "$log_file" "$EXT_ROUNDS"; then
        echo "[skip] ${extension} | ${env_name} | ${method}"
        return
    fi

    mapfile -t PARTITION_ARGS < <(env_args "$env_name")
    CMD=(
        "$PYTHON_BIN" main.py
        --dataset "$DATASET" --datadir "$DATA_DIR"
        --num_classes 100
        --n_clients 100 --sample_fraction 0.1
        --epochs "$LOCAL_EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --round "$EXT_ROUNDS" --seed "$SEED"
        --device "cuda:${gpu_id}" --logdir "$LOG_ROOT"
        --log_file_name "${EXT_LABEL}/${env_name}/fedavg/${name}"
    )
    CMD+=("${PARTITION_ARGS[@]}")

    if [ "$method" = "plain" ]; then
        CMD+=(--model "$(plain_model)")
        append_plain_algorithm_args
    else
        CMD+=(
            --model "$MODEL" --alg fedbyot
            --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum
            --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
            --temperature "$LOSS_TEMPERATURE"
            --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
            --byot_proxy_temperature "$PROXY_TEMPERATURE"
        )
        CMD+=("${FL_ARGS[@]}")
        append_method_args "$method"
    fi
    CMD+=("${WANDB_FLAGS[@]}")

    echo "[GPU ${gpu_id}] start: ${extension} | ${env_name} | ${method} | rounds=${EXT_ROUNDS}"
    "${CMD[@]}" > "${log_dir}/${name}_terminal.log" 2>&1
    echo "[GPU ${gpu_id}] complete: ${extension} | ${env_name} | ${method}"
}

run_queue() {
    local gpu_id=$1
    shift
    local job extension env_name method
    for job in "$@"; do
        [ -z "$job" ] && continue
        IFS='|' read -r extension env_name method <<< "$job"
        run_job "$gpu_id" "$extension" "$env_name" "$method"
    done
}

for extension in "${EXTENSIONS[@]}"; do
    configure_extension "$extension"
    case "$DATASET" in
        tinyimagenet|imagenet100_64)
            if [ ! -d "${DATA_DIR}/train" ] || [ ! -d "${DATA_DIR}/val" ]; then
                echo "Dataset directories are missing for ${extension}: ${DATA_DIR}/{train,val}" >&2
                exit 1
            fi
            ;;
    esac
done

declare -a QUEUES
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=""; done

job_count=0
for extension in "${EXTENSIONS[@]}"; do
    for env_name in "${ENVS[@]}"; do
        for method in "${METHODS[@]}"; do
            gpu_idx=$((job_count % NUM_GPUS))
            QUEUES[$gpu_idx]+="${extension}|${env_name}|${method}"$'\n'
            job_count=$((job_count + 1))
        done
    done
done

echo "========== Soft-b Extension Screen =========="
echo "gpus=${GPUS[*]}, jobs=${job_count}, default_rounds=${ROUNDS}, tinyimagenet_rounds=${TINYIMAGENET_ROUNDS}, imagenet100_rounds=${IMAGENET100_ROUNDS}, local_epochs=${LOCAL_EPOCHS}, seed=${SEED}"
echo "extensions=${EXTENSIONS[*]}"
echo "envs=${ENVS[*]}, methods=${METHODS[*]}"
echo "selected adaptive: T_KD=${KD_TEMPERATURE}, T_proxy=${PROXY_TEMPERATURE}, lambda_max=${LAMBDA_MAX}, tau=${SOFT_TAU}, warmup_ratio=${LAMBDA_WARMUP_RATIO}, warmup_override=${LAMBDA_WARMUP_OVERRIDE:-none}"
echo "FedProx mu=${FEDPROX_MU}; MOON mu=${MOON_MU}, contrastive temperature=${MOON_TEMPERATURE}"
echo "log_root=${LOG_ROOT}, skip_existing=${SKIP_EXISTING}, wandb=${USE_WANDB:-1}"

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    mapfile -t queue_jobs <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
    pids+=("$!")
done

wait "${pids[@]}"
echo "Soft-b extension screen complete (${job_count} jobs)"
