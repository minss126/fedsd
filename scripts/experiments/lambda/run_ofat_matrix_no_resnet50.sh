#!/usr/bin/env bash

# Complete only the missing cells of the non-factorial (one-factor-at-a-time)
# adaptive-lambda matrix, deliberately excluding ResNet-50.
#
# Fixed comparison axes for every scheduled cell:
#   partition = IID, beta={0.5, 0.3, 0.1}
#   method    = Plain, fixed lambda=0.3, soft-b adaptive
#
# Changed one factor at a time:
#   local epochs: E={1,10}; participation stays 0.1
#   participation: C={0.05,0.2}; local epochs stays 5
#   model: MobileNetV2-BYOT (ResNet18 is the shared default reference)
#   FL mechanism: FedProx, MOON, FedAvgM (ResNet18)
#
# The default skips known completed cells from prior experiment roots.  This
# is intentional: results need not live on the executing server to avoid
# rerunning an already completed, configuration-matched cell.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  GPUS_OVERRIDE="0 1 2 3" DATASETS_OVERRIDE="cifar100" \
    bash scripts/experiments/lambda/run_ofat_matrix_no_resnet50.sh

Overrides:
  DATASETS_OVERRIDE="cifar100 tinyimagenet imagenet100_64"
  AXES_OVERRIDE="protocol model mechanism"
  SKIP_PRIOR_COMPLETED=0   Re-run cells completed in earlier log roots.
  SKIP_EXISTING=0          Re-run cells already completed in this run root.
  DRY_RUN=1                Print only the missing-cell schedule.
EOF
    exit 0
fi

GPUS=(${GPUS_OVERRIDE:-0 1 2 3})
if (( ${#GPUS[@]} == 0 )); then
    echo "Set GPUS_OVERRIDE to one or more GPU ids." >&2
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

SEED="${SEED:-0}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
FIXED_LAMBDA="${FIXED_LAMBDA:-0.30}"
FEDPROX_MU="${FEDPROX_MU:-0.01}"
MOON_MU="${MOON_MU:-0.01}"
MOON_TEMPERATURE="${MOON_TEMPERATURE:-0.5}"
FEDAVGM_MOMENTUM="${FEDAVGM_MOMENTUM:-0.9}"

TINYIMAGENET_DATADIR="${TINYIMAGENET_DATADIR:-./data/tiny-imagenet-200}"
IMAGENET100_DATADIR="${IMAGENET100_DATADIR:-/data/imagenet100_resized_64_png}"
LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_ofat_matrix_no_resnet50}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_PRIOR_COMPLETED="${SKIP_PRIOR_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"

DATASETS=(${DATASETS_OVERRIDE:-cifar100 tinyimagenet imagenet100_64})
AXES=(${AXES_OVERRIDE:-protocol model mechanism})
PARTITIONS=(iid beta_0.5 beta_0.3 beta_0.1)
METHODS=(plain fixed_lambda0p30 soft_b_adaptive)

WANDB_FLAGS=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
    [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_FLAGS+=(--wandb_entity "$WANDB_ENTITY")
fi

value_tag() {
    local value formatted
    value="$1"
    printf -v formatted '%.2f' "$value"
    printf '%s' "${formatted/./p}"
}

configure_dataset() {
    DATASET="$1"
    case "$DATASET" in
        cifar100)
            DATA_DIR="./data"; NUM_CLASSES=100; ROUNDS=500; LR=0.1; NUM_WORKERS=0
            ;;
        tinyimagenet)
            DATA_DIR="$TINYIMAGENET_DATADIR"; NUM_CLASSES=200; ROUNDS=100; LR=0.01; NUM_WORKERS=2
            ;;
        imagenet100_64)
            DATA_DIR="$IMAGENET100_DATADIR"; NUM_CLASSES=100; ROUNDS=100; LR=0.01; NUM_WORKERS=2
            ;;
        *) echo "Unknown dataset: $DATASET" >&2; return 1 ;;
    esac
    WARMUP_ROUNDS=$((ROUNDS / 2))
}

for requested_dataset in "${DATASETS[@]}"; do
    configure_dataset "$requested_dataset"
    case "$requested_dataset" in
        tinyimagenet|imagenet100_64)
            if [[ ! -d "${DATA_DIR}/train" || ! -d "${DATA_DIR}/val" ]]; then
                echo "Dataset directories are missing for ${requested_dataset}: ${DATA_DIR}/{train,val}" >&2
                exit 1
            fi
            ;;
    esac
done

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

method_name() {
    case "$1" in
        plain) printf '%s' plain ;;
        fixed_lambda0p30) printf 'fixed_lambda0p30_tkd%s' "$(value_tag "$KD_TEMPERATURE")" ;;
        soft_b_adaptive) printf 'soft_b_tkd%s_lmax%s_warm%s_tau%s' \
            "$(value_tag "$KD_TEMPERATURE")" "$(value_tag "$LAMBDA_MAX")" \
            "$WARMUP_ROUNDS" "$(value_tag "$SOFT_TAU")" ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

pkl_complete() {
    local path="$1" expected_rounds="$2"
    [[ -s "$path" ]] || return 1
    "$PYTHON_BIN" -c '
import pickle, sys
try:
    with open(sys.argv[1], "rb") as f:
        payload = pickle.load(f)
    values = payload.get("acc_global", [])
    ok = isinstance(values, (list, tuple)) and len(values) >= int(sys.argv[2])
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
' "$path" "$expected_rounds"
}

# Returns success for cells that are already complete under exactly the final
# soft-b protocol, even if their files live on a different experiment server.
known_prior_complete() {
    local dataset="$1" axis="$2" value="$3" partition="$4" method="$5"
    local root candidate name
    configure_dataset "$dataset"
    name="$(method_name "$method")_r${ROUNDS}"

    case "$axis" in
        local_epochs|participation)
            # The earlier protocol extension completed beta={.5,.1} on C100.
            [[ "$dataset" == "cifar100" && ( "$partition" == "beta_0.5" || "$partition" == "beta_0.1" ) ]] || return 1
            candidate="logs/lambda/adaptive/logs_cifar100_protocol_extensions/${axis}/${value}/${partition}/fedavg/${name}.pkl"
            [[ -f "$candidate" || "$SKIP_PRIOR_COMPLETED" == "1" ]]
            ;;
        model_mobilenet)
            if [[ "$dataset" == "cifar100" ]]; then
                # C100 MobileNet is complete across all partitions, but the
                # three methods were intentionally stored in separate roots.
                return 0
            fi
            if [[ "$partition" == "iid" ]]; then
                candidate="logs/lambda/adaptive/logs_final_extension_matrix/model/mobilenet/${dataset}/iid/mobilenet/${name}.pkl"
                [[ -f "$candidate" || "$SKIP_PRIOR_COMPLETED" == "1" ]]
            else
                return 1
            fi
            ;;
        mechanism_fedprox|mechanism_moon)
            if [[ "$dataset" == "cifar100" ]]; then
                # C100 FedProx/MOON are complete across all partitions.
                return 0
            fi
            # Only two partial Tiny/ImageNet mechanism cells completed.
            if [[ "$axis" == "mechanism_fedprox" && "$partition" == "beta_0.5" ]]; then
                if [[ "$dataset" == "tinyimagenet" && "$method" == "soft_b_adaptive" ]] || \
                   [[ "$dataset" == "imagenet100_64" && "$method" == "plain" ]]; then
                    return 0
                fi
            fi
            return 1
            ;;
        mechanism_fedavgm)
            return 1
            ;;
        *) echo "Unknown axis: $axis" >&2; return 1 ;;
    esac
}

configure_axis() {
    AXIS="$1" VALUE="$2"
    EPOCHS="$LOCAL_EPOCHS"
    PARTICIPATION="$SAMPLE_FRACTION"
    BYOT_MODEL="resnet18_byot"
    PLAIN_MODEL="resnet18"
    BYOT_FL_ARGS=()
    PLAIN_FL_ARGS=()
    LOSS_TEMPERATURE="$KD_TEMPERATURE"

    case "$AXIS" in
        local_epochs)
            case "$VALUE" in e1|e10) EPOCHS="${VALUE#e}" ;; *) EPOCHS="$VALUE" ;; esac
            ;;
        participation)
            case "$VALUE" in
                c0p05) PARTICIPATION="0.05" ;;
                c0p20) PARTICIPATION="0.20" ;;
                *) PARTICIPATION="$VALUE" ;;
            esac
            ;;
        model_mobilenet)
            BYOT_MODEL="mobilenet_byot"; PLAIN_MODEL="mobilenet"
            ;;
        mechanism_fedprox)
            BYOT_FL_ARGS=(--use_fedprox --mu "$FEDPROX_MU")
            PLAIN_FL_ARGS=(--alg fedprox --mu "$FEDPROX_MU")
            ;;
        mechanism_moon)
            LOSS_TEMPERATURE="$MOON_TEMPERATURE"
            BYOT_FL_ARGS=(--use_moon --mu "$MOON_MU")
            PLAIN_FL_ARGS=(--alg moon --mu "$MOON_MU" --temperature "$MOON_TEMPERATURE")
            ;;
        mechanism_fedavgm)
            BYOT_FL_ARGS=(--server_momentum "$FEDAVGM_MOMENTUM")
            PLAIN_FL_ARGS=(--alg fedavgM --server_momentum "$FEDAVGM_MOMENTUM")
            ;;
        *) echo "Unknown axis: $AXIS" >&2; return 1 ;;
    esac
}

append_adaptive_args() {
    CMD+=(
        --byot_alpha "$LAMBDA_MAX"
        --byot_round_lambda_schedule linear --byot_round_lambda_min 0.00
        --byot_round_lambda_warmup "$WARMUP_ROUNDS" --alpha_min_scale 0.0
        --byot_client_proxy teacher_label_prob
        --byot_client_alpha_min 0.00 --byot_client_alpha_max 1.00
        --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
        --byot_client_skew_proxy prediction_entropy
        --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.00
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau "$SOFT_TAU"
        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
    )
}

run_job() {
    local gpu_id="$1" dataset="$2" axis="$3" value="$4" partition="$5" method="$6"
    local name log_dir output
    local -a PARTITION_FLAGS CMD
    configure_dataset "$dataset"
    configure_axis "$axis" "$value"
    name="$(method_name "$method")_r${ROUNDS}"
    log_dir="${LOG_ROOT}/${dataset}/${axis}/${value}/${partition}/${axis}"
    output="${log_dir}/${name}.pkl"

    if [[ "$SKIP_EXISTING" == "1" ]] && pkl_complete "$output" "$ROUNDS"; then
        echo "[GPU ${gpu_id}] skip-current: ${dataset} | ${axis}=${value} | ${partition} | ${method}"
        return 0
    fi
    if [[ "$SKIP_PRIOR_COMPLETED" == "1" ]] && known_prior_complete "$dataset" "$axis" "$value" "$partition" "$method"; then
        echo "[GPU ${gpu_id}] skip-prior: ${dataset} | ${axis}=${value} | ${partition} | ${method}"
        return 0
    fi
    mkdir -p "$log_dir"
    mapfile -t PARTITION_FLAGS < <(partition_args "$partition")
    CMD=(
        "$PYTHON_BIN" main.py
        --dataset "$dataset" --datadir "$DATA_DIR" --in_channels 3 --num_classes "$NUM_CLASSES"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$PARTICIPATION"
        --epochs "$EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --round "$ROUNDS" --seed "$SEED" --device "cuda:${gpu_id}"
        --logdir "$LOG_ROOT" --log_file_name "${dataset}/${axis}/${value}/${partition}/${axis}/${name}"
    )
    CMD+=("${PARTITION_FLAGS[@]}")
    if [[ "$method" == "plain" ]]; then
        CMD+=(--model "$PLAIN_MODEL")
        if (( ${#PLAIN_FL_ARGS[@]} == 0 )); then CMD+=(--alg fedavg); else CMD+=("${PLAIN_FL_ARGS[@]}"); fi
    else
        CMD+=(
            --model "$BYOT_MODEL" --alg fedbyot
            --byot_active_branches "1,2,3" --byot_branch_loss_reduction sum
            --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
            --temperature "$LOSS_TEMPERATURE"
            --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
            --byot_proxy_temperature "$PROXY_TEMPERATURE"
        )
        CMD+=("${BYOT_FL_ARGS[@]}")
        if [[ "$method" == "fixed_lambda0p30" ]]; then
            CMD+=(--byot_alpha "$FIXED_LAMBDA")
        else
            append_adaptive_args
        fi
    fi
    CMD+=("${WANDB_FLAGS[@]}")
    echo "[GPU ${gpu_id}] start: ${dataset} | ${axis}=${value} | ${partition} | ${method} | R=${ROUNDS}, E=${EPOCHS}, C=${PARTICIPATION}"
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    if ! "${CMD[@]}" > "${log_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu_id}] failed: ${dataset} | ${axis}=${value} | ${partition} | ${method}" >&2
        tail -25 "${log_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! pkl_complete "$output" "$ROUNDS"; then
        echo "[GPU ${gpu_id}] incomplete result: ${output}" >&2
        return 1
    fi
    echo "[GPU ${gpu_id}] complete: ${dataset} | ${axis}=${value} | ${partition} | ${method}"
}

declare -a JOBS
for dataset in "${DATASETS[@]}"; do
    for requested_axis in "${AXES[@]}"; do
        case "$requested_axis" in
            protocol)
                for value in e1 e10; do
                    for partition in "${PARTITIONS[@]}"; do for method in "${METHODS[@]}"; do
                        JOBS+=("${dataset}|local_epochs|${value}|${partition}|${method}")
                    done; done
                done
                for value in c0p05 c0p20; do
                    for partition in "${PARTITIONS[@]}"; do for method in "${METHODS[@]}"; do
                        JOBS+=("${dataset}|participation|${value}|${partition}|${method}")
                    done; done
                done
                ;;
            model)
                for partition in "${PARTITIONS[@]}"; do for method in "${METHODS[@]}"; do
                    JOBS+=("${dataset}|model_mobilenet|mobilenetv2|${partition}|${method}")
                done; done
                ;;
            mechanism)
                for mechanism in mechanism_fedprox mechanism_moon mechanism_fedavgm; do
                    for partition in "${PARTITIONS[@]}"; do for method in "${METHODS[@]}"; do
                        JOBS+=("${dataset}|${mechanism}|default|${partition}|${method}")
                    done; done
                done
                ;;
            *) echo "Unknown requested axis: $requested_axis" >&2; exit 1 ;;
        esac
    done
done

# Predicted duration weights are only for queue balance; all actual skips are
# resolved inside run_job.  ImageNet and adaptive jobs receive more weight.
job_weight() {
    local job="$1" dataset axis value partition method
    IFS='|' read -r dataset axis value partition method <<< "$job"
    local weight=100
    [[ "$dataset" == "tinyimagenet" ]] && weight=240
    [[ "$dataset" == "imagenet100_64" ]] && weight=300
    [[ "$axis" == "local_epochs" && "$value" == "e10" ]] && weight=$((weight * 2))
    [[ "$axis" == "participation" && "$value" == "c0p20" ]] && weight=$((weight * 2))
    [[ "$method" == "soft_b_adaptive" ]] && weight=$((weight + weight / 4))
    printf '%d' "$weight"
}

declare -a QUEUES LOADS
for ((i = 0; i < NUM_GPUS; i++)); do QUEUES[$i]=''; LOADS[$i]=0; done
for job in "${JOBS[@]}"; do
    target=0
    for ((i = 1; i < NUM_GPUS; i++)); do (( LOADS[i] < LOADS[target] )) && target=$i; done
    weight="$(job_weight "$job")"
    QUEUES[$target]+="$job"$'\n'
    LOADS[$target]=$((LOADS[$target] + weight))
done

run_queue() {
    local gpu_id="$1"
    shift
    local job dataset axis value partition method failed=0
    for job in "$@"; do
        [[ -z "$job" ]] && continue
        IFS='|' read -r dataset axis value partition method <<< "$job"
        if ! run_job "$gpu_id" "$dataset" "$axis" "$value" "$partition" "$method"; then failed=1; fi
    done
    return "$failed"
}

echo "========== OFAT matrix completion (ResNet50 excluded) =========="
echo "datasets=${DATASETS[*]}, requested_axes=${AXES[*]}, candidate_jobs=${#JOBS[@]}"
echo "methods=${METHODS[*]}, partitions=${PARTITIONS[*]}, seed=${SEED}"
echo "adaptive: T_KD=${KD_TEMPERATURE}, T_proxy=${PROXY_TEMPERATURE}, lmax=${LAMBDA_MAX}, p=${SKEW_POWER}, tau=${SOFT_TAU}, T_soft=${SOFT_TEMPERATURE}, warmup=0.5R"
echo "skip_current=${SKIP_EXISTING}, skip_prior=${SKIP_PRIOR_COMPLETED}, logs=${LOG_ROOT}"
for ((i = 0; i < NUM_GPUS; i++)); do
    count=$(printf '%s' "${QUEUES[$i]}" | sed '/^$/d' | wc -l)
    echo "GPU ${GPUS[$i]}: ${count} candidate jobs (relative load ${LOADS[$i]})"
done

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    mapfile -t queue_jobs <<< "${QUEUES[$i]}"
    run_queue "${GPUS[$i]}" "${queue_jobs[@]}" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then
    echo "One or more jobs failed. Completed results remain in ${LOG_ROOT}." >&2
    exit 1
fi
echo "OFAT matrix completion complete."
