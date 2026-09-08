#!/usr/bin/env bash

# Shared worker for the final local-teacher JS-branch rerun matrix.
# Invoke through run_js_branch_final_reruns_{4gpu,2gpu}.sh unless a custom
# DATASETS/GROUPS split is needed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3}"
(( ${#GPUS[@]} > 0 )) || { echo "GPUS_OVERRIDE is empty." >&2; exit 1; }
NUM_GPUS=${#GPUS[@]}

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x venv/bin/python ]]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

RUN_SET="${RUN_SET:?Set RUN_SET to cifar_tiny_4gpu or image_2gpu}"
SEED="${SEED:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

NUM_CLIENTS="${NUM_CLIENTS:-100}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
MIN_REQUIRE_SIZE="${MIN_REQUIRE_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"

FEATURE_BETA="${FEATURE_BETA:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-1.0}"
PROXY_TEMPERATURE="${PROXY_TEMPERATURE:-1.0}"
LAMBDA_MAX="${LAMBDA_MAX:-1.0}"
SKEW_POWER="${SKEW_POWER:-2.0}"
SOFT_TAU="${SOFT_TAU:-0.85}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
JS_GAIN="${JS_GAIN:-1.0}"

FEDPROX_MU="${FEDPROX_MU:-0.01}"
MOON_MU="${MOON_MU:-0.01}"
MOON_TEMPERATURE="${MOON_TEMPERATURE:-0.5}"

TINYIMAGENET_DATADIR="${TINYIMAGENET_DATADIR:-./data/tiny-imagenet-200}"
if [[ -n "${IMAGENET100_DATADIR:-}" ]]; then
    :
elif [[ -d /data/imagenet100_resized_64_png/train ]]; then
    IMAGENET100_DATADIR="/data/imagenet100_resized_64_png"
else
    IMAGENET100_DATADIR="${HOME}/data/imagenet100_resized_64_png"
fi

LOG_ROOT="${LOG_ROOT:-logs/lambda/adaptive/logs_js_branch_final_reruns}"
mkdir -p "$LOG_ROOT"

configure_dataset() {
    DATASET="$1"
    case "$DATASET" in
        cifar10)
            DATA_DIR="./data"; NUM_CLASSES=10; ROUNDS=500; LR=0.1; NUM_WORKERS=0 ;;
        cifar100)
            DATA_DIR="./data"; NUM_CLASSES=100; ROUNDS=500; LR=0.1; NUM_WORKERS=0 ;;
        tinyimagenet)
            DATA_DIR="$TINYIMAGENET_DATADIR"; NUM_CLASSES=200; ROUNDS=100; LR=0.01; NUM_WORKERS=2 ;;
        imagenet100_64)
            DATA_DIR="$IMAGENET100_DATADIR"; NUM_CLASSES=100; ROUNDS=100; LR=0.01; NUM_WORKERS=2 ;;
        *) echo "Unknown dataset: $DATASET" >&2; return 1 ;;
    esac
    WARMUP_ROUNDS=$((ROUNDS / 2))
}

partition_args() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.5) printf '%s\n' --partition noniid --beta 0.5 ;;
        beta_0.3) printf '%s\n' --partition noniid --beta 0.3 ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) echo "Unknown partition: $1" >&2; return 1 ;;
    esac
}

has_completed_run() {
    local log_file="$1" pkl_file="$2" expected_rounds="$3"
    [[ -f "$log_file" ]] \
        && grep -q "Round $((expected_rounds - 1)) result" "$log_file" \
        && [[ -s "$pkl_file" ]]
}

append_warmup() {
    CMD+=(
        --byot_round_lambda_schedule linear
        --byot_round_lambda_min 0.0
        --byot_round_lambda_warmup "$WARMUP_ROUNDS"
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

append_bias() {
    CMD+=(
        --byot_client_skew_proxy prediction_entropy
        --byot_client_skew_power "$JOB_SKEW_POWER"
        --byot_client_skew_min_scale 0.0
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau "$JOB_SOFT_TAU"
        --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
    )
}

append_branch_gate() {
    CMD+=(
        --byot_branch_need_proxy js
        --byot_branch_need_gain "$JOB_JS_GAIN"
        --byot_branch_need_min_gate 0.0
        --byot_branch_need_temperature "$PROXY_TEMPERATURE"
    )
}

append_method_components() {
    local method="$1"
    case "$method" in
        full)
            append_warmup; append_reliability; append_bias; append_branch_gate ;;
        wo_warmup)
            append_reliability; append_bias; append_branch_gate ;;
        wo_reliability)
            append_warmup; append_bias; append_branch_gate ;;
        wo_bias)
            append_warmup; append_reliability; append_branch_gate ;;
        wo_gate)
            append_warmup; append_reliability; append_bias ;;
        *) echo "Unknown adaptive component method: $method" >&2; return 1 ;;
    esac
}

# Job format: scope|dataset|axis|value|partition|method
JOBS=()
add_job() { JOBS+=("$1|$2|$3|$4|$5|$6"); }

add_partitions() {
    local scope="$1" dataset="$2" axis="$3" value="$4" method="${5:-full}"
    shift 5
    local partition
    for partition in "$@"; do
        add_job "$scope" "$dataset" "$axis" "$value" "$partition" "$method"
    done
}

add_protocol_jobs() {
    local dataset="$1" mode="${2:-all}" value
    local -a epoch_values=()
    case "$mode" in
        all) epoch_values=(e1 e10) ;;
        no_e10) epoch_values=(e1) ;;
        e10_only) epoch_values=(e10) ;;
        *) echo "Unknown protocol subset: $mode" >&2; return 1 ;;
    esac
    for value in "${epoch_values[@]}"; do
        add_partitions protocol "$dataset" local_epochs "$value" full \
            iid beta_0.5 beta_0.3 beta_0.1
    done
    if [[ "$mode" != e10_only ]]; then
        for value in c0p05 c0p20; do
            add_partitions protocol "$dataset" participation "$value" full \
                iid beta_0.5 beta_0.3 beta_0.1
        done
    fi
}

add_default_model_mechanism_jobs() {
    local dataset="$1"
    add_partitions dataset "$dataset" default default full \
        iid beta_0.5 beta_0.3 beta_0.1
    add_partitions model "$dataset" model mobilenetv2 full \
        iid beta_0.5 beta_0.3 beta_0.1
    add_partitions mechanism "$dataset" mechanism fedprox full beta_0.5 beta_0.3
    add_partitions mechanism "$dataset" mechanism moon full beta_0.5 beta_0.3
}

case "$RUN_SET" in
    cifar_tiny_4gpu)
        # Rebuild the complete basic adaptive rows under the same legacy
        # execution protocol as the reusable Plain/fixed-lambda baselines.
        add_partitions base cifar10 default default full \
            iid beta_0.5 beta_0.3 beta_0.1
        add_partitions base cifar100 default default full \
            iid beta_0.5 beta_0.3 beta_0.1

        # Core one-factor sensitivity around the fixed final point (1,1).
        # The already completed (lmax=1, gain=1) rows are the references.
        for partition in beta_0.5 beta_0.1; do
            add_job sensitivity cifar100 sensitivity lmax0p5 "$partition" full
            add_job sensitivity cifar100 sensitivity lmax2p0 "$partition" full
            add_job sensitivity cifar100 sensitivity gain0p5 "$partition" full
            add_job sensitivity cifar100 sensitivity gain2p0 "$partition" full
        done

        # Component ablation: full rows already exist (the C100 beta=.3 full
        # row is the base job above), so run only the four missing controls.
        for method in wo_warmup wo_reliability wo_bias wo_gate; do
            add_partitions ablation cifar100 default default "$method" \
                iid beta_0.3 beta_0.1
        done
        # IID and beta=.1 paired full rows already exist in the JS-selection
        # logs; beta=.3 needs one paired full reference for this ablation.
        add_job ablation cifar100 default default beta_0.3 full

        # CIFAR-100: all previously reported OFAT axes, adaptive row only.
        add_partitions model cifar100 model mobilenetv2 full \
            iid beta_0.5 beta_0.3 beta_0.1
        add_partitions mechanism cifar100 mechanism fedprox full beta_0.5 beta_0.3
        add_partitions mechanism cifar100 mechanism moon full beta_0.5 beta_0.3
        add_protocol_jobs cifar100

        # Most TinyImageNet protocol jobs stay here; E=10 is moved to the
        # faster two-GPU server to balance total wall time.
        add_protocol_jobs tinyimagenet no_e10
        ;;
    image_2gpu)
        # The faster two-GPU server receives all ImageNet100-64 jobs plus the
        # shorter TinyImageNet default/model/mechanism jobs.
        add_default_model_mechanism_jobs tinyimagenet
        add_protocol_jobs tinyimagenet e10_only
        add_default_model_mechanism_jobs imagenet100_64
        add_protocol_jobs imagenet100_64
        ;;
    *) echo "Unknown RUN_SET: $RUN_SET" >&2; exit 1 ;;
esac

# Fail before occupying any GPU if a server does not have the requested
# image-folder dataset. CIFAR datasets are validated by their loader.
declare -A CHECKED_DATASETS=()
for job in "${JOBS[@]}"; do
    IFS='|' read -r _scope dataset _axis _value _partition _method <<< "$job"
    [[ -z "${CHECKED_DATASETS[$dataset]:-}" ]] || continue
    CHECKED_DATASETS[$dataset]=1
    configure_dataset "$dataset"
    case "$dataset" in
        tinyimagenet|imagenet100_64)
            if [[ ! -d "${DATA_DIR}/train" || ! -d "${DATA_DIR}/val" ]]; then
                echo "Dataset directories are missing for ${dataset}: ${DATA_DIR}/{train,val}" >&2
                exit 1
            fi
            ;;
    esac
done

configure_job() {
    local scope="$1" dataset="$2" axis="$3" value="$4" method="$5"
    configure_dataset "$dataset"

    EPOCHS=5
    PARTICIPATION="$SAMPLE_FRACTION"
    MODEL="resnet18_byot"
    LOSS_TEMPERATURE="$KD_TEMPERATURE"
    JOB_LAMBDA_MAX="$LAMBDA_MAX"
    JOB_JS_GAIN="$JS_GAIN"
    JOB_SOFT_TAU="$SOFT_TAU"
    JOB_SKEW_POWER="$SKEW_POWER"
    FL_ARGS=()

    case "$axis|$value" in
        local_epochs\|e1) EPOCHS=1 ;;
        local_epochs\|e10) EPOCHS=10 ;;
        participation\|c0p05) PARTICIPATION=0.05 ;;
        participation\|c0p20) PARTICIPATION=0.20 ;;
        model\|mobilenetv2) MODEL="mobilenet_byot" ;;
        mechanism\|fedprox) FL_ARGS=(--use_fedprox --mu "$FEDPROX_MU") ;;
        mechanism\|moon)
            LOSS_TEMPERATURE="$MOON_TEMPERATURE"
            FL_ARGS=(--use_moon --mu "$MOON_MU")
            ;;
        sensitivity\|lmax0p5) JOB_LAMBDA_MAX=0.5 ;;
        sensitivity\|lmax2p0) JOB_LAMBDA_MAX=2.0 ;;
        sensitivity\|gain0p5) JOB_JS_GAIN=0.5 ;;
        sensitivity\|gain2p0) JOB_JS_GAIN=2.0 ;;
        default\|default) ;;
        *) echo "Unknown axis/value: $axis/$value" >&2; return 1 ;;
    esac

    # The finalized MobileNet follow-up doubled the short-horizon datasets
    # from 100 to 200 rounds. CIFAR-100 MobileNet remains at 500 rounds.
    if [[ "$axis" == model && "$value" == mobilenetv2 && "$dataset" != cifar100 ]]; then
        ROUNDS=200
        WARMUP_ROUNDS=100
    fi

    if [[ "$scope" == ablation ]]; then
        METHOD="$method"
    else
        METHOD=full
    fi
}

job_name() {
    local scope="$1" dataset="$2" axis="$3" value="$4" partition="$5" method="$6"
    printf '%s_%s_%s_%s_%s_seed%s_r%s' \
        "$dataset" "$partition" "$scope" "$value" "$method" "$SEED" "$ROUNDS"
}

job_weight() {
    local job="$1" scope dataset axis value partition method weight
    IFS='|' read -r scope dataset axis value partition method <<< "$job"
    case "$dataset" in
        cifar10|cifar100) weight=180 ;;
        tinyimagenet) weight=280 ;;
        imagenet100_64) weight=340 ;;
    esac
    [[ "$axis" == model ]] && weight=$((weight * 2 / 3))
    [[ "$axis" == model && "$dataset" != cifar100 ]] && weight=$((weight * 2))
    [[ "$axis" == mechanism ]] && weight=$((weight + weight / 5))
    [[ "$axis" == local_epochs && "$value" == e1 ]] && weight=$((weight / 3))
    [[ "$axis" == local_epochs && "$value" == e10 ]] && weight=$((weight * 2))
    [[ "$axis" == participation && "$value" == c0p05 ]] && weight=$((weight / 2))
    [[ "$axis" == participation && "$value" == c0p20 ]] && weight=$((weight * 2))
    printf '%d' "$weight"
}

run_job() {
    local gpu="$1" job="$2"
    local scope dataset axis value partition method name rel_dir log_file pkl_file
    local -a partition_flags CMD paired_flags
    IFS='|' read -r scope dataset axis value partition method <<< "$job"
    configure_job "$scope" "$dataset" "$axis" "$value" "$method"
    name="$(job_name "$scope" "$dataset" "$axis" "$value" "$partition" "$method")"
    rel_dir="${scope}/${dataset}/${axis}/${value}/${partition}/seed${SEED}"
    log_file="${LOG_ROOT}/${rel_dir}/${name}.log"
    pkl_file="${LOG_ROOT}/${rel_dir}/${name}.pkl"
    mkdir -p "${LOG_ROOT}/${rel_dir}"

    if [[ "$SKIP_EXISTING" == 1 ]] && has_completed_run "$log_file" "$pkl_file" "$ROUNDS"; then
        echo "[GPU ${gpu}] skip: ${scope} | ${dataset} | ${axis}=${value} | ${partition} | ${method}"
        return 0
    fi

    mapfile -t partition_flags < <(partition_args "$partition")
    paired_flags=()
    # Sensitivity and component ablation use the paired-control protocol of
    # the JS-selection experiment. Base and extension rows intentionally keep
    # the older performance protocol so existing Plain/fixed rows are reusable.
    if [[ "$scope" == sensitivity || "$scope" == ablation ]]; then
        paired_flags=(--paired_execution_rng --preserve_byot_proxy_rng)
        if [[ "$MODEL" == resnet18_byot ]]; then
            paired_flags+=(--paired_resnet_init)
        fi
    fi

    CMD=(
        "$PYTHON_BIN" main.py
        --dataset "$DATASET" --datadir "$DATA_DIR" --in_channels 3 --num_classes "$NUM_CLASSES"
        "${partition_flags[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$PARTICIPATION"
        --epochs "$EPOCHS" --lr "$LR" --batch_size "$BATCH_SIZE"
        --test_batch_size "$TEST_BATCH_SIZE" --num_workers "$NUM_WORKERS"
        --round "$ROUNDS" --seed "$SEED" --device "cuda:${gpu}"
        --logdir "$LOG_ROOT" --log_file_name "${rel_dir}/${name}"
        --sequential_client_execution "${paired_flags[@]}"
        --model "$MODEL" --alg fedbyot "${FL_ARGS[@]}"
        --byot_active_branches 1,2,3 --byot_branch_loss_reduction sum
        --byot_branch_objective kd_only --byot_beta "$FEATURE_BETA"
        --byot_teacher_source local
        --temperature "$LOSS_TEMPERATURE"
        --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
        --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
        --byot_proxy_temperature "$PROXY_TEMPERATURE"
        --byot_alpha "$JOB_LAMBDA_MAX" --alpha_min_scale 0.0
    )
    append_method_components "$METHOD"

    echo "[GPU ${gpu}] start: ${scope} | ${dataset} | ${axis}=${value} | ${partition} | ${method} | R=${ROUNDS}, E=${EPOCHS}, C=${PARTICIPATION}"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run][GPU %s] ' "$gpu"
        printf '%q ' "${CMD[@]}"
        printf '\n'
        return 0
    fi

    if ! "${CMD[@]}" > "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" 2>&1; then
        echo "[GPU ${gpu}] failed: ${job}" >&2
        tail -40 "${LOG_ROOT}/${rel_dir}/${name}_terminal.log" >&2 || true
        return 1
    fi
    if ! has_completed_run "$log_file" "$pkl_file" "$ROUNDS"; then
        echo "[GPU ${gpu}] incomplete: ${name}" >&2
        return 1
    fi
    echo "[GPU ${gpu}] complete: ${scope} | ${dataset} | ${axis}=${value} | ${partition} | ${method}"
}

declare -a QUEUES LOADS
for ((i=0; i<NUM_GPUS; i++)); do QUEUES[$i]=''; LOADS[$i]=0; done
for job in "${JOBS[@]}"; do
    target=0
    for ((i=1; i<NUM_GPUS; i++)); do
        (( LOADS[i] < LOADS[target] )) && target=$i
    done
    weight="$(job_weight "$job")"
    QUEUES[$target]+="$job"$'\n'
    LOADS[$target]=$((LOADS[$target] + weight))
done

echo "========== Final JS-branch reruns: ${RUN_SET} =========="
echo "GPUs=${GPUS[*]} | jobs=${#JOBS[@]} | seed=${SEED}"
echo "protocol=min${MIN_REQUIRE_SIZE}, keep_last=0, lambda_max=${LAMBDA_MAX}, tau=${SOFT_TAU}, JS gain=${JS_GAIN}"
echo "Only final adaptive/ablation rows are run; Plain and fixed lambda=0.3 are reused."
echo "logs=${LOG_ROOT}"
for ((i=0; i<NUM_GPUS; i++)); do
    count=$(printf '%s' "${QUEUES[$i]}" | sed '/^$/d' | wc -l)
    echo "GPU ${GPUS[$i]}: ${count} jobs (relative load ${LOADS[$i]})"
done

run_queue() {
    local index="$1" gpu="${GPUS[$1]}" job failed=0
    while IFS= read -r job; do
        [[ -n "$job" ]] || continue
        run_job "$gpu" "$job" || failed=1
    done <<< "${QUEUES[$index]}"
    return "$failed"
}

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
    run_queue "$i" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"

if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry run complete."
else
    echo "Final JS-branch rerun set complete: ${RUN_SET}"
fi
