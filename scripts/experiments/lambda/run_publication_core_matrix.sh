#!/usr/bin/env bash

# Final publication matrix, split across one 4-GPU and two 2-GPU servers.
#
#   resnet4 : ResNet18, all seeds (0,1,2)
#   mobile_a: MobileNetV2, seed 0 + half of seed 2
#   mobile_b: MobileNetV2, seed 1 + the other half of seed 2
#
# Matrix per model:
#   dataset   = CIFAR-100, TinyImageNet, ImageNet100-64
#   mechanism = FedAvg, FedProx, MOON
#   partition = IID, beta=.1
#   method    = Plain, final JS-client Adaptive
#
# Shorter datasets are deliberately queued first. Completed PKLs are skipped;
# an interrupted in-flight run is restarted from round zero.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

ROLE="${ROLE:?Set ROLE to resnet4, mobile_a, or mobile_b}"
read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1}"
(( ${#GPUS[@]} > 0 )) || { echo "GPUS_OVERRIDE is empty." >&2; exit 1; }

if [[ -n "${PYTHON_BIN:-}" ]]; then :
elif [[ -x venv/bin/python ]]; then PYTHON_BIN=venv/bin/python
else PYTHON_BIN=python3
fi

DATA_ROOT="${DATA_ROOT:-./data}"
TINYIMAGENET_DATADIR="${TINYIMAGENET_DATADIR:-${DATA_ROOT}/tiny-imagenet-200}"
IMAGENET100_DATADIR="${IMAGENET100_DATADIR:-}"
if [[ -z "$IMAGENET100_DATADIR" ]]; then
    for candidate in \
        "${DATA_ROOT}/imagenet100_resized_64_png" \
        "$HOME/data/imagenet100_resized_64_png" \
        "/data/imagenet100_resized_64_png"; do
        if [[ -d "$candidate/train" && -d "$candidate/val" ]]; then
            IMAGENET100_DATADIR="$candidate"
            break
        fi
    done
fi
IMAGENET100_DATADIR="${IMAGENET100_DATADIR:-${DATA_ROOT}/imagenet100_resized_64_png}"

LOG_ROOT="${LOG_ROOT:-logs/lambda/final/logs_publication_core_matrix}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REUSE_PRIOR="${REUSE_PRIOR:-1}"
DRY_RUN="${DRY_RUN:-0}"
USE_WANDB="${USE_WANDB:-0}"

NUM_CLIENTS=100
SAMPLE_FRACTION=0.1
LOCAL_EPOCHS=5
MIN_REQUIRE_SIZE=64
BATCH_SIZE=64
TEST_BATCH_SIZE=512

KD_TEMPERATURE=1.0
PROXY_TEMPERATURE=1.0
LAMBDA_MAX=1.0
SKEW_POWER=2.0
SOFT_TAU=0.85
SOFT_TEMPERATURE=0.05
JS_GAIN=1.0
FEDPROX_MU=0.01
MOON_MU=0.01
MOON_TEMPERATURE=0.5

case "$ROLE" in
    resnet4) MODEL_TAG=resnet18; PLAIN_MODEL=resnet18; BYOT_MODEL=resnet18_byot ;;
    mobile_a|mobile_b) MODEL_TAG=mobilenetv2; PLAIN_MODEL=mobilenet; BYOT_MODEL=mobilenet_byot ;;
    *) echo "Unknown ROLE=$ROLE (expected resnet4, mobile_a, mobile_b)." >&2; exit 2 ;;
esac

configure_dataset() {
    case "$1" in
        cifar100)
            DATA_DIR="$DATA_ROOT"; NUM_CLASSES=100; ROUNDS=500; LR=0.1; NUM_WORKERS=0 ;;
        tinyimagenet)
            DATA_DIR="$TINYIMAGENET_DATADIR"; NUM_CLASSES=200; ROUNDS=100; LR=0.01; NUM_WORKERS=2 ;;
        imagenet100_64)
            DATA_DIR="$IMAGENET100_DATADIR"; NUM_CLASSES=100; ROUNDS=100; LR=0.01; NUM_WORKERS=2 ;;
        *) echo "Unknown dataset: $1" >&2; return 1 ;;
    esac
    WARMUP_ROUNDS=$((ROUNDS / 2))
}

for dataset in tinyimagenet imagenet100_64; do
    configure_dataset "$dataset"
    [[ -d "$DATA_DIR/train" && -d "$DATA_DIR/val" ]] || {
        echo "Missing $dataset directories: $DATA_DIR/{train,val}" >&2
        exit 2
    }
done

partition_flags() {
    case "$1" in
        iid) printf '%s\n' --partition iid ;;
        beta_0.1) printf '%s\n' --partition noniid --beta 0.1 ;;
        *) return 1 ;;
    esac
}

pkl_complete() {
    local path="$1" expected="$2"
    [[ -s "$path" ]] || return 1
    "$PYTHON_BIN" -c '
import pickle, sys
try:
    with open(sys.argv[1], "rb") as f:
        x = pickle.load(f)
    n = int(sys.argv[2])
    ok = any(isinstance(x.get(k), (list, tuple)) and len(x[k]) >= n
             for k in ("acc_global", "branch_acc", "test_loss"))
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
' "$path" "$expected"
}

# Configuration-matched MobileNetV2 results already retained in the primary
# repository. These are intentionally identified by the full experimental
# cell, rather than by a loose filename search. The adaptive Tiny/ImageNet
# runs from the older soft-b/feature-loss protocol are not reusable.
prior_reusable() {
    local dataset="$1" mechanism="$2" partition="$3" method="$4" seed="$5"
    [[ "$REUSE_PRIOR" == 1 && "$MODEL_TAG" == mobilenetv2 ]] || return 1
    [[ "$mechanism" == fedavg && "$seed" == 0 ]] || return 1
    if [[ "$dataset" == cifar100 ]]; then
        [[ "$method" == plain || "$method" == adaptive ]]
        return
    fi
    if [[ "$dataset" == tinyimagenet || "$dataset" == imagenet100_64 ]]; then
        [[ "$method" == plain ]]
        return
    fi
    return 1
}

declare -a JOBS=()
seed2_index=0
for dataset in cifar100 tinyimagenet imagenet100_64; do
    for mechanism in fedavg fedprox moon; do
        for partition in iid beta_0.1; do
            for method in plain adaptive; do
                case "$ROLE" in
                    resnet4)
                        for seed in 0 1 2; do
                            JOBS+=("$dataset|$mechanism|$partition|$method|$seed")
                        done
                        ;;
                    mobile_a)
                        JOBS+=("$dataset|$mechanism|$partition|$method|0")
                        (( seed2_index % 2 == 0 )) && JOBS+=("$dataset|$mechanism|$partition|$method|2")
                        seed2_index=$((seed2_index + 1))
                        ;;
                    mobile_b)
                        JOBS+=("$dataset|$mechanism|$partition|$method|1")
                        (( seed2_index % 2 == 1 )) && JOBS+=("$dataset|$mechanism|$partition|$method|2")
                        seed2_index=$((seed2_index + 1))
                        ;;
                esac
            done
        done
    done
done

run_job() {
    local gpu="$1" job="$2"
    local dataset mechanism partition method seed tag name rel pkl terminal
    local -a part CMD
    IFS='|' read -r dataset mechanism partition method seed <<< "$job"
    configure_dataset "$dataset"
    mapfile -t part < <(partition_flags "$partition")

    if [[ "$method" == plain ]]; then tag=plain
    else tag=js_client_lmax1p00_tau0p85_tkd1p00_nofeat; fi
    name="${dataset}_${MODEL_TAG}_${mechanism}_${partition}_${tag}_canonical_seed${seed}_r${ROUNDS}"
    rel="${MODEL_TAG}/${dataset}/${mechanism}/${partition}/seed${seed}/${method}"
    pkl="${LOG_ROOT}/${rel}/${name}.pkl"
    terminal="${LOG_ROOT}/${rel}/${name}_terminal.log"

    if [[ "$SKIP_EXISTING" == 1 ]] && pkl_complete "$pkl" "$ROUNDS"; then
        echo "[GPU $gpu] skip: $job"
        return 0
    fi
    if prior_reusable "$dataset" "$mechanism" "$partition" "$method" "$seed"; then
        echo "[GPU $gpu] reuse prior: $job"
        return 0
    fi
    mkdir -p "${LOG_ROOT}/${rel}"

    CMD=(
        "$PYTHON_BIN" main.py
        --dataset "$dataset" --datadir "$DATA_DIR"
        --in_channels 3 --num_classes "$NUM_CLASSES"
        "${part[@]}" --min_require_size "$MIN_REQUIRE_SIZE"
        --n_clients "$NUM_CLIENTS" --sample_fraction "$SAMPLE_FRACTION"
        --round "$ROUNDS" --epochs "$LOCAL_EPOCHS"
        --optimizer sgd --lr "$LR" --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size "$BATCH_SIZE" --test_batch_size "$TEST_BATCH_SIZE"
        --num_workers "$NUM_WORKERS" --seed "$seed"
        --device "cuda:${gpu}" --sequential_client_execution
        --logdir "$LOG_ROOT" --log_file_name "${rel}/${name}"
    )

    if [[ "$method" == plain ]]; then
        CMD+=(--model "$PLAIN_MODEL")
        case "$mechanism" in
            fedavg) CMD+=(--alg fedavg) ;;
            fedprox) CMD+=(--alg fedprox --mu "$FEDPROX_MU") ;;
            moon) CMD+=(--alg moon --mu "$MOON_MU" --temperature "$MOON_TEMPERATURE") ;;
        esac
    else
        CMD+=(
            --model "$BYOT_MODEL" --alg fedbyot
            --byot_active_branches 1,2,3
            --byot_branch_loss_reduction sum --byot_branch_objective kd_only
            --byot_beta 0.0 --byot_teacher_source local --alpha_min_scale 0.0
            --byot_branch_kd_teacher_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_student_temperature "$KD_TEMPERATURE"
            --byot_branch_kd_loss_scale_mode native_t2
            --byot_proxy_temperature "$PROXY_TEMPERATURE"
            --byot_alpha "$LAMBDA_MAX"
            --byot_round_lambda_schedule linear --byot_round_lambda_min 0.0
            --byot_round_lambda_warmup "$WARMUP_ROUNDS"
            --byot_client_proxy teacher_label_prob
            --byot_client_alpha_min 0.0 --byot_client_alpha_max 1.0
            --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
            --byot_client_skew_proxy prediction_entropy
            --byot_client_skew_power "$SKEW_POWER" --byot_client_skew_min_scale 0.0
            --byot_client_skew_correction_mode soft_relax
            --byot_client_skew_soft_tau "$SOFT_TAU"
            --byot_client_skew_soft_temperature "$SOFT_TEMPERATURE"
            --byot_branch_need_proxy js_client --byot_branch_need_gain "$JS_GAIN"
            --byot_branch_need_min_gate 0.0 --byot_branch_need_temperature "$PROXY_TEMPERATURE"
        )
        case "$mechanism" in
            fedavg) ;;
            fedprox) CMD+=(--use_fedprox --mu "$FEDPROX_MU") ;;
            moon) CMD+=(--use_moon --mu "$MOON_MU" --temperature "$MOON_TEMPERATURE") ;;
        esac
    fi

    if [[ "$USE_WANDB" == 1 ]]; then
        CMD+=(--use_wandb --wandb_project "${WANDB_PROJECT:-dxfl}")
        [[ -n "${WANDB_ENTITY:-}" ]] && CMD+=(--wandb_entity "$WANDB_ENTITY")
    fi

    echo "[GPU $gpu] start: $job | R=$ROUNDS warm=$WARMUP_ROUNDS"
    if [[ "$DRY_RUN" == 1 ]]; then printf '[dry-run] '; printf '%q ' "${CMD[@]}"; printf '\n'; return 0; fi
    if ! "${CMD[@]}" > "$terminal" 2>&1; then
        echo "[GPU $gpu] failed: $job; see $terminal" >&2
        tail -40 "$terminal" >&2 || true
        return 1
    fi
    pkl_complete "$pkl" "$ROUNDS" || { echo "Incomplete output: $pkl" >&2; return 1; }
    echo "[GPU $gpu] complete: $job"
}

echo "========== Publication core matrix =========="
echo "role=$ROLE | model=$MODEL_TAG | GPUs=${GPUS[*]} | jobs=${#JOBS[@]}"
echo "short-to-long queue: CIFAR-100 -> TinyImageNet -> ImageNet100-64"
echo "completed PKLs are skipped; interrupted runs restart"
echo "reuse_prior=$REUSE_PRIOR (exact matched MobileNet seed-0 cells only)"

worker() {
    local gpu="$1" slot="$2" i failed=0
    for ((i=slot; i<${#JOBS[@]}; i+=${#GPUS[@]})); do
        run_job "$gpu" "${JOBS[$i]}" || failed=1
    done
    return "$failed"
}

pids=()
for i in "${!GPUS[@]}"; do worker "${GPUS[$i]}" "$i" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "At least one run failed." >&2; exit 1; }
echo "Publication core matrix complete for role=$ROLE."
