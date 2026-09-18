#!/usr/bin/env bash

# Memory smoke test for the reduced ResNet50 publication scope.
# GPU 0: TinyImageNet IID adaptive; GPU 1: ImageNet100-64 IID adaptive.
# Try batch 64 first and retry batch 32 only when CUDA OOM is detected.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1}"
(( ${#GPUS[@]} >= 2 )) || { echo "Two GPU ids are required." >&2; exit 2; }

if [[ -n "${PYTHON_BIN:-}" ]]; then :
elif [[ -x venv/bin/python ]]; then PYTHON_BIN=venv/bin/python
else PYTHON_BIN=python3
fi

DATA_ROOT="${DATA_ROOT:-./data}"
TINY_DIR="${TINYIMAGENET_DATADIR:-${DATA_ROOT}/tiny-imagenet-200}"
IMAGE_DIR="${IMAGENET100_DATADIR:-}"
if [[ -z "$IMAGE_DIR" ]]; then
    for candidate in \
        "${DATA_ROOT}/imagenet100_resized_64_png" \
        "$HOME/data/imagenet100_resized_64_png" \
        "/data/imagenet100_resized_64_png"; do
        if [[ -d "$candidate/train" && -d "$candidate/val" ]]; then
            IMAGE_DIR="$candidate"
            break
        fi
    done
fi
IMAGE_DIR="${IMAGE_DIR:-${DATA_ROOT}/imagenet100_resized_64_png}"

[[ -d "$TINY_DIR/train" && -d "$TINY_DIR/val" ]] || {
    echo "Missing TinyImageNet: $TINY_DIR/{train,val}" >&2; exit 2;
}
[[ -d "$IMAGE_DIR/train" && -d "$IMAGE_DIR/val" ]] || {
    echo "Missing ImageNet100-64: $IMAGE_DIR/{train,val}" >&2; exit 2;
}

LOG_ROOT="${LOG_ROOT:-logs/lambda/smoke/resnet50_adaptive_memory}"
DRY_RUN="${DRY_RUN:-0}"

run_attempt() {
    local gpu="$1" dataset="$2" datadir="$3" classes="$4" batch="$5"
    local stem="${dataset}_resnet50_adaptive_iid_seed0_r2_b${batch}"
    local dir="${LOG_ROOT}/${dataset}/batch${batch}"
    local terminal="${dir}/${stem}_terminal.log"
    mkdir -p "$dir"

    local -a cmd=(
        "$PYTHON_BIN" main.py
        --dataset "$dataset" --datadir "$datadir"
        --in_channels 3 --num_classes "$classes"
        --model resnet50_byot --alg fedbyot
        --partition iid --min_require_size 64
        --n_clients 100 --sample_fraction 0.1
        --round 2 --epochs 5
        --optimizer sgd --lr 0.01 --momentum 0.9 --reg 0.001
        --scheduler round --schedule_round 1 --lr_gamma 0.998
        --batch_size "$batch" --test_batch_size 256
        --num_workers 2 --seed 0 --device "cuda:${gpu}"
        --sequential_client_execution
        --logdir "$LOG_ROOT" --log_file_name "${dataset}/batch${batch}/${stem}"
        --byot_active_branches 1,2,3
        --byot_branch_loss_reduction sum --byot_branch_objective kd_only
        --byot_beta 0.0 --byot_teacher_source local --alpha_min_scale 0.0
        --byot_branch_kd_teacher_temperature 1.0
        --byot_branch_kd_student_temperature 1.0
        --byot_branch_kd_loss_scale_mode native_t2
        --byot_proxy_temperature 1.0 --byot_alpha 1.0
        --byot_round_lambda_schedule linear --byot_round_lambda_min 0.0
        --byot_round_lambda_warmup 1
        --byot_client_proxy teacher_label_prob
        --byot_client_alpha_min 0.0 --byot_client_alpha_max 1.0
        --byot_client_alpha_mode multiply --byot_client_reliability_power 1.0
        --byot_client_skew_proxy prediction_entropy
        --byot_client_skew_power 2.0 --byot_client_skew_min_scale 0.0
        --byot_client_skew_correction_mode soft_relax
        --byot_client_skew_soft_tau 0.85
        --byot_client_skew_soft_temperature 0.05
        --byot_branch_need_proxy js_client --byot_branch_need_gain 1.0
        --byot_branch_need_min_gate 0.0 --byot_branch_need_temperature 1.0
    )

    echo "[GPU $gpu] smoke start: $dataset | batch=$batch"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '[dry-run] '; printf '%q ' "${cmd[@]}"; printf '\n'
        return 0
    fi
    "${cmd[@]}" > "$terminal" 2>&1
}

run_dataset() {
    local gpu="$1" dataset="$2" datadir="$3" classes="$4"
    if run_attempt "$gpu" "$dataset" "$datadir" "$classes" 64; then
        echo "[GPU $gpu] PASS: $dataset batch=64"
        return 0
    fi
    local log64="${LOG_ROOT}/${dataset}/batch64/${dataset}_resnet50_adaptive_iid_seed0_r2_b64_terminal.log"
    if ! grep -Eqi 'out of memory|torch\.OutOfMemoryError|CUDA error: out of memory' "$log64"; then
        echo "[GPU $gpu] non-OOM failure: $dataset batch=64; inspect $log64" >&2
        return 1
    fi
    echo "[GPU $gpu] OOM: $dataset batch=64; retrying batch=32"
    if run_attempt "$gpu" "$dataset" "$datadir" "$classes" 32; then
        echo "[GPU $gpu] PASS: $dataset batch=32"
        return 0
    fi
    echo "[GPU $gpu] FAIL: $dataset batch=32" >&2
    return 1
}

run_dataset "${GPUS[0]}" tinyimagenet "$TINY_DIR" 200 & p0=$!
run_dataset "${GPUS[1]}" imagenet100_64 "$IMAGE_DIR" 100 & p1=$!

status=0
wait "$p0" || status=1
wait "$p1" || status=1
(( status == 0 )) || exit "$status"
echo "ResNet50 adaptive memory smoke test complete."
