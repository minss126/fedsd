#!/usr/bin/env bash

set -euo pipefail

# Secondary-server queue.  The C100 frozen probe is checkpoint-only: it never
# retrains teacher-only FL models, avoiding accidental duplicate 500-round runs
# when this script is used on a second server.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPUS=(${GPUS_OVERRIDE:-0 1})
if [ ${#GPUS[@]} -ne 2 ]; then
    echo "This queue expects exactly two GPU ids. Set GPUS_OVERRIDE='0 1'." >&2
    exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "venv/bin/python" ]; then PYTHON_BIN="venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

FROZEN_ROOT="${FROZEN_ROOT:-logs/analysis/logs_frozen_shallow_representation_probe_r500}"
RUN_FROZEN_PROBE="${RUN_FROZEN_PROBE:-1}"

run_cifar100_frozen_probe() {
    local seed=$1 gpu=$2
    local setting="${FROZEN_ROOT}/cifar100_resnet18/beta_0.5/fedavg/seed${seed}"
    local checkpoint="${setting}/teacher_only_frozen_shallow_probe_final.pt"
    local output="${setting}/teacher_only_frozen_shallow_probe_metrics.json"
    local terminal="${setting}/teacher_only_frozen_shallow_probe_probe_terminal.log"

    if [ -f "$output" ]; then
        echo "[GPU ${gpu}] C100 frozen probe already exists: seed${seed}"
        return
    fi
    if [ ! -f "$checkpoint" ]; then
        echo "Missing C100 frozen-probe checkpoint: $checkpoint" >&2
        echo "Copy the two checkpoint files from the primary server, or set RUN_FROZEN_PROBE=0." >&2
        return 2
    fi

    echo "[GPU ${gpu}] C100 frozen shallow probe: seed${seed}"
    "$PYTHON_BIN" scripts/experiments/analysis/frozen_shallow_representation_probe.py \
        --checkpoint "$checkpoint" --dataset cifar100 --datadir ./data \
        --device "cuda:${gpu}" --batch_size 128 --num_workers 0 \
        --probe_epochs 30 --probe_lr 0.05 --samples_per_class 500 \
        --branches 1,2,3 --seed "$seed" --output "$output" \
        > "$terminal" 2>&1
}

if [ "$RUN_FROZEN_PROBE" = "1" ]; then
    echo "========== Stage 1/3: C100 frozen probe =========="
    run_cifar100_frozen_probe 0 "${GPUS[0]}" &
    pid0=$!
    run_cifar100_frozen_probe 1 "${GPUS[1]}" &
    pid1=$!
    wait "$pid0"
    wait "$pid1"
fi

echo "========== Stage 2/3: branch target ablation =========="
GPUS_OVERRIDE="${GPUS[*]}" ./scripts/experiments/analysis/run_branch_target_ablation.sh

echo "========== Stage 3/3: paired branch gradient probe =========="
GPUS_OVERRIDE="${GPUS[*]}" ./scripts/experiments/analysis/run_paired_branch_gradient_probe.sh

echo "Secondary 2-GPU queue complete."
