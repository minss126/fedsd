#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Standalone two-GPU configuration for the controlled local-data-size probe.
# Every condition starts from the same K=20 teacher-only global checkpoint.
# The optimizer-step budget is fixed, so smaller local datasets are reused for
# more effective epochs while total optimization work stays comparable.
export GPUS_OVERRIDE="${GPUS_OVERRIDE:-0 1}"
export BASE_CHECKPOINT="${BASE_CHECKPOINT:-logs/analysis/logs_iid_client_count_representation/cifar100_resnet18/iid/clients_20/fedavg/seed0/teacher_only_client_count_final.pt}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-logs/analysis/logs_local_data_size_internal_probe}"
export SAMPLE_SIZES_OVERRIDE="${SAMPLE_SIZES_OVERRIDE:-100 250 500 1000 2500}"
export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-0 1 2}"
export CLIENTS_PER_CONDITION="${CLIENTS_PER_CONDITION:-10}"
export TRAIN_BUDGET="${TRAIN_BUDGET:-steps}"
export LOCAL_STEPS="${LOCAL_STEPS:-100}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-50}"
export LOCAL_LR="${LOCAL_LR:-0.01}"
export MOMENTUM="${MOMENTUM:-0.9}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

exec "$SCRIPT_DIR/run_local_data_size_internal_probe.sh" "$@"
