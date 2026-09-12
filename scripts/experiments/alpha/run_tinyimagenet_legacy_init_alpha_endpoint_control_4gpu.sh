#!/usr/bin/env bash

# Isolate the initialization change behind the old-vs-current TinyImageNet
# alpha trend. This delegates to the feature-on CE/KD endpoint screen while
# disabling only the deterministic name-paired ResNet reinitialization.
# Execution RNG pairing remains enabled, so alpha=0 and alpha=1 use the same
# client selections and minibatch RNG stream.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PAIRED_RESNET_INIT=0
export LOG_ROOT="${LOG_ROOT:-logs/alpha/logs_tinyimagenet_legacy_init_alpha_endpoint_control}"

exec bash "${SCRIPT_DIR}/run_tinyimagenet_feature_interaction_control_4gpu.sh" "$@"
