#!/usr/bin/env bash
# SETA-only DAPO baseline without dynamic sampling for Qwen3-8B.
#
# This wrapper pins DATASET=seta and delegates to the mixed nodynamic base
# script, while explicitly disabling DAPO rejection/dynamic sampling. It is
# intended for environment-stability runs where predictable rollout volume is
# preferred.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-8}"

DATASET="${DATASET:-seta}"
ALGO="${ALGO:-dapo}"
HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_qwen3_think.yaml}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_TURN="${MAX_TURN:-10}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"

# Critical baseline knob: do not add --dynamic-sampling-filter-path.
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"

# Keep EXTRA_DAPO_ARGS empty by default. Passing dynamic-sampling flags here
# would re-enable dynamic-sampling limits without the filter, which is not useful.
EXTRA_DAPO_ARGS="${EXTRA_DAPO_ARGS:-}"

RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_seta_dapo_nodynamic_think_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"

export RUN_TIMESTAMP NUM_GPUS RUN_ID RUN_NAME
export DATASET ALGO HARNESS_OPTION CUSTOM_CONFIG_PATH
export ROLLOUT_BATCH_SIZE N_SAMPLES MAX_TURN MAX_CKPT_KEEP
export DAPO_DYNAMIC_SAMPLING EXTRA_DAPO_ARGS

exec bash "${SCRIPT_DIR}/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh" "$@"
