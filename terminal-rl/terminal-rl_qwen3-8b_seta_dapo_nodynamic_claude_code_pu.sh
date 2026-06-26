#!/usr/bin/env bash
# SETA-only DAPO nodynamic baseline using the Claude Code harness.
#
# This wrapper mirrors terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh and only
# pins HARNESS_OPTION=claude_code. Claude Code CLI auth, endpoint, and model
# selection are read from environment variables by the shared base script.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-8}"

DATASET="${DATASET:-seta}"
ALGO="${ALGO:-dapo}"
HARNESS_OPTION="${HARNESS_OPTION:-claude_code}"
CLAUDE_CODE_LLM_BACKEND="${CLAUDE_CODE_LLM_BACKEND:-sglang}"
if [[ -z "${CLAUDE_CODE_MARK_NON_TRAINABLE+x}" ]]; then
  case "${CLAUDE_CODE_LLM_BACKEND}" in
    sglang|qwen|qwen-sglang|local|local-sglang)
      CLAUDE_CODE_MARK_NON_TRAINABLE="0"
      ;;
    *)
      CLAUDE_CODE_MARK_NON_TRAINABLE="1"
      ;;
  esac
fi
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_qwen3_think.yaml}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_TURN="${MAX_TURN:-10}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"

# Match the reference nodynamic baseline.
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
EXTRA_DAPO_ARGS="${EXTRA_DAPO_ARGS:-}"

RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_seta_dapo_nodynamic_think_harness-claude_code_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"

export RUN_TIMESTAMP NUM_GPUS RUN_ID RUN_NAME
export DATASET ALGO HARNESS_OPTION CUSTOM_CONFIG_PATH
export CLAUDE_CODE_LLM_BACKEND CLAUDE_CODE_MARK_NON_TRAINABLE
export ROLLOUT_BATCH_SIZE N_SAMPLES MAX_TURN MAX_CKPT_KEEP
export DAPO_DYNAMIC_SAMPLING EXTRA_DAPO_ARGS

exec bash "${SCRIPT_DIR}/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh" "$@"
