#!/usr/bin/env bash
# Terminal-RL Qwen3-8B a3s-code + DAPO SetA baseline.
#
# This wrapper keeps the shared PU training logic in terminal-rl_qwen3-8b_pu.sh
# and only pins the experiment defaults that define the a3s-code baseline.
#
# Usage:
#   bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh
#   DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ── Harness / data / algorithm defaults ──────────────────────────────
# terminal-rl_qwen3-8b_pu.sh includes this value as harness-a3s-code in
# the default RUN_NAME/RUN_ID.
export HARNESS_OPTION="${HARNESS_OPTION:-a3s-code}"
export DATASET="${DATASET:-seta}"
export ALGO="${ALGO:-dapo}"

# ── DAPO SetA baseline knobs ─────────────────────────────────────────
# Default to pure outcome reward. Enable ClawSentry explicitly with:
#   SETA_SAFETY=clawsentry SAFETY_REWARD_COEF=0.3
export SETA_SAFETY="${SETA_SAFETY:-none}"
export SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0}"
export MAX_TURN="${MAX_TURN:-10}"
export DAPO_EPS_CLIP_LOW="${DAPO_EPS_CLIP_LOW:-0.2}"
export DAPO_EPS_CLIP_HIGH="${DAPO_EPS_CLIP_HIGH:-0.28}"
export DAPO_CALCULATE_PER_TOKEN_LOSS="${DAPO_CALCULATE_PER_TOKEN_LOSS:-1}"
export DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-1}"
export DAPO_USE_KL_LOSS="${DAPO_USE_KL_LOSS:-0}"

exec bash "${SCRIPT_DIR}/terminal-rl_qwen3-8b_pu.sh" "$@"
