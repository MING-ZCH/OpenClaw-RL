#!/usr/bin/env bash
# Terminal-RL x off-policy GRPO/DAPO launcher for SETA on Qwen3-8B.
#
# This is a thin wrapper over the current harness SETA DAPO script. It switches
# SLIME_DIR to the local slime_offpolicy backend and injects replay/off-policy
# algorithm flags through EXTRA_ALGO_ARGS.
#
# Examples:
#   OFFPOLICY_MODE=dapo  bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
#   OFFPOLICY_MODE=per   bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
#   OFFPOLICY_MODE=topr  bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
#   OFFPOLICY_MODE=spear bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
#
# Legacy compatibility:
#   OFFPOLICY_V2_MODE is accepted as an alias of OFFPOLICY_MODE.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh}"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[ERROR] Base script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

export SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/slime_offpolicy}"
if [[ ! -f "${SLIME_DIR}/train_async.py" ]]; then
  echo "[ERROR] slime_offpolicy backend not found at: ${SLIME_DIR}" >&2
  echo "        Set SLIME_DIR=/path/to/slime_offpolicy and retry." >&2
  exit 1
fi

export DATASET="${DATASET:-seta}"
export ALGO="${ALGO:-dapo}"
OFFPOLICY_MODE_RESOLVED="${OFFPOLICY_MODE:-${OFFPOLICY_V2_MODE:-dapo}}"
export OFFPOLICY_MODE="${OFFPOLICY_MODE_RESOLVED}"
export OFFPOLICY_V2_MODE="${OFFPOLICY_V2_MODE:-${OFFPOLICY_MODE_RESOLVED}}"

# Match the original off-policy experiment default: DAPO rejection sampling is on
# unless the caller explicitly disables it. OFFPOLICY_MODE=none keeps the base
# script's own default so it can be used as a baseline smoke test.
if [[ "${OFFPOLICY_MODE}" != "none" ]]; then
  export DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-1}"
fi

OFFPOLICY_CORE_ARGS=(
  "--loss-type" "decoupled_policy_loss"
  "--max-staleness" "${OFFPOLICY_MAX_STALENESS:-4}"
  "--importance-weight-clip-min" "${OFFPOLICY_IW_CLIP_MIN:-0.5}"
  "--importance-weight-clip-max" "${OFFPOLICY_IW_CLIP_MAX:-2.0}"
  "--behav-imp-weight-cap" "${OFFPOLICY_BEHAV_IW_CAP:-5.0}"
  "--enable-proximal-policy-storage"
  "--prox-logp-method" "${OFFPOLICY_PROX_LOGP_METHOD:-recompute}"
  "--buffer-mode" "${OFFPOLICY_BUFFER_MODE:-in_process}"
  "--buffer-max-size" "${OFFPOLICY_BUFFER_SIZE:-1024}"
  "--log-version-staleness-stats"
  "--log-proximal-approximation-metrics"
)

if [[ "${OFFPOLICY_ENABLE_M2PO:-1}" == "1" ]]; then
  OFFPOLICY_CORE_ARGS+=(
    "--enable-m2po-filtering"
    "--m2po-threshold" "${OFFPOLICY_M2PO_THRESHOLD:-0.16}"
  )
fi

OFFPOLICY_MODE_ARGS=()
case "${OFFPOLICY_MODE}" in
  none)
    OFFPOLICY_CORE_ARGS=()
    ;;
  dapo|dapo_only)
    OFFPOLICY_MODE_ARGS+=(
      "--enable-dynamic-sampling"
      "--dynamic-sample-min-std" "${OFFPOLICY_DAPO_MIN_STD:-${OFFPOLICY_V2_DAPO_MIN_STD:-1e-4}}"
    )
    ;;
  per|per_only)
    OFFPOLICY_MODE_ARGS+=(
      "--buffer-sampling-strategy" "per"
      "--per-alpha" "${OFFPOLICY_PER_ALPHA:-0.6}"
      "--per-beta-start" "${OFFPOLICY_PER_BETA_START:-0.4}"
      "--per-beta-end" "${OFFPOLICY_PER_BETA_END:-1.0}"
      "--per-beta-anneal-steps" "${OFFPOLICY_PER_BETA_ANNEAL_STEPS:-1000}"
      "--per-priority-source" "${OFFPOLICY_PER_PRIORITY_SOURCE:-advantage}"
    )
    ;;
  topr|topr_only)
    OFFPOLICY_MODE_ARGS+=(
      "--use-topr"
      "--topr-logw-cap" "${OFFPOLICY_TOPR_LOGW_CAP:-2.0}"
      "--topr-w-min" "${OFFPOLICY_TOPR_W_MIN:-0.0}"
      "--topr-w-max" "${OFFPOLICY_TOPR_W_MAX:-5.0}"
      "--topr-blend" "${OFFPOLICY_TOPR_BLEND:-1.0}"
    )
    ;;
  spear|spear_only)
    OFFPOLICY_MODE_ARGS+=(
      "--enable-trajectory-replay"
      "--trajectory-buffer-size" "${OFFPOLICY_SPEAR_BUF:-2048}"
      "--trajectory-score-threshold" "${OFFPOLICY_SPEAR_THRESH:-1.0}"
      "--replay-loss-coef" "${OFFPOLICY_SPEAR_COEF:-0.001}"
      "--max-replay-loss-steps" "${OFFPOLICY_SPEAR_STEPS:-200}"
      "--weight-decay-trajectory-replay" "${OFFPOLICY_SPEAR_DECAY:--1.0}"
    )
    ;;
  all3)
    OFFPOLICY_MODE_ARGS+=(
      "--enable-dynamic-sampling"
      "--dynamic-sample-min-std" "${OFFPOLICY_DAPO_MIN_STD:-1e-4}"
      "--buffer-sampling-strategy" "per"
      "--per-alpha" "${OFFPOLICY_PER_ALPHA:-0.6}"
      "--per-beta-start" "${OFFPOLICY_PER_BETA_START:-0.4}"
      "--per-beta-end" "${OFFPOLICY_PER_BETA_END:-1.0}"
      "--per-beta-anneal-steps" "${OFFPOLICY_PER_BETA_ANNEAL_STEPS:-1000}"
      "--per-priority-source" "${OFFPOLICY_PER_PRIORITY_SOURCE:-advantage}"
      "--use-topr"
      "--topr-logw-cap" "${OFFPOLICY_TOPR_LOGW_CAP:-2.0}"
      "--topr-w-min" "${OFFPOLICY_TOPR_W_MIN:-0.0}"
      "--topr-w-max" "${OFFPOLICY_TOPR_W_MAX:-5.0}"
      "--topr-blend" "${OFFPOLICY_TOPR_BLEND:-1.0}"
    )
    ;;
  *)
    echo "[ERROR] Unknown OFFPOLICY_MODE='${OFFPOLICY_MODE}'" >&2
    echo "        Valid choices: none|dapo|per|topr|spear|all3" >&2
    exit 1
    ;;
esac

ALL_OFFPOLICY_ARGS=("${OFFPOLICY_CORE_ARGS[@]}" "${OFFPOLICY_MODE_ARGS[@]}")
export EXTRA_ALGO_ARGS="${EXTRA_ALGO_ARGS:-} ${ALL_OFFPOLICY_ARGS[*]}"
export WANDB_GROUP="${WANDB_GROUP:-terminal_rl_qwen3-8b_offpolicy_${OFFPOLICY_MODE}}"
export WANDB_PROJECT="${WANDB_PROJECT:-terminal_rl}"
export RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS:-8}gpu_${DATASET}_offpolicy_${OFFPOLICY_MODE}_${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H%M%S)}}"
export RUN_NAME="${RUN_NAME:-${RUN_ID}}"

echo "============================================================"
echo "terminal-rl off-policy launcher"
echo "  REPO_ROOT        : ${REPO_ROOT}"
echo "  SLIME_DIR        : ${SLIME_DIR}"
echo "  BASE_SCRIPT      : ${BASE_SCRIPT}"
echo "  DATASET          : ${DATASET}"
echo "  ALGO             : ${ALGO}"
echo "  OFFPOLICY_MODE   : ${OFFPOLICY_MODE}"
echo "  DAPO_DYNAMIC     : ${DAPO_DYNAMIC_SAMPLING}"
echo "  EXTRA_ALGO_ARGS  : ${EXTRA_ALGO_ARGS}"
echo "============================================================"

exec bash "${BASE_SCRIPT}" "$@"
