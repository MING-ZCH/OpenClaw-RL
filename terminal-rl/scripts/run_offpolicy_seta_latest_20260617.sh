#!/usr/bin/env bash
# One-command launcher for terminal-rl + SETA off-policy ablations.
#
# Usage:
#   bash terminal-rl/scripts/run_offpolicy_seta_latest_20260617.sh dapo
#   bash terminal-rl/scripts/run_offpolicy_seta_latest_20260617.sh spear
#   OFFPOLICY_MODE=per bash terminal-rl/scripts/run_offpolicy_seta_latest_20260617.sh
#
# Valid modes: none, dapo, per, topr, spear, all3.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}/terminal-rl"

MODE="${1:-${OFFPOLICY_MODE:-dapo}}"
case "${MODE}" in
  none|dapo|per|topr|spear|all3) ;;
  *)
    echo "[ERROR] Unknown mode='${MODE}'. Use: none|dapo|per|topr|spear|all3" >&2
    exit 1
    ;;
esac

# Keep worker selection reproducible but repo-generic. Set WORKER_URLS in the
# shell, or set OFFPOLICY_DEFAULT_WORKER_URLS for a local cluster default.
DEFAULT_WORKER_URLS="${OFFPOLICY_DEFAULT_WORKER_URLS:-}"
if [[ -n "${WORKER_URLS:-}" ]]; then
  export WORKER_URLS
elif [[ -n "${DEFAULT_WORKER_URLS}" && "${OFFPOLICY_USE_DEFAULT_WORKERS:-1}" == "1" ]]; then
  export WORKER_URLS="${DEFAULT_WORKER_URLS}"
else
  export WORKER_URLS=""
fi
# Let the base launcher choose a per-host Python that can import
# aiohttp/fastapi/uvicorn. Training pods share the repo filesystem but not
# necessarily /root/miniconda3 site-packages with this dev host.
export ROUTER_LOAD_AWARE_ALLOCATE="${ROUTER_LOAD_AWARE_ALLOCATE:-1}"
export ENV_ALLOCATE_MAX_RETRIES="${ENV_ALLOCATE_MAX_RETRIES:-180}"
export ENV_ALLOCATE_RETRY_BACKOFF="${ENV_ALLOCATE_RETRY_BACKOFF:-4.0}"
export ENV_ALLOCATE_RETRY_MAX_DELAY="${ENV_ALLOCATE_RETRY_MAX_DELAY:-20.0}"
export ENV_ALLOCATE_RETRY_STATUSES="${ENV_ALLOCATE_RETRY_STATUSES:-429,500,502,503,504}"
export ENV_ALLOCATE_NON_RETRY_STATUSES="${ENV_ALLOCATE_NON_RETRY_STATUSES:-400,401,403,404,409,422}"

export OFFPOLICY_MODE="${MODE}"
export OFFPOLICY_V2_MODE="${OFFPOLICY_V2_MODE:-${MODE}}"
export OFFPOLICY_USE_INTEGRATED_SLIME="${OFFPOLICY_USE_INTEGRATED_SLIME:-1}"
export SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/slime}"
export ALGO="${ALGO:-dapo}"
export DATASET="${DATASET:-seta}"
export SETA_SAFETY="${SETA_SAFETY:-clawsentry}"
export SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0.3}"
export MAX_TURN="${MAX_TURN:-10}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${REPO_ROOT}/terminal-rl/configs/rollout_qwen3_think.yaml}"
export MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}"

case "${MODE}" in
  none)
    export RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_seta_baseline_none_$(date +%Y-%m-%d_%H%M%S)}"
    ;;
  dapo)
    export TRAIN_ITERS_PER_ROLLOUT="${TRAIN_ITERS_PER_ROLLOUT:-2}"
    export UPDATE_POLICY_VERSION_EVERY_TRAIN_ITER="${UPDATE_POLICY_VERSION_EVERY_TRAIN_ITER:-1}"
    ;;
  per)
    export TRAIN_ITERS_PER_ROLLOUT="${TRAIN_ITERS_PER_ROLLOUT:-2}"
    export UPDATE_POLICY_VERSION_EVERY_TRAIN_ITER="${UPDATE_POLICY_VERSION_EVERY_TRAIN_ITER:-1}"
    export OFFPOLICY_PER_PRIORITY_SOURCE="${OFFPOLICY_PER_PRIORITY_SOURCE:-reward_dev}"
    ;;
  topr)
    export TRAIN_ITERS_PER_ROLLOUT="${TRAIN_ITERS_PER_ROLLOUT:-2}"
    export UPDATE_POLICY_VERSION_EVERY_TRAIN_ITER="${UPDATE_POLICY_VERSION_EVERY_TRAIN_ITER:-1}"
    ;;
  spear)
    # SPEAR adds SIL replay loss; keep rollout fan-out lower unless overridden.
    export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
    export N_SAMPLES="${N_SAMPLES:-2}"
    export OFFPOLICY_SPEAR_BUF="${OFFPOLICY_SPEAR_BUF:-2048}"
    export OFFPOLICY_SPEAR_THRESH="${OFFPOLICY_SPEAR_THRESH:-1.0}"
    export OFFPOLICY_SPEAR_COEF="${OFFPOLICY_SPEAR_COEF:-0.001}"
    export OFFPOLICY_SPEAR_STEPS="${OFFPOLICY_SPEAR_STEPS:-200}"
    export OFFPOLICY_SPEAR_DECAY="${OFFPOLICY_SPEAR_DECAY:--1.0}"
    ;;
  all3)
    export TRAIN_ITERS_PER_ROLLOUT="${TRAIN_ITERS_PER_ROLLOUT:-2}"
    export UPDATE_POLICY_VERSION_EVERY_TRAIN_ITER="${UPDATE_POLICY_VERSION_EVERY_TRAIN_ITER:-1}"
    ;;
esac

# Keep rollout pressure below two 16-slot workers by default. Increase only
# after healthz/readyz and slot usage are stable.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
export N_SAMPLES="${N_SAMPLES:-2}"
export DAPO_OVER_SAMPLING_BATCH_SIZE="${DAPO_OVER_SAMPLING_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"

echo "[launch] repo=${REPO_ROOT}"
echo "[launch] mode=${OFFPOLICY_MODE} algo=${ALGO} dataset=${DATASET}"
echo "[launch] workers=${WORKER_URLS}"
echo "[launch] rollout_batch_size=${ROLLOUT_BATCH_SIZE} n_samples=${N_SAMPLES}"
echo "[launch] max_ckpt_keep=${MAX_CKPT_KEEP}"

exec bash terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
