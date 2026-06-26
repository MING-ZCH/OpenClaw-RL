#!/usr/bin/env bash
# Terminal-RL Qwen3-8B mixed-data DAPO baseline without dynamic sampling.
#
# Defaults:
#   * DATASET=mixed with seta:safety:agentharm = 6:2:2
#   * ALGO=dapo, DAPO_DYNAMIC_SAMPLING=0
#   * CUSTOM_CONFIG_PATH=configs/rollout_qwen3_think.yaml
#   * HARNESS_OPTION=camel-agent, MAX_TURN=10
#
# Prerequisites (remote 4-GPU worker):
#   1. Pool server(s) running on reachable host(s), default port 18081:
#        bash terminal-rl/remote/run_pool_server_pu_v2.sh
#   2. WORKER_URLS exported, e.g.
#        export WORKER_URLS="http://<worker-ip>:18081"
#   3. Converted SetA / Agent-SafetyBench / AgentHarm datasets available.
#
# Usage:
#   bash terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_baseline_nodynamic_pu.sh
#   DEBUG_MODE=1 bash terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_baseline_nodynamic_pu.sh
#   NUM_GPUS=4 ACTOR_GPUS=2 ROLLOUT_GPUS=2 bash ..._mixed_dapo_baseline_nodynamic_pu.sh
#
# Structured reward observability:
#   TERMINAL_STRUCTURED_METRICS=1 writes per-rollout dataset reward breakdowns
#   to logs and to ${RUN_DIR}/logs/metrics.jsonl.

set -euo pipefail
set -x

log() { echo "[$(date +'%F %T')] $*"; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] missing cmd: $1"; exit 1; }; }

# ── Conda env (Python 3.12 with transformer_engine + sglang) ─────────
# This is THE only python on this worker that has TE installed:
#   /usr/bin/python3                 → 3.10, no TE
#   /root/miniconda3/bin/python3     → 3.13, no TE  (default `which python3`!)
#   lightrft_py312/bin/python        → 3.12, TE 2.14.1   ← required
# Megatron itself lives in this repo (Megatron-LM/) and is injected via
# PYTHONPATH on the Ray runtime env below.
# v1 (run_swe_rl_4b_remote_1node.sh) worked because the user's shell already
# had this dir on PATH; we make that explicit here so the script is self-contained.
LIGHTRFT_PY312_BIN="${LIGHTRFT_PY312_BIN:-/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin}"
export PATH="${LIGHTRFT_PY312_BIN}:${PATH}"
DRY_RUN="${DRY_RUN:-0}"

# ── Cleanup previous processes ───────────────────────────────────────
if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1: skipping process cleanup and Ray startup"
else
  pkill -9 sglang || true
  sleep 2
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 -f "terminal-rl.router_server" || true
  sleep 2
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ── GPU allocation (auto-split: half actor, half rollout) ────────────
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
else
  DETECTED_GPUS=0
fi
if [[ "${DETECTED_GPUS}" -le 0 ]]; then
  DETECTED_GPUS=4
fi
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS}}"
HALF_GPUS=$(( NUM_GPUS / 2 ))
# Default: each gets half of available GPUs (4-GPU node → 2/2, 8-GPU → 4/4).
# Important: matching node size avoids SIGSEGV in NCCL getenv() observed when
# only a subset of GPUs are used on multi-NUMA 8-GPU nodes.
ACTOR_GPUS="${ACTOR_GPUS:-${HALF_GPUS}}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-${HALF_GPUS}}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${HALF_GPUS}}"
TP_SIZE="${TP_SIZE:-${ACTOR_GPUS}}"

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS(${ACTOR_GPUS}) + ROLLOUT_GPUS(${ROLLOUT_GPUS}) > NUM_GPUS(${NUM_GPUS})"
  exit 1
fi
log "GPU config: total=${NUM_GPUS}, actor=${ACTOR_GPUS}, rollout=${ROLLOUT_GPUS}, TP=${TP_SIZE}, engine_tp=${ROLLOUT_NUM_GPUS_PER_ENGINE}"

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export REPO_ROOT
export SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/slime}"
export MEGATRON_DIR="${MEGATRON_DIR:-${REPO_ROOT}/Megatron-LM}"

CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_qwen3_think.yaml}"

# Hardcoded Qwen3-8B (matches swe-rl v4 pattern)
HF_CKPT="${HF_CKPT:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B/}"
REF_LOAD="${REF_LOAD:-/mnt/shared-storage-user/puyuan/code/slime/Qwen3-8B_torch_dist/}"

EXPORT_ROOT="${EXPORT_ROOT:-/mnt/shared-storage-user/narmodel/agenticrl}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}"
DEBUG_MODE="${DEBUG_MODE:-0}"
# Defaults needed early so the run directory name carries the key experiment
# identity. Dataset construction and full validation still happen below.
ALGO="${ALGO:-dapo}"
case "${ALGO}" in
  grpo|dapo) ;;
  *)
    echo "[ERROR] Unknown ALGO=${ALGO}. Use: grpo|dapo"
    exit 1
    ;;
esac
export ALGO
DATASET="${DATASET:-mixed}"
case "${DATASET}" in
  seta|safety|agentharm|mixed) ;;
  *)
    echo "[ERROR] Unknown DATASET=${DATASET}. Use: seta|safety|agentharm|mixed"
    exit 1
    ;;
esac
HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"
case "${HARNESS_OPTION}" in
  camel-agent|camel_agent)
    HARNESS_OPTION="camel-agent"
    ;;
  a3s-code|a3s_code)
    HARNESS_OPTION="a3s-code"
    ;;
  claude-code|claude_code)
    HARNESS_OPTION="claude-code"
    ;;
  *)
    echo "[ERROR] Unknown HARNESS_OPTION=${HARNESS_OPTION}. Use: camel-agent|a3s-code|claude_code"
    exit 1
    ;;
esac
SETA_SAFETY="${SETA_SAFETY:-none}"
SAFETY_BENCH_REWARD="${SAFETY_BENCH_REWARD:-dense_rule}"
AGENTHARM_REWARD="${AGENTHARM_REWARD:-dense_rule}"
SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0}"
MIX_SETA_RATIO="${MIX_SETA_RATIO:-6}"
MIX_SAFETY_RATIO="${MIX_SAFETY_RATIO:-2}"
MIX_AGENTHARM_RATIO="${MIX_AGENTHARM_RATIO:-2}"
MIX_MODE="${MIX_MODE:-all_visible}"
MAX_TURN="${MAX_TURN:-10}"
DAPO_EPS_CLIP_HIGH="${DAPO_EPS_CLIP_HIGH:-0.28}"
DAPO_CALCULATE_PER_TOKEN_LOSS="${DAPO_CALCULATE_PER_TOKEN_LOSS:-1}"
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"

# Exploration defaults are defined in the main script as well as the wrapper so
# direct invocations remain stable under `set -u`, and Ray runtime_env can always
# receive explicit values. Bug fix: previously the exploration wrapper exported
# EXPLORE_* in the parent shell, but ray job submit workers only received the
# hand-built RUNTIME_ENV_JSON below, so generate.py often saw all exploration
# switches as disabled.
EXPLORATION_PROFILE="${EXPLORATION_PROFILE:-${EXPLORE_PROFILE:-off}}"
EXPLORE_ENTROPY_COEF="${EXPLORE_ENTROPY_COEF:-0.0}"
EXPLORE_THINK_MODE="${EXPLORE_THINK_MODE:-1}"
EXPLORE_TEMP_HIGH="${EXPLORE_TEMP_HIGH:-}"
EXPLORE_INTRINSIC="${EXPLORE_INTRINSIC:-0}"
EXPLORE_INTRINSIC_ENABLED="${EXPLORE_INTRINSIC_ENABLED:-${EXPLORE_INTRINSIC}}"
EXPLORE_INTRINSIC_COEF="${EXPLORE_INTRINSIC_COEF:-0.1}"
EXPLORE_INTRINSIC_SCHEDULE="${EXPLORE_INTRINSIC_SCHEDULE:-constant}"
EXPLORE_INTRINSIC_DECAY_STEPS="${EXPLORE_INTRINSIC_DECAY_STEPS:-0}"
EXPLORE_INTRINSIC_REDUCER="${EXPLORE_INTRINSIC_REDUCER:-sum}"
EXPLORE_INTRINSIC_GRANULARITY="${EXPLORE_INTRINSIC_GRANULARITY:-raw}"
EXPLORE_INTRINSIC_SCOPE="${EXPLORE_INTRINSIC_SCOPE:-process}"
EXPLORE_SCORE_BONUS_COMPONENTS="${EXPLORE_SCORE_BONUS_COMPONENTS:-legacy}"
EXPLORE_SAFETY_FILTER="${EXPLORE_SAFETY_FILTER:-0}"
EXPLORE_SAFETY_FILTER_ENABLED="${EXPLORE_SAFETY_FILTER_ENABLED:-${EXPLORE_SAFETY_FILTER}}"
EXPLORE_SAFETY_FILTER_COEF="${EXPLORE_SAFETY_FILTER_COEF:--0.5}"
EXPLORE_LPRND="${EXPLORE_LPRND:-0}"
EXPLORE_LPRND_ENABLED="${EXPLORE_LPRND_ENABLED:-${EXPLORE_LPRND}}"
EXPLORE_LPRND_COEF="${EXPLORE_LPRND_COEF:-0.05}"
EXPLORE_LPRND_SCHEDULE="${EXPLORE_LPRND_SCHEDULE:-constant}"
EXPLORE_LPRND_DECAY_STEPS="${EXPLORE_LPRND_DECAY_STEPS:-0}"
EXPLORE_LPRND_CLIP="${EXPLORE_LPRND_CLIP:-3.0}"
EXPLORE_LPRND_WARMUP="${EXPLORE_LPRND_WARMUP:-32}"
EXPLORE_AGENT57_LITE="${EXPLORE_AGENT57_LITE:-0}"
EXPLORE_AGENT57_LITE_ENABLED="${EXPLORE_AGENT57_LITE_ENABLED:-${EXPLORE_AGENT57_LITE}}"
EXPLORE_AGENT57_K="${EXPLORE_AGENT57_K:-8}"
EXPLORE_AGENT57_ARM_BETAS="${EXPLORE_AGENT57_ARM_BETAS:-0,0.003,0.006,0.01,0.015,0.02,0.03,0.04}"
EXPLORE_AGENT57_COMBINE_MODE="${EXPLORE_AGENT57_COMBINE_MODE:-add}"
EXPLORE_AGENT57_NGU_MOD_CLIP="${EXPLORE_AGENT57_NGU_MOD_CLIP:-5.0}"
EXPLORE_AGENT57_NGU_EPISODIC_SOURCE="${EXPLORE_AGENT57_NGU_EPISODIC_SOURCE:-signature_intrinsic}"
EXPLORE_AGENT57_NGU_EPISODIC_REDUCER="${EXPLORE_AGENT57_NGU_EPISODIC_REDUCER:-${EXPLORE_INTRINSIC_REDUCER}}"
EXPLORE_AGENT57_NGU_LIFE_MOD_MODE="${EXPLORE_AGENT57_NGU_LIFE_MOD_MODE:-linear}"
EXPLORE_AGENT57_NGU_LIFE_MOD_STD_CLIP="${EXPLORE_AGENT57_NGU_LIFE_MOD_STD_CLIP:-5.0}"
EXPLORE_AGENT57_MAX_BONUS="${EXPLORE_AGENT57_MAX_BONUS:-0}"
EXPLORE_AGENT57_ARM_TEMPERATURES="${EXPLORE_AGENT57_ARM_TEMPERATURES:-}"
EXPLORE_AGENT57_ARM_TEMPERATURE_WARMUP_ROLLOUTS="${EXPLORE_AGENT57_ARM_TEMPERATURE_WARMUP_ROLLOUTS:-0}"
EXPLORE_AGENT57_ARM_TOP_PS="${EXPLORE_AGENT57_ARM_TOP_PS:-}"
EXPLORE_AGENT57_ARM_TOP_KS="${EXPLORE_AGENT57_ARM_TOP_KS:-}"
EXPLORE_AGENT57_CONTROLLER="${EXPLORE_AGENT57_CONTROLLER:-fixed}"
EXPLORE_AGENT57_UCB_C="${EXPLORE_AGENT57_UCB_C:-0.5}"
EXPLORE_AGENT57_UCB_WINDOW="${EXPLORE_AGENT57_UCB_WINDOW:-256}"
EXPLORE_AGENT57_UCB_EPSILON="${EXPLORE_AGENT57_UCB_EPSILON:-0}"
EXPLORE_AGENT57_UCB_MIN_PER_ARM="${EXPLORE_AGENT57_UCB_MIN_PER_ARM:-0}"
EXPLORE_AGENT57_UCB_VALUE="${EXPLORE_AGENT57_UCB_VALUE:-legacy}"
EXPLORE_AGENT57_UCB_DATASET_AWARE="${EXPLORE_AGENT57_UCB_DATASET_AWARE:-0}"
EXPLORE_AGENT57_UCB_RANDOM_SEED="${EXPLORE_AGENT57_UCB_RANDOM_SEED:-}"
EXPLORE_AGENT57_UCB_SEED_SALT="${EXPLORE_AGENT57_UCB_SEED_SALT:-}"
EXPLORE_AGENT57_KEEP_BASELINE="${EXPLORE_AGENT57_KEEP_BASELINE:-1}"
EPISODIC_MEMORY_BACKEND="${EPISODIC_MEMORY_BACKEND:-simhash_knn}"
EXPLORE_AGENT57_EPISODIC_BACKEND="${EXPLORE_AGENT57_EPISODIC_BACKEND:-${EPISODIC_MEMORY_BACKEND}}"
EXPLORE_AGENT57_EPISODIC_CAPACITY="${EXPLORE_AGENT57_EPISODIC_CAPACITY:-4096}"
EXPLORE_AGENT57_EPISODIC_COUNT_DECAY="${EXPLORE_AGENT57_EPISODIC_COUNT_DECAY:-1.0}"
EXPLORE_AGENT57_EPISODIC_CLEAR_ON_RESET="${EXPLORE_AGENT57_EPISODIC_CLEAR_ON_RESET:-1}"
EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS="${EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS:-64}"
EXPLORE_AGENT57_EPISODIC_BUCKET_CAPACITY="${EXPLORE_AGENT57_EPISODIC_BUCKET_CAPACITY:-256}"
EXPLORE_AGENT57_EPISODIC_K="${EXPLORE_AGENT57_EPISODIC_K:-5}"
EXPLORE_AGENT57_EPISODIC_DISTANCE="${EXPLORE_AGENT57_EPISODIC_DISTANCE:-cosine}"
EXPLORE_AGENT57_EPISODIC_VECTOR_DIM="${EXPLORE_AGENT57_EPISODIC_VECTOR_DIM:-256}"
EXPLORE_AGENT57_EPISODIC_RANDOM_SEED="${EXPLORE_AGENT57_EPISODIC_RANDOM_SEED:-}"
EXPLORE_AGENT57_EPISODIC_OBS_MODE="${EXPLORE_AGENT57_EPISODIC_OBS_MODE:-fingerprint}"
EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN="${EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN:-1}"
EXPLORE_AGENT57_EPISODIC_MULTI_PROBE_RADIUS="${EXPLORE_AGENT57_EPISODIC_MULTI_PROBE_RADIUS:-1}"
EXPLORE_AGENT57_EPISODIC_NOVELTY_FLOOR="${EXPLORE_AGENT57_EPISODIC_NOVELTY_FLOOR:-0.05}"
EXPLORE_AGENT57_LIFELONG="${EXPLORE_AGENT57_LIFELONG:-0}"
EXPLORE_AGENT57_LIFELONG_ENABLED="${EXPLORE_AGENT57_LIFELONG_ENABLED:-${EXPLORE_AGENT57_LIFELONG}}"
EXPLORE_AGENT57_LIFELONG_COEF="${EXPLORE_AGENT57_LIFELONG_COEF:-0.01}"
EXPLORE_AGENT57_LIFELONG_CLIP="${EXPLORE_AGENT57_LIFELONG_CLIP:-2.0}"
EXPLORE_AGENT57_LIFELONG_WARMUP="${EXPLORE_AGENT57_LIFELONG_WARMUP:-64}"
EXPLORE_AGENT57_LIFELONG_COUNT_DECAY="${EXPLORE_AGENT57_LIFELONG_COUNT_DECAY:-1.0}"
EXPLORE_AGENT57_LIFELONG_CAPACITY="${EXPLORE_AGENT57_LIFELONG_CAPACITY:-0}"
EXPLORE_AGENT57_LIFELONG_BACKEND="${EXPLORE_AGENT57_LIFELONG_BACKEND:-local}"
EXPLORE_AGENT57_LIFELONG_KEY_VERSION="${EXPLORE_AGENT57_LIFELONG_KEY_VERSION:-v1}"
EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET="${EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET:-1}"
EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK="${EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK:-1}"
EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN="${EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN:-0}"
EXPLORE_AGENT57_LIFELONG_OBS_MODE="${EXPLORE_AGENT57_LIFELONG_OBS_MODE:-fingerprint}"
EXPLORE_AGENT57_LIFELONG_HIERARCHICAL="${EXPLORE_AGENT57_LIFELONG_HIERARCHICAL:-1}"
EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT="${EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT:-0.5}"
EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT="${EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT:-0.35}"
EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT="${EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT:-0.15}"
EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS="${EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS:-30000}"
EXPLORE_AGENT57_SQLITE_WAL="${EXPLORE_AGENT57_SQLITE_WAL:-0}"
EXPLORE_AGENT57_TRUST_GATE="${EXPLORE_AGENT57_TRUST_GATE:-hard}"
EXPLORE_AGENT57_TRUST_COMPLETED="${EXPLORE_AGENT57_TRUST_COMPLETED:-1.0}"
EXPLORE_AGENT57_TRUST_TRUNCATED="${EXPLORE_AGENT57_TRUST_TRUNCATED:-0.3}"
EXPLORE_AGENT57_TRUST_FAILED="${EXPLORE_AGENT57_TRUST_FAILED:-0.1}"
EXPLORE_AGENT57_TRUST_PARSE_ERROR="${EXPLORE_AGENT57_TRUST_PARSE_ERROR:-0.1}"
EXPLORE_AGENT57_TRUST_WARMUP="${EXPLORE_AGENT57_TRUST_WARMUP:-0.3}"
EXPLORE_AGENT57_STATE_PATH="${EXPLORE_AGENT57_STATE_PATH:-}"
EXPLORE_AGENT57_SUCCESS_THRESHOLD="${EXPLORE_AGENT57_SUCCESS_THRESHOLD:-0.0}"
EXPLORE_ADVANTAGE_BONUS="${EXPLORE_ADVANTAGE_BONUS:-0}"
EXPLORE_ADVANTAGE_BONUS_ENABLED="${EXPLORE_ADVANTAGE_BONUS_ENABLED:-${EXPLORE_ADVANTAGE_BONUS}}"
EXPLORE_ADVANTAGE_BONUS_MODE="${EXPLORE_ADVANTAGE_BONUS_MODE:-component}"
EXPLORE_ADVANTAGE_BONUS_COMPONENTS="${EXPLORE_ADVANTAGE_BONUS_COMPONENTS:-explore_intrinsic_scaled}"
EXPLORE_ADVANTAGE_BONUS_COEF="${EXPLORE_ADVANTAGE_BONUS_COEF:-1.0}"
EXPLORE_ADVANTAGE_BONUS_CLIP="${EXPLORE_ADVANTAGE_BONUS_CLIP:-0.5}"
EXPLORE_ADVANTAGE_INTRINSIC_KEY="${EXPLORE_ADVANTAGE_INTRINSIC_KEY:-explore_agent57_intrinsic_signal}"
EXPLORE_ADVANTAGE_LAMBDA="${EXPLORE_ADVANTAGE_LAMBDA:-0.05}"
EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE="${EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE:-constant}"
EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS="${EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS:-0}"
EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE="${EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE:-normalized_beta}"
EXPLORE_ADVANTAGE_TRUST_KEY="${EXPLORE_ADVANTAGE_TRUST_KEY:-explore_agent57_trust}"
EXPLORE_ADVANTAGE_TRUNCATED_INTRINSIC_SCALE="${EXPLORE_ADVANTAGE_TRUNCATED_INTRINSIC_SCALE:-1.0}"
EXPLORE_ADVANTAGE_FAILED_INTRINSIC_SCALE="${EXPLORE_ADVANTAGE_FAILED_INTRINSIC_SCALE:-1.0}"
EXPLORE_TRUNCATION_PENALTY="${EXPLORE_TRUNCATION_PENALTY:-0}"
EXPLORE_ADVANTAGE_TRUNCATION_PENALTY="${EXPLORE_ADVANTAGE_TRUNCATION_PENALTY:-${EXPLORE_TRUNCATION_PENALTY}}"
EXPLORE_CDE_ACTOR="${EXPLORE_CDE_ACTOR:-0}"
EXPLORE_CDE_ACTOR_ENABLED="${EXPLORE_CDE_ACTOR_ENABLED:-${EXPLORE_CDE_ACTOR}}"
EXPLORE_CDE_ACTOR_OMEGA="${EXPLORE_CDE_ACTOR_OMEGA:-0.05}"
EXPLORE_CDE_ACTOR_KAPPA="${EXPLORE_CDE_ACTOR_KAPPA:-2.0}"
EXPLORE_CDE_ACTOR_ALPHA="${EXPLORE_CDE_ACTOR_ALPHA:-0.1}"
EXPLORE_CDE_ACTOR_DECAY_STEPS="${EXPLORE_CDE_ACTOR_DECAY_STEPS:-0}"
EXPLORE_CDE_ACTOR_REWARD_GATE="${EXPLORE_CDE_ACTOR_REWARD_GATE:-nonzero}"
EXPLORE_RETRY_ATTEMPTS="${EXPLORE_RETRY_ATTEMPTS:-1}"
EXPLORE_RETRY_TRAJ_GAMMA="${EXPLORE_RETRY_TRAJ_GAMMA:-1.0}"
SLIME_SKIP_ZERO_TRAINABLE_ROLLOUT="${SLIME_SKIP_ZERO_TRAINABLE_ROLLOUT:-0}"
SLIME_SKIP_ZERO_TRAINABLE_TRAIN="${SLIME_SKIP_ZERO_TRAINABLE_TRAIN:-1}"

short_mode() {
  case "$1" in
    clawsentry) echo "cs" ;;
    dense_rule) echo "dense" ;;
    *) echo "$1" ;;
  esac
}

sanitize_run_part() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '-' | sed 's/-\\{1,\\}/-/g; s/^-//; s/-$//'
}

build_dataset_tag() {
  case "${DATASET}" in
    seta)
      echo "seta-$(short_mode "${SETA_SAFETY}")-c${SAFETY_REWARD_COEF}"
      ;;
    safety)
      echo "asb-$(short_mode "${SAFETY_BENCH_REWARD}")"
      ;;
    agentharm)
      echo "agentharm-$(short_mode "${AGENTHARM_REWARD}")"
      ;;
    mixed)
      local seta_ratio safety_ratio agentharm_ratio
      if [[ -n "${MIX_AGENTHARM_RATIO:-}" ]]; then
        seta_ratio="${MIX_SETA_RATIO:-0}"
        safety_ratio="${MIX_SAFETY_RATIO:-0}"
        agentharm_ratio="${MIX_AGENTHARM_RATIO:-0}"
      else
        seta_ratio="${MIX_SETA_RATIO:-1}"
        safety_ratio="${MIX_SAFETY_RATIO:-1}"
        agentharm_ratio="0"
      fi
      echo "mixed-s${seta_ratio}_asb${safety_ratio}_ah${agentharm_ratio}-rw$(short_mode "${SETA_SAFETY}")_$(short_mode "${SAFETY_BENCH_REWARD}")_$(short_mode "${AGENTHARM_REWARD}")-c${SAFETY_REWARD_COEF}"
      ;;
    *)
      echo "${DATASET}"
      ;;
  esac
}

build_algo_tag() {
  case "${ALGO}" in
    dapo)
      echo "dapo-ch${DAPO_EPS_CLIP_HIGH}-tok${DAPO_CALCULATE_PER_TOKEN_LOSS}-dyn${DAPO_DYNAMIC_SAMPLING}"
      ;;
    *)
      echo "${ALGO}"
      ;;
  esac
}

RUN_DATASET_TAG="$(sanitize_run_part "$(build_dataset_tag)")"
RUN_ALGO_TAG="$(sanitize_run_part "$(build_algo_tag)")"
RUN_HARNESS_TAG="$(sanitize_run_part "${HARNESS_OPTION}")"
# Checkpoint saving keeps only the latest N checkpoints by default.
# When enabled, only the latest N checkpoints are kept; older ones are auto-deleted.
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-8}"
if [[ "${DEBUG_MODE}" == "1" ]]; then
  RUN_NAME="${RUN_NAME:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_debug_mixed_dapo_nodynamic_think_s${MIX_SETA_RATIO}_asb${MIX_SAFETY_RATIO}_ah${MIX_AGENTHARM_RATIO}_harness-${RUN_HARNESS_TAG}_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
  # Debug mode: never save checkpoints regardless of MAX_CKPT_KEEP
  MAX_CKPT_KEEP=0
else
  RUN_NAME="${RUN_NAME:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_mixed_dapo_nodynamic_think_s${MIX_SETA_RATIO}_asb${MIX_SAFETY_RATIO}_ah${MIX_AGENTHARM_RATIO}_harness-${RUN_HARNESS_TAG}_mt${MAX_TURN}_${RUN_TIMESTAMP}}"
fi

# ── Unified run directory (see STORAGE.md) ───────────────────────────────
# All outputs for this run go under runs/{RUN_ID}/ with structured subdirs.
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
CKPT_ROOT="${CKPT_ROOT:-${EXPORT_ROOT}/ckpt}"
RUN_ID="${RUN_ID:-${RUN_NAME}}"
RUN_DIR="${RUNS_ROOT}/${RUN_ID}"

# Create directory structure via run_paths.py
MAX_CKPT_KEEP="${MAX_CKPT_KEEP}" python3 "${SCRIPT_DIR}/run_paths.py" init \
  --runs-root "${RUNS_ROOT}" \
  --ckpt-root "${CKPT_ROOT}" \
  --run-id "${RUN_ID}" > /dev/null 2>&1

# Derive all paths from RUN_DIR
RUN_LOG_DIR="${RUN_DIR}/logs"
if [[ "${TERMINAL_SAVE_TRAJ_DIR+x}" ]]; then
  TERMINAL_SAVE_TRAJ_DIR="${TERMINAL_SAVE_TRAJ_DIR}"
else
  TERMINAL_SAVE_TRAJ_DIR="${RUN_DIR}/trajectories"
fi
WANDB_DIR="${RUN_DIR}/metrics/wandb"
TERMINAL_STRUCTURED_METRICS="${TERMINAL_STRUCTURED_METRICS:-1}"
TERMINAL_METRICS_JSONL="${TERMINAL_METRICS_JSONL:-${RUN_LOG_DIR}/metrics.jsonl}"
TERMINAL_WANDB_METRIC_PROFILE="${TERMINAL_WANDB_METRIC_PROFILE:-full}"
export TERMINAL_STRUCTURED_METRICS TERMINAL_METRICS_JSONL TERMINAL_WANDB_METRIC_PROFILE
TRAIN_PYTHON="${TRAIN_PYTHON:-python3}"
A3S_CODE_REPO_ROOT="${A3S_CODE_REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/Code}"
A3S_CODE_CONFIG_PATH="${A3S_CODE_CONFIG_PATH:-${REPO_ROOT}/a3s-code-adapter/generated_configs/a3s-code-shared.hcl}"
A3S_CODE_PY_TAG="$("${TRAIN_PYTHON}" - <<'PY'
import platform
import sys

machine = platform.machine().lower().replace("amd64", "x86_64")
print(f"cp{sys.version_info.major}{sys.version_info.minor}-{machine}")
PY
)"
A3S_CODE_CACHE_DIR="${A3S_CODE_CACHE_DIR:-/mnt/shared-storage-user/puyuan/.cache/a3s-code-${A3S_CODE_PY_TAG}}"
A3S_CODE_WORKSPACE_ROOT="${A3S_CODE_WORKSPACE_ROOT:-${RUN_DIR}/a3s_code_workspaces}"
A3S_CODE_EXTRA_SITE_PACKAGES="${A3S_CODE_EXTRA_SITE_PACKAGES:-}"
A3S_CODE_TURN_TIMEOUT_SEC="${A3S_CODE_TURN_TIMEOUT_SEC:-900}"
A3S_CODE_TOOL_TIMEOUT_MS="${A3S_CODE_TOOL_TIMEOUT_MS:-300000}"
A3S_CODE_MAX_TOOL_ROUNDS="${A3S_CODE_MAX_TOOL_ROUNDS:-10}"
A3S_CODE_MAX_PARSE_RETRIES="${A3S_CODE_MAX_PARSE_RETRIES:-4}"
A3S_CODE_OUTPUT_TOKENS="${A3S_CODE_OUTPUT_TOKENS:-8192}"
A3S_CODE_PLANNING_MODE="${A3S_CODE_PLANNING_MODE:-disabled}"
A3S_CODE_THINKING_BUDGET="${A3S_CODE_THINKING_BUDGET:-}"
A3S_CODE_EXTERNAL_TOOL_ERRORS_AS_RESULTS="${A3S_CODE_EXTERNAL_TOOL_ERRORS_AS_RESULTS:-1}"
A3S_CODE_LOCAL_WORKSPACE_GUARD="${A3S_CODE_LOCAL_WORKSPACE_GUARD:-1}"
A3S_CODE_PIP_PACKAGE="${A3S_CODE_PIP_PACKAGE:-a3s-code==3.3.0}"
CLAUDE_CODE_XTRACE_WAS_ON=0
if [[ "${HARNESS_OPTION}" == "claude-code" && "$-" == *x* ]]; then
  CLAUDE_CODE_XTRACE_WAS_ON=1
  set +x
fi
CLAUDE_CODE_CLI="${CLAUDE_CODE_CLI:-claude}"
CLAUDE_CODE_LLM_BACKEND="${CLAUDE_CODE_LLM_BACKEND:-sglang}"
case "${CLAUDE_CODE_LLM_BACKEND}" in
  sglang|qwen|qwen-sglang|local|local-sglang)
    CLAUDE_CODE_LLM_BACKEND="sglang"
    ;;
  anthropic|claude|claude-api|external)
    CLAUDE_CODE_LLM_BACKEND="anthropic"
    ;;
  *)
    echo "[ERROR] Unknown CLAUDE_CODE_LLM_BACKEND=${CLAUDE_CODE_LLM_BACKEND}. Use: sglang|anthropic" >&2
    exit 1
    ;;
esac
CLAUDE_CODE_MODEL="${CLAUDE_CODE_MODEL:-}"
CLAUDE_CODE_QWEN_GATEWAY_MODEL="${CLAUDE_CODE_QWEN_GATEWAY_MODEL:-qwen-8b-sglang}"
CLAUDE_CODE_WORKSPACE_ROOT="${CLAUDE_CODE_WORKSPACE_ROOT:-${RUN_DIR}/claude_code_workspaces}"
CLAUDE_CODE_TURN_TIMEOUT_SEC="${CLAUDE_CODE_TURN_TIMEOUT_SEC:-900}"
CLAUDE_CODE_TOOL_TIMEOUT_MS="${CLAUDE_CODE_TOOL_TIMEOUT_MS:-300000}"
CLAUDE_CODE_MAX_TOOL_ROUNDS="${CLAUDE_CODE_MAX_TOOL_ROUNDS:-10}"
CLAUDE_CODE_OUTPUT_FORMAT="${CLAUDE_CODE_OUTPUT_FORMAT:-json}"
CLAUDE_CODE_PERMISSION_MODE="${CLAUDE_CODE_PERMISSION_MODE:-bypassPermissions}"
CLAUDE_CODE_ALLOWED_TOOLS="${CLAUDE_CODE_ALLOWED_TOOLS:-mcp__terminal_rl__shell_exec,mcp__terminal_rl__shell_view,mcp__terminal_rl__shell_write_to_process,mcp__terminal_rl__shell_write_content_to_file}"
CLAUDE_CODE_DISALLOWED_TOOLS="${CLAUDE_CODE_DISALLOWED_TOOLS:-}"
CLAUDE_CODE_EXTRA_ARGS="${CLAUDE_CODE_EXTRA_ARGS:-}"
CLAUDE_CODE_SYSTEM_PROMPT="${CLAUDE_CODE_SYSTEM_PROMPT:-}"
CLAUDE_CODE_MCP_PYTHON="${CLAUDE_CODE_MCP_PYTHON:-${TRAIN_PYTHON}}"
CLAUDE_CODE_HTTP_MAX_RETRIES="${CLAUDE_CODE_HTTP_MAX_RETRIES:-3}"
CLAUDE_CODE_HTTP_RETRY_DELAY="${CLAUDE_CODE_HTTP_RETRY_DELAY:-1.0}"
if [[ -z "${CLAUDE_CODE_MARK_NON_TRAINABLE+x}" ]]; then
  if [[ "${CLAUDE_CODE_LLM_BACKEND}" == "sglang" ]]; then
    CLAUDE_CODE_MARK_NON_TRAINABLE="0"
  else
    CLAUDE_CODE_MARK_NON_TRAINABLE="1"
  fi
fi
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-}"
ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-}"
ANTHROPIC_API_URL="${ANTHROPIC_API_URL:-}"
if [[ "${CLAUDE_CODE_XTRACE_WAS_ON}" == "1" ]]; then
  set -x
fi
ENV_HTTP_MAX_RETRIES="${ENV_HTTP_MAX_RETRIES:-10}"
ENV_ALLOCATE_MAX_RETRIES="${ENV_ALLOCATE_MAX_RETRIES:-20}"
ENV_ALLOCATE_RETRY_BASE_DELAY="${ENV_ALLOCATE_RETRY_BASE_DELAY:-2.0}"
ENV_ALLOCATE_RETRY_MAX_DELAY="${ENV_ALLOCATE_RETRY_MAX_DELAY:-30.0}"
ENV_ALLOCATE_RETRY_BACKOFF="${ENV_ALLOCATE_RETRY_BACKOFF:-2.0}"
ENV_ALLOCATE_RETRY_JITTER="${ENV_ALLOCATE_RETRY_JITTER:-0.25}"
HTTP_RETRY_LOG_EVERY_N="${HTTP_RETRY_LOG_EVERY_N:-25}"
HTTP_RETRY_LOG_RESPONSE_CHARS="${HTTP_RETRY_LOG_RESPONSE_CHARS:-512}"
TERMINAL_RL_GENERATE_FAILURE_TRACEBACK="${TERMINAL_RL_GENERATE_FAILURE_TRACEBACK:-0}"
ENV_EVALUATE_MAX_RETRIES="${ENV_EVALUATE_MAX_RETRIES:-1}"
ENV_CLOSE_MAX_RETRIES="${ENV_CLOSE_MAX_RETRIES:-3}"
ENV_EXEC_TOOL_MAX_RETRIES="${ENV_EXEC_TOOL_MAX_RETRIES:-3}"
ENV_ALLOCATE_HTTP_TIMEOUT="${ENV_ALLOCATE_HTTP_TIMEOUT:-300}"
ENV_RESET_HTTP_TIMEOUT="${ENV_RESET_HTTP_TIMEOUT:-2100}"
ENV_CLOSE_HTTP_TIMEOUT="${ENV_CLOSE_HTTP_TIMEOUT:-90}"
ENV_REMOTE_MAX_ACTIVE_TASKS="${ENV_REMOTE_MAX_ACTIVE_TASKS:-12}"
ENV_REMOTE_MAX_ACTIVE_RUNS="${ENV_REMOTE_MAX_ACTIVE_RUNS:-0}"
ENV_REMOTE_MAX_RUNS_PER_TASK="${ENV_REMOTE_MAX_RUNS_PER_TASK:-8}"
ENV_REMOTE_ADMISSION_TIMEOUT="${ENV_REMOTE_ADMISSION_TIMEOUT:-900}"
ENV_REMOTE_ADMISSION_LOG_INTERVAL="${ENV_REMOTE_ADMISSION_LOG_INTERVAL:-30}"
ENV_REMOTE_MAX_CONCURRENT_CLOSES="${ENV_REMOTE_MAX_CONCURRENT_CLOSES:-8}"

a3s_code_import_check() {
  mkdir -p "${RUN_LOG_DIR}"
  A3S_CODE_CACHE_DIR="${A3S_CODE_CACHE_DIR}" \
    "${TRAIN_PYTHON}" -c "import a3s_code" > "${RUN_LOG_DIR}/a3s_code_import_check.log" 2>&1
}

a3s_code_print_cache_hint() {
  echo "[ERROR] HARNESS_OPTION=a3s-code but a3s_code import failed."
  echo "[ERROR] a3s-code==3.3.0 downloads its native wheel on first import."
  echo "[ERROR] Offline GPU workers need a prewarmed shared native cache."
  echo "[ERROR] On an online CPU/shared-storage node, run:"
  echo "  A3S_CODE_CACHE_DIR=${A3S_CODE_CACHE_DIR} ${TRAIN_PYTHON} -m pip install ${A3S_CODE_PIP_PACKAGE}"
  echo "  A3S_CODE_CACHE_DIR=${A3S_CODE_CACHE_DIR} ${TRAIN_PYTHON} -c 'import a3s_code'"
  echo "[ERROR] Then retry on the GPU worker with the same A3S_CODE_CACHE_DIR."
  echo "[ERROR] Expected native cache under: ${A3S_CODE_CACHE_DIR}/3.3.0/_native.*"
  echo "[ERROR] Import log: ${RUN_LOG_DIR}/a3s_code_import_check.log"
}

claude_code_preflight() {
  mkdir -p "${RUN_LOG_DIR}"
  if ! command -v "${CLAUDE_CODE_CLI}" >/dev/null 2>&1; then
    echo "[ERROR] HARNESS_OPTION=claude-code but CLAUDE_CODE_CLI=${CLAUDE_CODE_CLI} is not on PATH."
    echo "[ERROR] Install Claude Code CLI or set CLAUDE_CODE_CLI=/absolute/path/to/claude."
    return 1
  fi
  if ! "${CLAUDE_CODE_MCP_PYTHON}" -c "import mcp.server.fastmcp" > "${RUN_LOG_DIR}/claude_code_mcp_import_check.log" 2>&1; then
    echo "[ERROR] HARNESS_OPTION=claude-code but Python cannot import mcp.server.fastmcp."
    echo "[ERROR] Set CLAUDE_CODE_MCP_PYTHON to a Python with the mcp package installed."
    echo "[ERROR] Import log: ${RUN_LOG_DIR}/claude_code_mcp_import_check.log"
    return 1
  fi
  if [[ "${CLAUDE_CODE_LLM_BACKEND}" != "sglang" && -z "${ANTHROPIC_API_KEY}${ANTHROPIC_AUTH_TOKEN}" ]]; then
    echo "[WARN] claude-code auth env vars are empty. This is OK only if the Claude Code CLI is already authenticated via its own config."
  fi
}

# ── Rollout knobs (env-configurable, baked into per-run yaml below) ──────
# MAX_TURN: max model turns per rollout (terminal_max_iterations in generate.py).
#   Empirical guidance based on 05-21 trajectory analysis (1743 trajectories):
#     - 30.0% trajectories hit max_iteration=15 (TRUNCATED) → most tasks need fewer turns
#     - Pass cases averaged 5-9 turns; tasks taking 10+ turns rarely passed
#     - Lowering to 10 trims tail-latency rollouts ≈ 33%, saving ~3 hours / 78 rollouts at 14h
#     - For exploratory runs needing more turns, override with MAX_TURN=15 or higher.
MAX_TURN="${MAX_TURN:-10}"
# TRAJECTORY_SAVE_INTERVAL controls full trajectory artifact storage globally.
# Per-dataset knobs override the global value and keep eval/training metrics
# unchanged; they only throttle full traj.json/meta.json artifact writes.
#   unset / config value 1: save every rollout step (backward compatible)
#   N>1: save only when train_step % N == 0
#   0: disable trajectory artifact writes even when TERMINAL_SAVE_TRAJ_DIR is set
TRAJECTORY_SAVE_INTERVAL="${TRAJECTORY_SAVE_INTERVAL:-}"
if [[ -n "${TRAJECTORY_SAVE_INTERVAL}" ]]; then
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_SETA=""
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH=""
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENTHARM=""
elif [[ "${DEBUG_MODE}" == "1" ]]; then
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_SETA="1"
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH="1"
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENTHARM="1"
else
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_SETA="5"
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH="5"
  DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENTHARM="10"
fi
TRAJECTORY_SAVE_INTERVAL_SETA="${TRAJECTORY_SAVE_INTERVAL_SETA:-${SAVE_INTERVAL_SETA:-${DEFAULT_TRAJECTORY_SAVE_INTERVAL_SETA}}}"
TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH="${TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH:-${TRAJECTORY_SAVE_INTERVAL_ASB:-${SAVE_INTERVAL_AGENT_SAFETYBENCH:-${SAVE_INTERVAL_ASB:-${DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH}}}}}"
TRAJECTORY_SAVE_INTERVAL_AGENTHARM="${TRAJECTORY_SAVE_INTERVAL_AGENTHARM:-${SAVE_INTERVAL_AGENTHARM:-${DEFAULT_TRAJECTORY_SAVE_INTERVAL_AGENTHARM}}}"
TRAJECTORY_SAVE_POLICY="${TRAJECTORY_SAVE_POLICY:-step_interval}"
TRAJECTORY_TASK_SAVE_INTERVAL="${TRAJECTORY_TASK_SAVE_INTERVAL:-}"
TRAJECTORY_TASK_MAX_PER_STEP="${TRAJECTORY_TASK_MAX_PER_STEP:-2}"
TRAJECTORY_TASK_MAX_PER_TASK="${TRAJECTORY_TASK_MAX_PER_TASK:-24}"
TRAJECTORY_MAX_TOTAL="${TRAJECTORY_MAX_TOTAL:-5000}"
TRAJECTORY_SAVE_REWARD_STRATA="${TRAJECTORY_SAVE_REWARD_STRATA:-best,worst}"
TRAJECTORY_SAVE_LOG_DECISIONS="${TRAJECTORY_SAVE_LOG_DECISIONS:-0}"

# Generate a per-run yaml that overlays MAX_TURN onto the base CUSTOM_CONFIG_PATH.
# This is cleaner than mutating the base yaml — different concurrent runs can pick
# different MAX_TURN without stepping on each other.
BASE_CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH}"
RUN_CUSTOM_CONFIG_PATH="${RUN_DIR}/config/rollout_config.yaml"
mkdir -p "$(dirname "${RUN_CUSTOM_CONFIG_PATH}")"
if [[ -f "${BASE_CUSTOM_CONFIG_PATH}" ]]; then
  python3 - "$BASE_CUSTOM_CONFIG_PATH" "$RUN_CUSTOM_CONFIG_PATH" "$MAX_TURN" "$TRAJECTORY_SAVE_INTERVAL" "$HARNESS_OPTION" "$TRAJECTORY_SAVE_INTERVAL_SETA" "$TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH" "$TRAJECTORY_SAVE_INTERVAL_AGENTHARM" <<'PY'
import sys, yaml
(
    src,
    dst,
    max_turn,
    traj_interval,
    harness_option,
    traj_seta,
    traj_asb,
    traj_agentharm,
) = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    sys.argv[4].strip(),
    sys.argv[5].strip(),
    sys.argv[6].strip(),
    sys.argv[7].strip(),
    sys.argv[8].strip(),
)
with open(src) as f:
    cfg = yaml.safe_load(f) or {}
cfg["max_iteration"] = max_turn
cfg["harness_option"] = harness_option
if traj_interval:
    cfg["trajectory_save_interval"] = int(traj_interval)
if traj_seta:
    cfg["trajectory_save_interval_seta"] = int(traj_seta)
if traj_asb:
    cfg["trajectory_save_interval_agent_safetybench"] = int(traj_asb)
if traj_agentharm:
    cfg["trajectory_save_interval_agentharm"] = int(traj_agentharm)
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=True)
PY
  CUSTOM_CONFIG_PATH="${RUN_CUSTOM_CONFIG_PATH}"
  if [[ -n "${TRAJECTORY_SAVE_INTERVAL}" ]]; then
    echo "[config] rollout yaml -> ${RUN_CUSTOM_CONFIG_PATH} (max_iteration=${MAX_TURN}, harness_option=${HARNESS_OPTION}, trajectory_save_interval=${TRAJECTORY_SAVE_INTERVAL}, per_dataset=seta:${TRAJECTORY_SAVE_INTERVAL_SETA}/asb:${TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH}/agentharm:${TRAJECTORY_SAVE_INTERVAL_AGENTHARM}, traj_policy=${TRAJECTORY_SAVE_POLICY}, task_interval=${TRAJECTORY_TASK_SAVE_INTERVAL:-<per-dataset>}, per_step=${TRAJECTORY_TASK_MAX_PER_STEP}, per_task=${TRAJECTORY_TASK_MAX_PER_TASK}, max_total=${TRAJECTORY_MAX_TOTAL})"
  else
    echo "[config] rollout yaml -> ${RUN_CUSTOM_CONFIG_PATH} (max_iteration=${MAX_TURN}, harness_option=${HARNESS_OPTION}, per_dataset=seta:${TRAJECTORY_SAVE_INTERVAL_SETA}/asb:${TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH}/agentharm:${TRAJECTORY_SAVE_INTERVAL_AGENTHARM}, traj_policy=${TRAJECTORY_SAVE_POLICY}, task_interval=${TRAJECTORY_TASK_SAVE_INTERVAL:-<per-dataset>}, per_step=${TRAJECTORY_TASK_MAX_PER_STEP}, per_task=${TRAJECTORY_TASK_MAX_PER_TASK}, max_total=${TRAJECTORY_MAX_TOTAL})"
  fi
else
  echo "[config] base yaml ${BASE_CUSTOM_CONFIG_PATH} not found; MAX_TURN=${MAX_TURN} will not take effect"
fi

if [[ "${HARNESS_OPTION}" == "a3s-code" && "${DRY_RUN}" != "1" ]]; then
  if ! a3s_code_import_check; then
    A3S_SDK_DIR="${A3S_CODE_REPO_ROOT}/sdk/python"
    if [[ -d "${A3S_SDK_DIR}" ]]; then
      log "Installing a3s_code SDK with ${TRAIN_PYTHON} from ${A3S_SDK_DIR}"
      (
        cd "${A3S_SDK_DIR}"
        "${TRAIN_PYTHON}" -m pip install -q maturin
        "${TRAIN_PYTHON}" -m maturin develop --release
      )
    elif "${TRAIN_PYTHON}" -m pip show a3s-code >/dev/null 2>&1; then
      a3s_code_print_cache_hint
      exit 1
    else
      log "Installing ${A3S_CODE_PIP_PACKAGE} with ${TRAIN_PYTHON}"
      "${TRAIN_PYTHON}" -m pip install "${A3S_CODE_PIP_PACKAGE}"
    fi
    if ! a3s_code_import_check; then
      a3s_code_print_cache_hint
      exit 1
    fi
  fi
fi

if [[ "${HARNESS_OPTION}" == "claude-code" && "${DRY_RUN}" != "1" ]]; then
  claude_code_preflight
fi

# Symlinks for backward compatibility. Dry-run avoids touching stable repo links.
if [[ "${DRY_RUN}" != "1" ]]; then
  ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/latest" 2>/dev/null || true
  # Keep old logs/latest symlink for tools that expect it
  LOG_BASE="${SCRIPT_DIR}/logs"
  mkdir -p "${LOG_BASE}" 2>/dev/null || true
  ln -sfn "${RUN_LOG_DIR}" "${LOG_BASE}/latest" 2>/dev/null || true
fi

# Only create ckpt dir and set SAVE_CKPT when saving is enabled
if (( MAX_CKPT_KEEP > 0 )); then
  SAVE_CKPT="${SAVE_CKPT:-${CKPT_ROOT}/${RUN_ID}}"
else
  SAVE_CKPT=""
fi
RESUME_LOAD="${RESUME_LOAD:-${SAVE_CKPT}}"

# Pre-flight: refuse to start if EXPORT_ROOT has < 80GB free (only when saving).
if (( MAX_CKPT_KEEP > 0 )); then
  AVAIL_GB=$(df -BG --output=avail "${EXPORT_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [[ -n "${AVAIL_GB}" && "${AVAIL_GB}" -lt 80 ]]; then
    echo "[ERROR] Free space at ${EXPORT_ROOT} is only ${AVAIL_GB}G, need >= 80G"
    echo "        Clean old ckpts or set EXPORT_ROOT to a larger disk."
    df -h "${EXPORT_ROOT}" 2>&1 | tail -2
    exit 1
  fi
fi

RUN_LOG="${RUN_LOG_DIR}/train.log"

# ── Auto-mirror logs to a stable path that Claude can Read directly ──
# Canonical logs live under runs/<run>/.  The repo-root tmp_doc_latest path is
# kept only as a compatibility symlink to the current run directory, so legacy
# tmp_doc_latest/remote_logs also resolves under runs/<run>/remote_logs.  New
# tmp_doc_<timestamp> directories are no longer created at the repo root.
TMP_DOC_ROOT="${RUN_LOG_DIR}/mirror"
TMP_DOC_LATEST="${TMP_DOC_ROOT}"
mkdir -p "${TMP_DOC_ROOT}"

GPU_RUN_LOG="${TMP_DOC_ROOT}/gpu_run.log"      # full stdout/stderr
GPU_ERR_LOG="${TMP_DOC_ROOT}/gpu_err.log"      # filtered errors (populated on failure)
GPU_TAIL_LOG="${TMP_DOC_ROOT}/gpu_tail.log"    # last ~300 lines (populated on failure)
if [[ "${DRY_RUN}" != "1" ]]; then
  TMP_DOC_LATEST="${REPO_ROOT}/tmp_doc_latest"
  if ! ln -sfnT "${RUN_DIR}" "${TMP_DOC_LATEST}" 2>/dev/null; then
    echo "[WARN] Could not update ${TMP_DOC_LATEST}; existing non-symlink directory may need manual archival."
  fi
  ln -sfnT "${GPU_RUN_LOG}" "${RUN_DIR}/gpu_run.log" 2>/dev/null || true
  ln -sfnT "${GPU_ERR_LOG}" "${RUN_DIR}/gpu_err.log" 2>/dev/null || true
  ln -sfnT "${GPU_TAIL_LOG}" "${RUN_DIR}/gpu_tail.log" 2>/dev/null || true
fi

# Tee everything to both the run-specific file and tmp_doc copy
exec > >(tee -a "${RUN_LOG}" "${GPU_RUN_LOG}") 2>&1
echo "========================================"
echo "  Terminal-RL Run: ${RUN_NAME}"
echo "  Log dir:  ${RUN_LOG_DIR}"
echo "  Metrics:  ${TERMINAL_METRICS_JSONL} (structured=${TERMINAL_STRUCTURED_METRICS}, wandb=${TERMINAL_WANDB_METRIC_PROFILE})"
echo "  Harness:  ${HARNESS_OPTION}"
echo "  Ckpt:     ${SAVE_CKPT:-<disabled>}"
echo "  HF_CKPT:  ${HF_CKPT}"
echo "  REF_LOAD: ${REF_LOAD}"
echo "  MAX_CKPT_KEEP: ${MAX_CKPT_KEEP}"
echo "========================================"

# ── Model args (source qwen3-8B.sh) ──────────────────────────────────
source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"

# ── Dataset & Reward Configuration ───────────────────────────────────
# DATASET: which data to train on
#   seta    = seta_env only (capability tasks, Docker-based evaluation)
#   safety  = Agent-SafetyBench only (safety tasks, no Docker needed)
#   agentharm = inspect_evals/agentharm only (safety tool tasks, no Docker needed)
#   mixed   = configurable mix of seta / safety / agentharm
#
# SETA_SAFETY: safety reward mode for seta_env data
#   none       = pure outcome reward (2*accuracy - 1), no safety signal
#   clawsentry = outcome + ClawSentry L1 safety score (weighted by SAFETY_REWARD_COEF)
#
# SAFETY_BENCH_REWARD: reward mode for Agent-SafetyBench data
#   rule       = fulfillable-based: refuse harmful=+1, comply harmful=-1
#   dense_rule = rule-compatible dense shaping for GRPO
#   clawsentry = use ClawSentry safety score (same mechanism as seta)
#
# AGENTHARM_REWARD: reward mode for AgentHarm data
#   rule       = target-tool/refusal based safety reward
#   dense_rule = rule-compatible dense shaping for GRPO
#   clawsentry = use ClawSentry safety score (same mechanism as seta)
#
# ALGO:
#   grpo = existing baseline path
#   dapo = verl DAPO recipe knobs on top of GRPO estimator:
#          clip-higher, token-level loss, optional dynamic sampling, overlong shaping
ALGO="${ALGO:-dapo}"
case "${ALGO}" in
  grpo|dapo) ;;
  *)
    echo "[ERROR] Unknown ALGO=${ALGO}. Use: grpo|dapo"
    exit 1
    ;;
esac
export ALGO

DATASET="${DATASET:-mixed}"
SETA_SAFETY="${SETA_SAFETY:-none}"
SAFETY_BENCH_REWARD="${SAFETY_BENCH_REWARD:-dense_rule}"
AGENT_SAFETYBENCH_REMOTE_ENV="${AGENT_SAFETYBENCH_REMOTE_ENV:-0}"
AGENT_SAFETYBENCH_ROOT="${AGENT_SAFETYBENCH_ROOT:-/mnt/shared-storage-user/puyuan/code/Agent-SafetyBench}"
AGENTHARM_REWARD="${AGENTHARM_REWARD:-dense_rule}"
AGENTHARM_REMOTE_ENV="${AGENTHARM_REMOTE_ENV:-0}"
AGENTHARM_ROOT="${AGENTHARM_ROOT:-/mnt/shared-storage-user/puyuan/code/inspect_evals/src/inspect_evals/agentharm}"

SETA_DATA="${SCRIPT_DIR}/dataset/seta_env_convert/train.jsonl"
SAFETY_DATA="${SCRIPT_DIR}/dataset/agent_safetybench_convert/train.jsonl"
AGENTHARM_RAW_DIR="${SCRIPT_DIR}/dataset/agentharm"
AGENTHARM_DATA="${SCRIPT_DIR}/dataset/agentharm_convert/train.jsonl"

ensure_agentharm_dataset() {
  if [[ ! -d "${AGENTHARM_RAW_DIR}" ]]; then
    echo "[ERROR] AgentHarm raw data dir not found: ${AGENTHARM_RAW_DIR}"
    exit 1
  fi
  python3 "${SCRIPT_DIR}/data_utils/convert_agentharm_to_dataset.py" \
    --input-dir "${AGENTHARM_RAW_DIR}" \
    --output-dir "${SCRIPT_DIR}/dataset/agentharm_convert"
}

INCLUDES_SETA="0"
INCLUDES_SAFETY="0"
INCLUDES_AGENTHARM="0"
MIX_SETA_RATIO="${MIX_SETA_RATIO:-6}"
MIX_SAFETY_RATIO="${MIX_SAFETY_RATIO:-2}"
MIX_AGENTHARM_RATIO="${MIX_AGENTHARM_RATIO:-2}"
MIX_MODE="${MIX_MODE:-all_visible}"
export MIX_MODE

case "${DATASET}" in
  seta)
    INCLUDES_SETA="1"
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${SETA_DATA}}"
    ;;
  safety)
    INCLUDES_SAFETY="1"
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${SAFETY_DATA}}"
    ;;
  agentharm)
    INCLUDES_AGENTHARM="1"
    ensure_agentharm_dataset
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${AGENTHARM_DATA}}"
    ;;
  mixed)
    if [[ -n "${MIX_AGENTHARM_RATIO:-}" ]]; then
      ensure_agentharm_dataset
      MIXED_DATA="${SCRIPT_DIR}/dataset/mixed_sources.jsonl"
      MIX_ARGS=(
        --output "${MIXED_DATA}"
        --seed "${MIX_SEED:-42}"
        --mode "${MIX_MODE}"
      )
      MIX_LABELS=()
      add_mix_source() {
        local path="$1"
        local ratio="$2"
        local label="$3"
        if [[ -z "${ratio}" || "${ratio}" == "0" ]]; then
          return
        fi
        if [[ ! -f "${path}" ]]; then
          echo "[ERROR] mixed source not found: ${path}"
          exit 1
        fi
        MIX_ARGS+=(--source "${path}:${ratio}")
        MIX_LABELS+=("${label}(${ratio})")
      }
      add_mix_source "${SETA_DATA}" "${MIX_SETA_RATIO:-}" "seta"
      add_mix_source "${SAFETY_DATA}" "${MIX_SAFETY_RATIO:-}" "safety"
      add_mix_source "${AGENTHARM_DATA}" "${MIX_AGENTHARM_RATIO:-}" "agentharm"
      if [[ "${#MIX_LABELS[@]}" -eq 0 ]]; then
        echo "[ERROR] No mixed sources selected. Set MIX_SETA_RATIO, MIX_SAFETY_RATIO, or MIX_AGENTHARM_RATIO to a positive value."
        exit 1
      fi
      [[ -n "${MIX_SETA_RATIO:-}" && "${MIX_SETA_RATIO}" != "0" ]] && INCLUDES_SETA="1"
      [[ -n "${MIX_SAFETY_RATIO:-}" && "${MIX_SAFETY_RATIO}" != "0" ]] && INCLUDES_SAFETY="1"
      [[ -n "${MIX_AGENTHARM_RATIO:-}" && "${MIX_AGENTHARM_RATIO}" != "0" ]] && INCLUDES_AGENTHARM="1"
      if [[ -n "${MIX_TOTAL:-}" ]]; then
        MIX_ARGS+=(--total "${MIX_TOTAL}")
      fi
      if [[ -n "${MIX_OVERSAMPLE:-}" ]]; then
        MIX_ARGS+=(--oversample)
      fi
      python "${SCRIPT_DIR}/data_utils/mix_jsonl_datasets.py" "${MIX_ARGS[@]}"
      echo "[dataset] mixed sources: ${MIX_LABELS[*]} -> ${MIXED_DATA}"
    else
      INCLUDES_SETA="1"
      INCLUDES_SAFETY="1"
      MIXED_DATA="${SCRIPT_DIR}/dataset/mixed_seta_safety.jsonl"
      if [[ ! -f "${MIXED_DATA}" ]] || [[ "${SETA_DATA}" -nt "${MIXED_DATA}" ]] || [[ "${SAFETY_DATA}" -nt "${MIXED_DATA}" ]]; then
        if [[ -n "${MIX_SETA_RATIO:-}" ]] || [[ -n "${MIX_SAFETY_RATIO:-}" ]]; then
          MIX_ARGS=(
            --source "${SETA_DATA}:${MIX_SETA_RATIO:-1}"
            --source "${SAFETY_DATA}:${MIX_SAFETY_RATIO:-1}"
            --output "${MIXED_DATA}"
            --seed "${MIX_SEED:-42}"
            --mode "${MIX_MODE}"
          )
          if [[ -n "${MIX_TOTAL:-}" ]]; then
            MIX_ARGS+=(--total "${MIX_TOTAL}")
          fi
          if [[ -n "${MIX_OVERSAMPLE:-}" ]]; then
            MIX_ARGS+=(--oversample)
          fi
          python "${SCRIPT_DIR}/data_utils/mix_jsonl_datasets.py" "${MIX_ARGS[@]}"
        else
          cat "${SETA_DATA}" "${SAFETY_DATA}" > "${MIXED_DATA}"
          echo "[dataset] merged seta($(wc -l < "${SETA_DATA}")) + safety($(wc -l < "${SAFETY_DATA}")) -> ${MIXED_DATA}"
        fi
      fi
    fi
    ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${MIXED_DATA}}"
    ;;
  *)
    echo "[ERROR] Unknown DATASET=${DATASET}. Use: seta|safety|agentharm|mixed"
    exit 1
    ;;
esac

if [[ -z "${ROLLOUT_PROMPT_DATA}" ]]; then
  echo "[ERROR] ROLLOUT_PROMPT_DATA is unset."
  exit 1
fi
if [[ ! -f "${ROLLOUT_PROMPT_DATA}" ]]; then
  echo "[ERROR] ROLLOUT_PROMPT_DATA=${ROLLOUT_PROMPT_DATA} not found"
  exit 1
fi
echo "[config] ALGO=${ALGO} DATASET=${DATASET} SETA_SAFETY=${SETA_SAFETY} SAFETY_BENCH_REWARD=${SAFETY_BENCH_REWARD} AGENTHARM_REWARD=${AGENTHARM_REWARD}"
echo "[config] sources seta=${INCLUDES_SETA} safety=${INCLUDES_SAFETY} agentharm=${INCLUDES_AGENTHARM}"
echo "[config] data=${ROLLOUT_PROMPT_DATA}"

NEEDS_ENV_ROUTER="0"
if [[ "${INCLUDES_SETA}" == "1" ]]; then
  NEEDS_ENV_ROUTER="1"
fi
if [[ "${INCLUDES_SAFETY}" == "1" && "${AGENT_SAFETYBENCH_REMOTE_ENV}" == "1" ]]; then
  NEEDS_ENV_ROUTER="1"
fi
if [[ "${INCLUDES_AGENTHARM}" == "1" && "${AGENTHARM_REMOTE_ENV}" == "1" ]]; then
  NEEDS_ENV_ROUTER="1"
fi
echo "[config] needs_env_router=${NEEDS_ENV_ROUTER} AGENT_SAFETYBENCH_REMOTE_ENV=${AGENT_SAFETYBENCH_REMOTE_ENV} AGENTHARM_REMOTE_ENV=${AGENTHARM_REMOTE_ENV}"

# Optional dataset blacklist (issue #3 §1.X / §2.x stuck offenders).
# Default-ON; set USE_BLACKLIST=0 to keep the raw dataset.
USE_BLACKLIST="${USE_BLACKLIST:-1}"
DATASET_BLACKLIST="${DATASET_BLACKLIST:-786,96,90,456,856,210,999,305,25,684,345,553,962,916,1264,282,324,768,46,996}"
if [[ "${USE_BLACKLIST}" == "1" && -n "${DATASET_BLACKLIST}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    FILTERED_DATA="${RUN_DIR}/config/$(basename "${ROLLOUT_PROMPT_DATA%.jsonl}").filtered.jsonl"
  else
    FILTERED_DATA="${ROLLOUT_PROMPT_DATA%.jsonl}.filtered.jsonl"
  fi
  python3 - "$ROLLOUT_PROMPT_DATA" "$FILTERED_DATA" "$DATASET_BLACKLIST" <<'PY'
import json, sys
src, dst, blk = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
kept = dropped = 0
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            fout.write(line); kept += 1; continue
        # try common task-id fields
        tid = str(obj.get("task_name") or obj.get("task_id")
                  or obj.get("metadata", {}).get("task_name")
                  or obj.get("metadata", {}).get("task_id") or "")
        if tid in blk:
            dropped += 1
        else:
            fout.write(line); kept += 1
print(f"[blacklist] kept={kept} dropped={dropped} blacklist_size={len(blk)}")
PY
  ROLLOUT_PROMPT_DATA="${FILTERED_DATA}"
  echo "[blacklist] using filtered dataset: ${ROLLOUT_PROMPT_DATA}"
fi

# ── Router / worker URLs ─────────────────────────────────────────────
WORKER_URLS="${WORKER_URLS:-}"
WORKER_URLS_FILE="${WORKER_URLS_FILE:-${SCRIPT_DIR}/worker_urls.txt}"
WORKER_URLS_RELOAD_INTERVAL="${WORKER_URLS_RELOAD_INTERVAL:-120}"

read_worker_urls_from_file() {
  local file="$1"
  python3 - "$file" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    sys.exit(0)
urls = []
for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.split("#", 1)[0].strip()
    if not line:
        continue
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if line.startswith("WORKER_URLS="):
        line = line.split("=", 1)[1].strip()
    line = line.strip().strip('"').strip("'")
    urls.extend(part.rstrip("/") for part in re.split(r"[,\s]+", line) if part)
print(",".join(urls))
PY
}

if [[ -z "${WORKER_URLS}" && -f "${WORKER_URLS_FILE}" ]]; then
  WORKER_URLS="$(read_worker_urls_from_file "${WORKER_URLS_FILE}")"
fi
if [[ "${DRY_RUN}" != "1" && "${NEEDS_ENV_ROUTER}" == "1" && -z "${WORKER_URLS}" ]]; then
  echo "[ERROR] WORKER_URLS is unset. Example:"
  echo "        export WORKER_URLS=http://<worker-ip>:18081"
  echo "        or write that URL into WORKER_URLS_FILE=${WORKER_URLS_FILE}"
  exit 1
fi
if [[ "${NEEDS_ENV_ROUTER}" == "1" && -n "${WORKER_URLS}" ]]; then
  mkdir -p "$(dirname "${WORKER_URLS_FILE}")"
  if [[ ! -s "${WORKER_URLS_FILE}" ]]; then
    printf "%s\n" "${WORKER_URLS}" > "${WORKER_URLS_FILE}"
  fi
fi
export WORKER_URLS WORKER_URLS_FILE WORKER_URLS_RELOAD_INTERVAL

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
if [[ -z "${MASTER_ADDR:-}" ]]; then
  MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
export MASTER_ADDR
NODE_IP="${MASTER_ADDR}"

export USE_REMOTE_ENV="${USE_REMOTE_ENV:-${NEEDS_ENV_ROUTER}}"
export PROVIDER_NAME="${PROVIDER_NAME:-build}"
export ENV_SERVER_BIND_HOST="${ENV_SERVER_BIND_HOST:-0.0.0.0}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18080}"
export ENV_SERVER_HOST="${ENV_SERVER_HOST:-${MASTER_ADDR}}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://${ENV_SERVER_HOST}:${ENV_SERVER_PORT}}"
export START_ENV_POOL_SERVER="${START_ENV_POOL_SERVER:-${NEEDS_ENV_ROUTER}}"
export AGENT_SAFETYBENCH_REMOTE_ENV
export AGENTHARM_REMOTE_ENV
export AGENTHARM_ROOT
export AGENTHARM_REWARD

ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-${ENV_SERVER_PORT}}"
CHECK_HOST="${CHECK_HOST:-127.0.0.1}"
CHECK_WAIT_SECS="${CHECK_WAIT_SECS:-60}"
READY_PROBE_TIMEOUT="${READY_PROBE_TIMEOUT:-5}"
ROUTER_REQUIRE_READY="${ROUTER_REQUIRE_READY:-1}"
ROUTER_READY_WAIT_FOREVER="${ROUTER_READY_WAIT_FOREVER:-0}"
WORKER_PREFLIGHT_REQUIRE_READY="${WORKER_PREFLIGHT_REQUIRE_READY:-1}"
WORKER_PREFLIGHT_TIMEOUT="${WORKER_PREFLIGHT_TIMEOUT:-5}"
export ROUTER_READYZ_WORKER_TIMEOUT="${ROUTER_READYZ_WORKER_TIMEOUT:-${WORKER_PREFLIGHT_TIMEOUT}}"
AUTO_CLOSE_STALE_WORKER_RUNS="${AUTO_CLOSE_STALE_WORKER_RUNS:-1}"
STALE_WORKER_CLOSE_INTERVAL="${STALE_WORKER_CLOSE_INTERVAL:-10}"
STALE_WORKER_CLOSE_TIMEOUT="${STALE_WORKER_CLOSE_TIMEOUT:-10}"
STALE_WORKER_REPAIR_MIN_AGE="${STALE_WORKER_REPAIR_MIN_AGE:-0}"
STALE_WORKER_REPAIR_MAX_REPAIRS="${STALE_WORKER_REPAIR_MAX_REPAIRS:-20}"
if ! [[ "${STALE_WORKER_CLOSE_INTERVAL}" =~ ^[0-9]+$ ]] || [[ "${STALE_WORKER_CLOSE_INTERVAL}" -le 0 ]]; then
  STALE_WORKER_CLOSE_INTERVAL=10
fi

probe_ready_endpoint() {
  local base_url="$1"
  local label="$2"
  local timeout_s="${3:-${READY_PROBE_TIMEOUT}}"
  local tmp code path body

  tmp="$(mktemp /tmp/openclaw_ready.XXXXXX 2>/dev/null || printf '/tmp/openclaw_ready.%s' "$$")"
  path="/readyz"
  code="$(curl -sS --max-time "${timeout_s}" --noproxy '*' \
    -o "${tmp}" -w '%{http_code}' "${base_url}${path}" 2>/dev/null || echo "000")"
  if [[ "${code}" == "404" ]]; then
    path="/healthz"
    code="$(curl -sS --max-time "${timeout_s}" --noproxy '*' \
      -o "${tmp}" -w '%{http_code}' "${base_url}${path}" 2>/dev/null || echo "000")"
  fi

  if [[ "${code}" =~ ^2[0-9][0-9]$ ]]; then
    log "  [OK] ${label}${path}"
    rm -f "${tmp}" 2>/dev/null || true
    return 0
  fi

  body="$(head -c 300 "${tmp}" 2>/dev/null || true)"
  log "  [WARN] ${label}${path} not ready HTTP ${code}${body:+: ${body}}"
  rm -f "${tmp}" 2>/dev/null || true
  return 1
}

extract_stale_lease_ids() {
  local json_path="$1"
  python3 - "${json_path}" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
except Exception:
    sys.exit(0)

seen = set()
out = []

def walk(obj):
    if isinstance(obj, dict):
        stale_runs = obj.get("stale_runs")
        if isinstance(stale_runs, list):
            for item in stale_runs:
                if not isinstance(item, dict):
                    continue
                lease_id = item.get("lease_id")
                if isinstance(lease_id, str) and lease_id and lease_id not in seen:
                    seen.add(lease_id)
                    out.append(lease_id)
        for value in obj.values():
            walk(value)
    elif isinstance(obj, list):
        for value in obj:
            walk(value)

walk(data)
for lease_id in out:
    print(lease_id)
PY
}

close_stale_worker_runs() {
  local base_url="$1"
  local label="$2"
  local timeout_s="${3:-${STALE_WORKER_CLOSE_TIMEOUT}}"
  local tmp ids_tmp path code lease_id close_tmp close_code close_body count repair_tmp repair_code repair_body

  if [[ "${AUTO_CLOSE_STALE_WORKER_RUNS}" != "1" ]]; then
    return 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    log "  [WARN] stale-run cleanup skipped for ${label}: python3 not found"
    return 1
  fi

  tmp="$(mktemp /tmp/openclaw_stale_ready.XXXXXX 2>/dev/null || printf '/tmp/openclaw_stale_ready.%s' "$$")"
  ids_tmp="$(mktemp /tmp/openclaw_stale_ids.XXXXXX 2>/dev/null || printf '/tmp/openclaw_stale_ids.%s' "$$")"
  : > "${ids_tmp}"
  for path in /readyz /status; do
    code="$(curl -sS --max-time "${timeout_s}" --noproxy '*' \
      -o "${tmp}" -w '%{http_code}' "${base_url}${path}" 2>/dev/null || echo "000")"
    if [[ "${code}" =~ ^[0-9][0-9][0-9]$ ]]; then
      extract_stale_lease_ids "${tmp}" >> "${ids_tmp}" || true
    fi
  done

  if [[ ! -s "${ids_tmp}" ]]; then
    rm -f "${tmp}" "${ids_tmp}" 2>/dev/null || true
    return 1
  fi

  repair_tmp="$(mktemp /tmp/openclaw_stale_repair.XXXXXX 2>/dev/null || printf '/tmp/openclaw_stale_repair.%s' "$$")"
  repair_code="$(curl -sS --max-time "${timeout_s}" --noproxy '*' \
    -X POST -H 'Content-Type: application/json' \
    --data "{\"reason\":\"startup_readyz_repair\",\"min_age\":${STALE_WORKER_REPAIR_MIN_AGE},\"max_repairs\":${STALE_WORKER_REPAIR_MAX_REPAIRS}}" \
    -o "${repair_tmp}" -w '%{http_code}' "${base_url}/repair/stale_runs" 2>/dev/null || echo "000")"
  repair_body="$(head -c 320 "${repair_tmp}" 2>/dev/null || true)"
  if [[ "${repair_code}" =~ ^2[0-9][0-9]$ ]]; then
    log "  [REPAIR] ${label}: repair stale runs HTTP ${repair_code}${repair_body:+: ${repair_body}}"
    rm -f "${tmp}" "${ids_tmp}" "${repair_tmp}" 2>/dev/null || true
    return 0
  elif [[ "${repair_code}" != "404" && "${repair_code}" != "000" ]]; then
    log "  [WARN] ${label}: repair stale runs endpoint HTTP ${repair_code}${repair_body:+: ${repair_body}}"
  else
    log "  [WARN] ${label}: /repair/stale_runs unavailable; falling back to duplicate /close requests. Restart worker to load the repair endpoint if stale in-flight runs persist."
  fi
  rm -f "${repair_tmp}" 2>/dev/null || true

  count=0
  while IFS= read -r lease_id; do
    [[ -n "${lease_id}" ]] || continue
    close_tmp="$(mktemp /tmp/openclaw_stale_close.XXXXXX 2>/dev/null || printf '/tmp/openclaw_stale_close.%s' "$$")"
    close_code="$(curl -sS --max-time "${timeout_s}" --noproxy '*' \
      -X POST -H 'Content-Type: application/json' \
      --data "{\"lease_id\":\"${lease_id}\"}" \
      -o "${close_tmp}" -w '%{http_code}' "${base_url}/close" 2>/dev/null || echo "000")"
    close_body="$(head -c 240 "${close_tmp}" 2>/dev/null || true)"
    log "  [REPAIR] ${label}: close stale lease=${lease_id} HTTP ${close_code}${close_body:+: ${close_body}}"
    rm -f "${close_tmp}" 2>/dev/null || true
    count=$((count + 1))
  done < <(python3 - "${ids_tmp}" <<'PY'
import sys
seen = set()
for line in open(sys.argv[1], "r", encoding="utf-8", errors="ignore"):
    value = line.strip()
    if value and value not in seen:
        seen.add(value)
        print(value)
PY
  )

  rm -f "${tmp}" "${ids_tmp}" 2>/dev/null || true
  [[ "${count}" -gt 0 ]]
}

close_stale_runs_for_all_workers() {
  local reason="$1"
  local repaired=0
  local _w
  IFS=',' read -r -a _STALE_WORKERS <<< "${WORKER_URLS}"
  for _w in "${_STALE_WORKERS[@]}"; do
    [[ -n "${_w}" ]] || continue
    if close_stale_worker_runs "${_w}" "${_w} (${reason})" "${STALE_WORKER_CLOSE_TIMEOUT}"; then
      repaired=1
    fi
  done
  [[ "${repaired}" -eq 1 ]]
}

# ── Robustness knobs (informed by issue #3 postmortem) ───────────────
# Router forward to pool_server (tuned for burst of docker-compose down/up):
#   - ROUTER_FORWARD_TIMEOUT: raise 600 → 900 (issue #3 §1.X-E: 90s → 600s still
#     tight when pool is processing 32 concurrent closes; 15min is generous).
#   - ROUTER_FORWARD_RETRIES: 1 → 3 (matches agent_runner http_utils retries).
#   - ROUTER_FORWARD_RETRY_BACKOFF: 0.2 → 1.0 (exponential-ish, gives the pool
#     a real window to finish in-flight docker operations before the retry).
export ROUTER_FORWARD_TIMEOUT="${ROUTER_FORWARD_TIMEOUT:-900}"
export ROUTER_FORWARD_RETRIES="${ROUTER_FORWARD_RETRIES:-3}"
export ROUTER_FORWARD_RETRY_BACKOFF="${ROUTER_FORWARD_RETRY_BACKOFF:-1.0}"
export ROUTER_PRESSURE_COOLDOWN="${ROUTER_PRESSURE_COOLDOWN:-60}"

# ── ClawSentry safety reward (L1-only, reward-only, linear-fusion baseline) ──
# Gateway runs on the same host as router_server (CPU master). All decisions
# are reward-shaping signals; agent actions are never blocked.
# ClawSentry is enabled only for the active dataset family.
# SAFETY_REWARD_COEF controls the linear weight (default 0 unless ClawSentry is explicitly enabled).
CLAWSENTRY_NEEDED="0"
if [[ "${INCLUDES_SETA}" == "1" && "${SETA_SAFETY}" == "clawsentry" ]]; then
  CLAWSENTRY_NEEDED="1"
fi
if [[ "${INCLUDES_SAFETY}" == "1" && "${SAFETY_BENCH_REWARD}" == "clawsentry" ]]; then
  CLAWSENTRY_NEEDED="1"
fi
if [[ "${INCLUDES_AGENTHARM}" == "1" && "${AGENTHARM_REWARD}" == "clawsentry" ]]; then
  CLAWSENTRY_NEEDED="1"
fi
export SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0}"
export SAFETY_REWARD_SUMMARY_WEIGHT="${SAFETY_REWARD_SUMMARY_WEIGHT:-0.3}"
export SAFETY_REWARD_TIMEOUT="${SAFETY_REWARD_TIMEOUT:-2.0}"
export SAFETY_REWARD_ZERO_THRESHOLD="${SAFETY_REWARD_ZERO_THRESHOLD:-1.5}"
export CS_GATEWAY_PORT="${CS_GATEWAY_PORT:-8090}"
export CS_HTTP_HOST="${CS_HTTP_HOST:-127.0.0.1}"
export CS_HTTP_URL="http://${CS_HTTP_HOST}:${CS_GATEWAY_PORT}"
export CS_AUTH_TOKEN="${CS_AUTH_TOKEN:-}"
export CS_TRAJECTORY_DB_PATH="${CS_TRAJECTORY_DB_PATH:-/tmp/clawsentry-train.db}"
export CS_LLM_PROVIDER="${CS_LLM_PROVIDER:-}"
export CS_L3_ENABLED="${CS_L3_ENABLED:-false}"
export CS_EVOLVING_ENABLED="${CS_EVOLVING_ENABLED:-false}"

# ── Trajectory export (parallels swe-rl export/swe_rollouts) ─────────────────
# Trajectory export is now ON by default (writes to runs/{run_id}/trajectories/).
# Set TERMINAL_SAVE_TRAJ_DIR="" to disable.
export TERMINAL_SAVE_TRAJ_DIR="${TERMINAL_SAVE_TRAJ_DIR}"
export TRAJECTORY_SAVE_INTERVAL="${TRAJECTORY_SAVE_INTERVAL}"
export TRAJECTORY_SAVE_INTERVAL_SETA
export TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH
export TRAJECTORY_SAVE_INTERVAL_AGENTHARM
export TRAJECTORY_SAVE_POLICY TRAJECTORY_TASK_SAVE_INTERVAL TRAJECTORY_TASK_MAX_PER_STEP
export TRAJECTORY_TASK_MAX_PER_TASK TRAJECTORY_MAX_TOTAL TRAJECTORY_SAVE_REWARD_STRATA
export TRAJECTORY_SAVE_LOG_DECISIONS

# Proxy bypass: some environments inject http_proxy/HTTPS_PROXY via shell rc.
# aiohttp + requests will then try to tunnel the internal router→worker traffic
# through a proxy, causing spurious connection failures. Explicitly list all
# hosts on the rollout datapath as NO_PROXY (matches swe-rl v1/v4 pattern).
ALL_WORKER_HOSTS=""
if [[ -n "${WORKER_URLS}" ]]; then
  ALL_WORKER_HOSTS="$(echo "${WORKER_URLS}" | tr ',' '\n' \
    | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
fi
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,${MASTER_ADDR}${ALL_WORKER_HOSTS:+,${ALL_WORKER_HOSTS}}}"
export no_proxy="${NO_PROXY}"

# Router uses `python3` which, after the PATH export above, resolves to
# lightrft_py312/bin/python. Override ROUTER_PYTHON if you want a different env.
ROUTER_PYTHON="${ROUTER_PYTHON:-python3}"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_terminal_rl}"
mkdir -p "${RAY_TMPDIR}"
SLIME_RAY_PLACEMENT_GPU_PROBE="${SLIME_RAY_PLACEMENT_GPU_PROBE:-0}"
export SLIME_RAY_PLACEMENT_GPU_PROBE

# ── Args ─────────────────────────────────────────────────────────────
CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --rotary-base 1000000
)
# Only add --save / --load / --save-interval when checkpointing is enabled
if [[ -n "${SAVE_CKPT}" ]]; then
  CKPT_ARGS+=(--save "${SAVE_CKPT}" --save-interval "${SAVE_INTERVAL}")
fi
if [[ -n "${RESUME_LOAD}" ]]; then
  CKPT_ARGS+=(--load "${RESUME_LOAD}")
fi

if [[ "${DEBUG_MODE}" == "1" ]]; then
  NUM_ROLLOUT=4
  ROLLOUT_BATCH_SIZE=4
  N_SAMPLES=2
  MAX_TOKENS_PER_GPU=8192
else
  NUM_ROLLOUT="${NUM_ROLLOUT:-2000}"
  # each rollout = ROLLOUT_BATCH_SIZE * N_SAMPLES concurrent lease requests.
  # Keep this baseline explicit and predictable without dynamic sampling.
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
  N_SAMPLES="${N_SAMPLES:-8}"
  MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
fi
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-16384}"
ROLLOUT_GENERATION_MAX_RETRIES="${ROLLOUT_GENERATION_MAX_RETRIES:-3}"
ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF="${ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF:-60}"
ROLLOUT_GENERATION_RETRY_MAX_BACKOFF="${ROLLOUT_GENERATION_RETRY_MAX_BACKOFF:-300}"
ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER="${ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER:-2.0}"
ROLLOUT_GENERATION_ENV_STORM_MAX_RETRIES="${ROLLOUT_GENERATION_ENV_STORM_MAX_RETRIES:-3}"
ROLLOUT_GENERATION_SKIP_ON_FAILURE="${ROLLOUT_GENERATION_SKIP_ON_FAILURE:-0}"

ROLLOUT_ARGS=(
  --prompt-data "${ROLLOUT_PROMPT_DATA}"
  --input-key task
  --rollout-shuffle
  --reward-key score
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"
  --num-steps-per-rollout 2
  --balance-data
  --rollout-generation-max-retries "${ROLLOUT_GENERATION_MAX_RETRIES}"
  --rollout-generation-retry-initial-backoff "${ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF}"
  --rollout-generation-retry-max-backoff "${ROLLOUT_GENERATION_RETRY_MAX_BACKOFF}"
  --rollout-generation-retry-backoff-multiplier "${ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER}"
  --rollout-generation-env-storm-max-retries "${ROLLOUT_GENERATION_ENV_STORM_MAX_RETRIES}"
)
if [[ "${ROLLOUT_GENERATION_SKIP_ON_FAILURE}" == "1" ]]; then
  ROLLOUT_ARGS+=(--rollout-generation-skip-on-failure)
fi

EVAL_ARGS=(
  --n-samples-per-eval-prompt 16
  --eval-max-response-len 16384
  --eval-top-p 1
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 1024
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --dynamic_history
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type k3
)

DAPO_EPS_CLIP_LOW="${DAPO_EPS_CLIP_LOW:-0.2}"
DAPO_EPS_CLIP_HIGH="${DAPO_EPS_CLIP_HIGH:-0.28}"
DAPO_USE_KL_LOSS="${DAPO_USE_KL_LOSS:-0}"
DAPO_KL_LOSS_COEF="${DAPO_KL_LOSS_COEF:-0.0}"
DAPO_KL_LOSS_TYPE="${DAPO_KL_LOSS_TYPE:-k3}"
DAPO_CALCULATE_PER_TOKEN_LOSS="${DAPO_CALCULATE_PER_TOKEN_LOSS:-1}"
DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"
DAPO_DYNAMIC_FILTER_PATH="${DAPO_DYNAMIC_FILTER_PATH:-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
DAPO_OVER_SAMPLING_BATCH_SIZE="${DAPO_OVER_SAMPLING_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
DAPO_FAILED_GROUP_ABORT_MIN_GROUPS="${DAPO_FAILED_GROUP_ABORT_MIN_GROUPS:-${ROLLOUT_BATCH_SIZE}}"
DAPO_FAILED_GROUP_ABORT_RATIO="${DAPO_FAILED_GROUP_ABORT_RATIO:-1.0}"
DAPO_GRPO_STD_NORMALIZATION="${DAPO_GRPO_STD_NORMALIZATION:-1}"
DAPO_OVERLONG_BUFFER_ENABLE="${DAPO_OVERLONG_BUFFER_ENABLE:-1}"
DAPO_OVERLONG_BUFFER_LEN="${DAPO_OVERLONG_BUFFER_LEN:-4096}"
DAPO_OVERLONG_PENALTY_FACTOR="${DAPO_OVERLONG_PENALTY_FACTOR:-1.0}"

DAPO_ARGS=(
  --advantage-estimator grpo
  --dynamic_history
  --eps-clip "${DAPO_EPS_CLIP_LOW}"
  --eps-clip-high "${DAPO_EPS_CLIP_HIGH}"
)

if [[ "${DAPO_CALCULATE_PER_TOKEN_LOSS}" == "1" ]]; then
  DAPO_ARGS+=(--calculate-per-token-loss)
fi
if [[ "${DAPO_USE_KL_LOSS}" == "1" ]]; then
  DAPO_ARGS+=(--use-kl-loss --kl-loss-coef "${DAPO_KL_LOSS_COEF}" --kl-loss-type "${DAPO_KL_LOSS_TYPE}")
fi
if [[ "${DAPO_DYNAMIC_SAMPLING}" == "1" ]]; then
  if (( DAPO_OVER_SAMPLING_BATCH_SIZE < ROLLOUT_BATCH_SIZE )); then
    echo "[ERROR] DAPO_OVER_SAMPLING_BATCH_SIZE(${DAPO_OVER_SAMPLING_BATCH_SIZE}) must be >= ROLLOUT_BATCH_SIZE(${ROLLOUT_BATCH_SIZE})"
    exit 1
  fi
  DAPO_ARGS+=(
    --dynamic-sampling-filter-path "${DAPO_DYNAMIC_FILTER_PATH}"
    --over-sampling-batch-size "${DAPO_OVER_SAMPLING_BATCH_SIZE}"
  )
  if (( DAPO_FAILED_GROUP_ABORT_MIN_GROUPS > 0 )); then
    DAPO_ARGS+=(
      --dynamic-sampling-failed-group-abort-min-groups "${DAPO_FAILED_GROUP_ABORT_MIN_GROUPS}"
      --dynamic-sampling-failed-group-abort-ratio "${DAPO_FAILED_GROUP_ABORT_RATIO}"
    )
  fi
fi
if [[ "${DAPO_GRPO_STD_NORMALIZATION}" == "0" ]]; then
  DAPO_ARGS+=(--disable-grpo-std-normalization)
fi

case "${ALGO}" in
  grpo)
    ALGO_ARGS=("${GRPO_ARGS[@]}")
    ALGO_EXTRA_ARGS="${EXTRA_GRPO_ARGS:-} ${EXTRA_ALGO_ARGS:-}"
    ;;
  dapo)
    ALGO_ARGS=("${DAPO_ARGS[@]}")
    ALGO_EXTRA_ARGS="${EXTRA_DAPO_ARGS:-} ${EXTRA_ALGO_ARGS:-}"
    ;;
esac
ALGO_EXTRA_ARGS_ARRAY=()
if [[ -n "${ALGO_EXTRA_ARGS// }" ]]; then
  # Preserve the existing unquoted EXTRA_GRPO_ARGS behavior for compatibility.
  ALGO_EXTRA_ARGS_ARRAY=(${ALGO_EXTRA_ARGS})
fi
log "Algorithm config: ALGO=${ALGO} args=${ALGO_ARGS[*]} extra=${ALGO_EXTRA_ARGS_ARRAY[*]:-<none>}"
log "Rollout retry config: max_retries=${ROLLOUT_GENERATION_MAX_RETRIES} initial_backoff=${ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF}s max_backoff=${ROLLOUT_GENERATION_RETRY_MAX_BACKOFF}s multiplier=${ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER} env_storm_max_retries=${ROLLOUT_GENERATION_ENV_STORM_MAX_RETRIES} skip_on_failure=${ROLLOUT_GENERATION_SKIP_ON_FAILURE}"
log "Exploration config: profile=${EXPLORATION_PROFILE} entropy=${EXPLORE_ENTROPY_COEF} intrinsic=${EXPLORE_INTRINSIC_ENABLED}/${EXPLORE_INTRINSIC} coef=${EXPLORE_INTRINSIC_COEF} schedule=${EXPLORE_INTRINSIC_SCHEDULE}/${EXPLORE_INTRINSIC_DECAY_STEPS} reducer=${EXPLORE_INTRINSIC_REDUCER} granularity=${EXPLORE_INTRINSIC_GRANULARITY} scope=${EXPLORE_INTRINSIC_SCOPE} score_components=${EXPLORE_SCORE_BONUS_COMPONENTS} safety_filter=${EXPLORE_SAFETY_FILTER_ENABLED}/${EXPLORE_SAFETY_FILTER} lprnd=${EXPLORE_LPRND_ENABLED}/${EXPLORE_LPRND} coef=${EXPLORE_LPRND_COEF} schedule=${EXPLORE_LPRND_SCHEDULE}/${EXPLORE_LPRND_DECAY_STEPS} agent57=${EXPLORE_AGENT57_LITE_ENABLED}/${EXPLORE_AGENT57_LITE} k=${EXPLORE_AGENT57_K} controller=${EXPLORE_AGENT57_CONTROLLER} ucb_eps=${EXPLORE_AGENT57_UCB_EPSILON} ucb_min=${EXPLORE_AGENT57_UCB_MIN_PER_ARM} ucb_value=${EXPLORE_AGENT57_UCB_VALUE} dataset_aware=${EXPLORE_AGENT57_UCB_DATASET_AWARE} ucb_seed=${EXPLORE_AGENT57_UCB_RANDOM_SEED:-<legacy>} episodic=${EXPLORE_AGENT57_EPISODIC_BACKEND} episodic_obs=${EXPLORE_AGENT57_EPISODIC_OBS_MODE} episodic_turn=${EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN} episodic_probe=${EXPLORE_AGENT57_EPISODIC_MULTI_PROBE_RADIUS} episodic_floor=${EXPLORE_AGENT57_EPISODIC_NOVELTY_FLOOR} combine=${EXPLORE_AGENT57_COMBINE_MODE} ngu_clip=${EXPLORE_AGENT57_NGU_MOD_CLIP} ngu_reducer=${EXPLORE_AGENT57_NGU_EPISODIC_REDUCER} life_mod=${EXPLORE_AGENT57_NGU_LIFE_MOD_MODE}/${EXPLORE_AGENT57_NGU_LIFE_MOD_STD_CLIP} max_bonus=${EXPLORE_AGENT57_MAX_BONUS} betas=${EXPLORE_AGENT57_ARM_BETAS} temps=${EXPLORE_AGENT57_ARM_TEMPERATURES:-<inherit>} temp_warmup=${EXPLORE_AGENT57_ARM_TEMPERATURE_WARMUP_ROLLOUTS} lifelong=${EXPLORE_AGENT57_LIFELONG_ENABLED}/${EXPLORE_AGENT57_LIFELONG} life_coef=${EXPLORE_AGENT57_LIFELONG_COEF} life_backend=${EXPLORE_AGENT57_LIFELONG_BACKEND} life_key=${EXPLORE_AGENT57_LIFELONG_KEY_VERSION} life_dataset=${EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET} life_task=${EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK} life_turn=${EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN} life_obs=${EXPLORE_AGENT57_LIFELONG_OBS_MODE} life_hier=${EXPLORE_AGENT57_LIFELONG_HIERARCHICAL} life_weights=${EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT}/${EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT}/${EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT} sqlite_timeout_ms=${EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS} sqlite_wal=${EXPLORE_AGENT57_SQLITE_WAL} life_decay=${EXPLORE_AGENT57_LIFELONG_COUNT_DECAY} life_capacity=${EXPLORE_AGENT57_LIFELONG_CAPACITY} trust=${EXPLORE_AGENT57_TRUST_GATE} cde_actor=${EXPLORE_CDE_ACTOR_ENABLED}/${EXPLORE_CDE_ACTOR} omega=${EXPLORE_CDE_ACTOR_OMEGA} alpha=${EXPLORE_CDE_ACTOR_ALPHA} kappa=${EXPLORE_CDE_ACTOR_KAPPA} gate=${EXPLORE_CDE_ACTOR_REWARD_GATE} decay_steps=${EXPLORE_CDE_ACTOR_DECAY_STEPS} post_norm_bonus=${EXPLORE_ADVANTAGE_BONUS_ENABLED}/${EXPLORE_ADVANTAGE_BONUS} mode=${EXPLORE_ADVANTAGE_BONUS_MODE} components=${EXPLORE_ADVANTAGE_BONUS_COMPONENTS} coef=${EXPLORE_ADVANTAGE_BONUS_COEF} lambda=${EXPLORE_ADVANTAGE_LAMBDA} lambda_schedule=${EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE}/${EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS} intrinsic_key=${EXPLORE_ADVANTAGE_INTRINSIC_KEY} arm_weight=${EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE} clip=${EXPLORE_ADVANTAGE_BONUS_CLIP} trunc_intr_scale=${EXPLORE_ADVANTAGE_TRUNCATED_INTRINSIC_SCALE} fail_intr_scale=${EXPLORE_ADVANTAGE_FAILED_INTRINSIC_SCALE} trunc_penalty=${EXPLORE_TRUNCATION_PENALTY} skip_zero_trainable=${SLIME_SKIP_ZERO_TRAINABLE_ROLLOUT}/${SLIME_SKIP_ZERO_TRAINABLE_TRAIN}"
if [[ "${ALGO}" == "dapo" ]]; then
  log "DAPO knobs: clip_low=${DAPO_EPS_CLIP_LOW} clip_high=${DAPO_EPS_CLIP_HIGH} token_loss=${DAPO_CALCULATE_PER_TOKEN_LOSS} dynamic_sampling=${DAPO_DYNAMIC_SAMPLING} failed_group_abort=${DAPO_FAILED_GROUP_ABORT_MIN_GROUPS}/${DAPO_FAILED_GROUP_ABORT_RATIO} overlong=${DAPO_OVERLONG_BUFFER_ENABLE}/${DAPO_OVERLONG_BUFFER_LEN}/${DAPO_OVERLONG_PENALTY_FACTOR}"
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --clip-grad 1.0
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

if [[ -n "${WANDB_KEY:-}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-terminal_rl}"
    --wandb-group   "${WANDB_GROUP:-qwen3-8b_4gpu}"
    --wandb-key     "${WANDB_KEY}"
    --wandb-dir     "${WANDB_DIR}"
  )
else
  WANDB_ARGS=()
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --no-gradient-accumulation-fusion
)

CUSTOM_ARGS=(
  --custom-generate-function-path generate.generate
  --custom-rollout-log-function-path rollout_log.rollout_log
  --custom-eval-rollout-log-function-path rollout_log.eval_rollout_log
)
if [[ "${EXPLORE_ADVANTAGE_BONUS_ENABLED}" == "1" ]]; then
  CUSTOM_ARGS+=(--custom-reward-post-process-path reward_postprocess.post_process_rewards)
fi
# --custom-config-path is optional in slime; only attach it if the yaml exists.
if [[ -f "${CUSTOM_CONFIG_PATH}" ]]; then
  CUSTOM_ARGS+=(--custom-config-path "${CUSTOM_CONFIG_PATH}")
else
  echo "WARN: custom config not found at ${CUSTOM_CONFIG_PATH}; skipping --custom-config-path"
fi

TRAIN_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${ACTOR_GPUS}"
  --rollout-num-gpus "${ROLLOUT_GPUS}"
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${ALGO_ARGS[@]}"
  "${ALGO_EXTRA_ARGS_ARRAY[@]}"
  "${WANDB_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${EVAL_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${CUSTOM_ARGS[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1: final train_async command only; router/Ray/training will not start"
  printf '[dry-run] '
  printf '%q ' "${TRAIN_PYTHON}" -u "${SLIME_DIR}/train_async.py" "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

# NOTE: safety reward params are passed via env vars (RUNTIME_ENV_JSON below),
# not CLI flags, because slime's argparse rejects unknown flags.

# ── Start router ─────────────────────────────────────────────────────
ROUTER_PID=""
CS_GATEWAY_PID=""
cleanup() {
  set +e
  if [[ -n "${ROUTER_PID}" ]] && kill -0 "${ROUTER_PID}" 2>/dev/null; then
    kill "${ROUTER_PID}" || true
  fi
  if [[ -n "${CS_GATEWAY_PID}" ]] && kill -0 "${CS_GATEWAY_PID}" 2>/dev/null; then
    kill "${CS_GATEWAY_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

ROUTER_LOG="${RUN_LOG_DIR}/router.log"
require_cmd curl
if [[ "${NEEDS_ENV_ROUTER}" == "1" ]]; then
  if [[ "${AUTO_CLOSE_STALE_WORKER_RUNS}" == "1" ]]; then
    log "Pre-cleaning stale worker runs before router readiness check..."
    close_stale_runs_for_all_workers "pre_router_start" || true
  fi
  log "Starting router on ${ROUTER_HOST}:${ROUTER_PORT} -> ${WORKER_URLS} (python=${ROUTER_PYTHON})"
  log "  worker_urls_file=${WORKER_URLS_FILE} reload_interval=${WORKER_URLS_RELOAD_INTERVAL}s"
  log "  forward_timeout=${ROUTER_FORWARD_TIMEOUT}s retries=${ROUTER_FORWARD_RETRIES} backoff=${ROUTER_FORWARD_RETRY_BACKOFF}s pressure_cooldown=${ROUTER_PRESSURE_COOLDOWN}s no_proxy=${NO_PROXY}"
  log "  readiness require_router=${ROUTER_REQUIRE_READY} wait_forever=${ROUTER_READY_WAIT_FOREVER} require_worker=${WORKER_PREFLIGHT_REQUIRE_READY} probe_timeout=${READY_PROBE_TIMEOUT}s worker_timeout=${ROUTER_READYZ_WORKER_TIMEOUT}s auto_close_stale=${AUTO_CLOSE_STALE_WORKER_RUNS}"
  (
    cd "${REPO_ROOT}"
    "${ROUTER_PYTHON}" -m terminal-rl.router_server \
      --host "${ROUTER_HOST}" --port "${ROUTER_PORT}" --workers "${WORKER_URLS}" \
      --workers-file "${WORKER_URLS_FILE}" --workers-reload-interval "${WORKER_URLS_RELOAD_INTERVAL}" \
      > "${ROUTER_LOG}" 2>&1 &
    echo $! > "${RUN_LOG_DIR}/router.pid"
  )
  ROUTER_PID="$(cat "${RUN_LOG_DIR}/router.pid")"
  log "Router PID=${ROUTER_PID}, log=${ROUTER_LOG}"

  # Wait for router readiness. /readyz validates at least one env worker; /healthz
  # is only used as fallback for older router implementations.
  ROUTER_READY=0
  i=1
  while true; do
    if probe_ready_endpoint "http://${CHECK_HOST}:${ROUTER_PORT}" "router http://${CHECK_HOST}:${ROUTER_PORT}" "${READY_PROBE_TIMEOUT}"; then
      log "router ready (attempt ${i})"
      ROUTER_READY=1
      break
    fi
    if [[ "${AUTO_CLOSE_STALE_WORKER_RUNS}" == "1" && $((i % STALE_WORKER_CLOSE_INTERVAL)) -eq 0 ]]; then
      close_stale_runs_for_all_workers "router_wait_attempt_${i}" || true
    fi
    if [[ -n "${ROUTER_PID}" ]] && ! kill -0 "${ROUTER_PID}" 2>/dev/null; then
      log "ERROR: router process exited before becoming ready; see ${ROUTER_LOG}"
      break
    fi
    if [[ "${ROUTER_READY_WAIT_FOREVER}" != "1" && "${i}" -ge "${CHECK_WAIT_SECS}" ]]; then
      break
    fi
    sleep 1
    i=$((i + 1))
  done
  if [[ "${ROUTER_READY}" != "1" ]]; then
    log "ERROR: router not ready after ${CHECK_WAIT_SECS}s"
    if [[ "${ROUTER_REQUIRE_READY}" == "1" ]]; then
      exit 1
    fi
  fi
  curl -fsS "http://${CHECK_HOST}:${ROUTER_PORT}/status" || true
  echo
else
  log "Skipping terminal env router; Agent-SafetyBench uses local env backend"
fi

# ── Start ClawSentry gateway (L1-only, reward-only) ──────────────────
if [[ "${CLAWSENTRY_NEEDED}" == "1" ]]; then
  CS_GATEWAY_LOG="${RUN_LOG_DIR}/clawsentry_gateway.log"
  log "Starting clawsentry-gateway on ${CS_HTTP_HOST}:${CS_GATEWAY_PORT} (L1-only, reward-only)"
  if ! command -v clawsentry >/dev/null 2>&1; then
    log "WARN: 'clawsentry' CLI not found in PATH; safety reward will fail-open to 0"
  else
    (
      CS_HTTP_HOST="${CS_HTTP_HOST}" \
      CS_HTTP_PORT="${CS_GATEWAY_PORT}" \
      CS_AUTH_TOKEN="${CS_AUTH_TOKEN}" \
      CS_TRAJECTORY_DB_PATH="${CS_TRAJECTORY_DB_PATH}" \
      CS_LLM_PROVIDER="${CS_LLM_PROVIDER}" \
      CS_L3_ENABLED="${CS_L3_ENABLED}" \
      CS_EVOLVING_ENABLED="${CS_EVOLVING_ENABLED}" \
      clawsentry gateway \
        --gateway-host "${CS_HTTP_HOST}" \
        --gateway-port "${CS_GATEWAY_PORT}" \
        > "${CS_GATEWAY_LOG}" 2>&1 &
      echo $! > "${RUN_LOG_DIR}/clawsentry_gateway.pid"
    )
    CS_GATEWAY_PID="$(cat "${RUN_LOG_DIR}/clawsentry_gateway.pid" 2>/dev/null || echo '')"
    log "ClawSentry gateway PID=${CS_GATEWAY_PID}, log=${CS_GATEWAY_LOG}"

    CS_OK=0
    for ((i=1; i<=20; i++)); do
      if curl -fsS --max-time 2 --noproxy '*' "${CS_HTTP_URL}/health" >/dev/null 2>&1; then
        log "clawsentry-gateway ready (attempt ${i})"
        CS_OK=1
        break
      fi
      sleep 1
    done
    if [[ "${CS_OK}" != "1" ]]; then
      log "WARN: clawsentry-gateway not healthy at ${CS_HTTP_URL}/health; safety reward will fail-open to 0"
    fi
  fi
fi

# Pre-flight: sanity check each pool worker before launching training
# (issue #3 §1.X-E: early detection of worker transport flakes).
if [[ "${NEEDS_ENV_ROUTER}" == "1" ]]; then
  log "Probing worker endpoints..."
  IFS=',' read -r -a _WORKERS <<< "${WORKER_URLS}"
  READY_WORKERS=0
  for _w in "${_WORKERS[@]}"; do
    if probe_ready_endpoint "${_w}" "${_w}" "${WORKER_PREFLIGHT_TIMEOUT}"; then
      READY_WORKERS=$((READY_WORKERS + 1))
    elif [[ "${AUTO_CLOSE_STALE_WORKER_RUNS}" == "1" ]]; then
      close_stale_worker_runs "${_w}" "${_w} (worker_preflight)" "${STALE_WORKER_CLOSE_TIMEOUT}" || true
    fi
  done
  log "Worker readiness: ${READY_WORKERS}/${#_WORKERS[@]} ready"
  if [[ "${READY_WORKERS}" -le 0 && "${WORKER_PREFLIGHT_REQUIRE_READY}" == "1" ]]; then
    log "ERROR: no ready docker env worker; aborting before Ray job submit"
    exit 1
  fi
fi

# ── NVLink detection ─────────────────────────────────────────────────
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "${NVLINK_COUNT:-0}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
log "HAS_NVLINK=${HAS_NVLINK}"

# ── Dump run config ──────────────────────────────────────────────────
cat > "${RUN_DIR}/config/run_config.json" <<CFGEOF
{
  "run_name": "${RUN_NAME}",
  "timestamp": "${RUN_TIMESTAMP}",
  "run_dataset_tag": "${RUN_DATASET_TAG}",
  "run_algo_tag": "${RUN_ALGO_TAG}",
  "debug_mode": ${DEBUG_MODE},
  "dry_run": "${DRY_RUN}",
  "algo": "${ALGO}",
  "harness_option": "${HARNESS_OPTION}",
  "model": "Qwen3-8B",
  "hf_ckpt": "${HF_CKPT}",
  "ref_load": "${REF_LOAD}",
  "save_ckpt": "${SAVE_CKPT}",
  "num_gpus": ${NUM_GPUS},
  "actor_gpus": ${ACTOR_GPUS},
  "rollout_gpus": ${ROLLOUT_GPUS},
  "tp_size": ${TP_SIZE},
  "rollout_engine_gpus": ${ROLLOUT_NUM_GPUS_PER_ENGINE},
  "dataset": "${DATASET}",
  "includes_seta": "${INCLUDES_SETA}",
  "includes_safety": "${INCLUDES_SAFETY}",
  "includes_agentharm": "${INCLUDES_AGENTHARM}",
  "prompt_data": "${ROLLOUT_PROMPT_DATA}",
  "mix_mode": "${MIX_MODE}",
  "num_rollout": ${NUM_ROLLOUT},
  "rollout_batch_size": ${ROLLOUT_BATCH_SIZE},
  "n_samples": ${N_SAMPLES},
  "rollout_max_response_len": ${ROLLOUT_MAX_RESPONSE_LEN},
  "rollout_max_context_len": ${ROLLOUT_MAX_CONTEXT_LEN},
  "rollout_generation_max_retries": "${ROLLOUT_GENERATION_MAX_RETRIES}",
  "rollout_generation_retry_initial_backoff": "${ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF}",
  "rollout_generation_retry_max_backoff": "${ROLLOUT_GENERATION_RETRY_MAX_BACKOFF}",
  "rollout_generation_skip_on_failure": "${ROLLOUT_GENERATION_SKIP_ON_FAILURE}",
  "rollout_generation_retry_backoff_multiplier": "${ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER}",
  "rollout_generation_env_storm_max_retries": "${ROLLOUT_GENERATION_ENV_STORM_MAX_RETRIES}",
  "max_tokens_per_gpu": ${MAX_TOKENS_PER_GPU},
  "worker_urls": "${WORKER_URLS}",
  "worker_urls_file": "${WORKER_URLS_FILE}",
  "worker_urls_reload_interval": "${WORKER_URLS_RELOAD_INTERVAL}",
  "env_server_url": "${ENV_SERVER_URL}",
  "needs_env_router": "${NEEDS_ENV_ROUTER}",
  "router_pressure_cooldown": "${ROUTER_PRESSURE_COOLDOWN}",
  "router_require_ready": "${ROUTER_REQUIRE_READY}",
  "router_readyz_worker_timeout": "${ROUTER_READYZ_WORKER_TIMEOUT}",
  "worker_preflight_require_ready": "${WORKER_PREFLIGHT_REQUIRE_READY}",
  "router_ready_wait_forever": "${ROUTER_READY_WAIT_FOREVER}",
  "auto_close_stale_worker_runs": "${AUTO_CLOSE_STALE_WORKER_RUNS}",
  "stale_worker_close_interval": "${STALE_WORKER_CLOSE_INTERVAL}",
  "stale_worker_close_timeout": "${STALE_WORKER_CLOSE_TIMEOUT}",
  "stale_worker_repair_min_age": "${STALE_WORKER_REPAIR_MIN_AGE}",
  "stale_worker_repair_max_repairs": "${STALE_WORKER_REPAIR_MAX_REPAIRS}",
  "env_http_max_retries": "${ENV_HTTP_MAX_RETRIES}",
  "env_allocate_max_retries": "${ENV_ALLOCATE_MAX_RETRIES}",
  "http_retry_log_every_n": "${HTTP_RETRY_LOG_EVERY_N}",
  "http_retry_log_response_chars": "${HTTP_RETRY_LOG_RESPONSE_CHARS}",
  "terminal_rl_generate_failure_traceback": "${TERMINAL_RL_GENERATE_FAILURE_TRACEBACK}",
  "env_reset_http_timeout": "${ENV_RESET_HTTP_TIMEOUT}",
  "env_close_http_timeout": "${ENV_CLOSE_HTTP_TIMEOUT}",
  "env_remote_max_active_tasks": "${ENV_REMOTE_MAX_ACTIVE_TASKS}",
  "env_remote_max_active_runs": "${ENV_REMOTE_MAX_ACTIVE_RUNS}",
  "env_remote_max_runs_per_task": "${ENV_REMOTE_MAX_RUNS_PER_TASK}",
  "agent_safetybench_remote_env": "${AGENT_SAFETYBENCH_REMOTE_ENV}",
  "agentharm_remote_env": "${AGENTHARM_REMOTE_ENV}",
  "safety_reward_enable": "${CLAWSENTRY_NEEDED}",
  "seta_safety": "${SETA_SAFETY}",
  "safety_bench_reward": "${SAFETY_BENCH_REWARD}",
  "agentharm_reward": "${AGENTHARM_REWARD}",
  "agentharm_root": "${AGENTHARM_ROOT}",
  "dapo_eps_clip_low": "${DAPO_EPS_CLIP_LOW}",
  "dapo_eps_clip_high": "${DAPO_EPS_CLIP_HIGH}",
  "dapo_calculate_per_token_loss": "${DAPO_CALCULATE_PER_TOKEN_LOSS}",
  "dapo_dynamic_sampling": "${DAPO_DYNAMIC_SAMPLING}",
  "dapo_dynamic_filter_path": "${DAPO_DYNAMIC_FILTER_PATH}",
  "dapo_over_sampling_batch_size": "${DAPO_OVER_SAMPLING_BATCH_SIZE}",
  "dapo_failed_group_abort_min_groups": "${DAPO_FAILED_GROUP_ABORT_MIN_GROUPS}",
  "dapo_failed_group_abort_ratio": "${DAPO_FAILED_GROUP_ABORT_RATIO}",
  "dapo_grpo_std_normalization": "${DAPO_GRPO_STD_NORMALIZATION}",
  "dapo_use_kl_loss": "${DAPO_USE_KL_LOSS}",
  "dapo_kl_loss_coef": "${DAPO_KL_LOSS_COEF}",
  "dapo_overlong_buffer_enable": "${DAPO_OVERLONG_BUFFER_ENABLE}",
  "dapo_overlong_buffer_len": "${DAPO_OVERLONG_BUFFER_LEN}",
  "dapo_overlong_penalty_factor": "${DAPO_OVERLONG_PENALTY_FACTOR}",
  "safety_reward_coef": "${SAFETY_REWARD_COEF}",
  "safety_reward_summary_weight": "${SAFETY_REWARD_SUMMARY_WEIGHT}",
  "safety_reward_zero_threshold": "${SAFETY_REWARD_ZERO_THRESHOLD}",
  "trajectory_save_interval_env": "${TRAJECTORY_SAVE_INTERVAL}",
  "trajectory_save_interval_seta": "${TRAJECTORY_SAVE_INTERVAL_SETA}",
  "trajectory_save_interval_agent_safetybench": "${TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH}",
  "trajectory_save_interval_agentharm": "${TRAJECTORY_SAVE_INTERVAL_AGENTHARM}",
  "trajectory_save_policy": "${TRAJECTORY_SAVE_POLICY}",
  "trajectory_task_save_interval": "${TRAJECTORY_TASK_SAVE_INTERVAL}",
  "trajectory_task_max_per_step": "${TRAJECTORY_TASK_MAX_PER_STEP}",
  "trajectory_task_max_per_task": "${TRAJECTORY_TASK_MAX_PER_TASK}",
  "trajectory_max_total": "${TRAJECTORY_MAX_TOTAL}",
  "trajectory_save_reward_strata": "${TRAJECTORY_SAVE_REWARD_STRATA}",
  "trajectory_save_log_decisions": "${TRAJECTORY_SAVE_LOG_DECISIONS}",
  "exploration_profile": "${EXPLORATION_PROFILE}",
  "explore_entropy_coef": "${EXPLORE_ENTROPY_COEF}",
  "explore_think_mode": "${EXPLORE_THINK_MODE}",
  "explore_temp_high": "${EXPLORE_TEMP_HIGH}",
  "explore_intrinsic": "${EXPLORE_INTRINSIC}",
  "explore_intrinsic_enabled": "${EXPLORE_INTRINSIC_ENABLED}",
  "explore_intrinsic_coef": "${EXPLORE_INTRINSIC_COEF}",
  "explore_intrinsic_schedule": "${EXPLORE_INTRINSIC_SCHEDULE}",
  "explore_intrinsic_decay_steps": "${EXPLORE_INTRINSIC_DECAY_STEPS}",
  "explore_intrinsic_reducer": "${EXPLORE_INTRINSIC_REDUCER}",
  "explore_intrinsic_granularity": "${EXPLORE_INTRINSIC_GRANULARITY}",
  "explore_intrinsic_scope": "${EXPLORE_INTRINSIC_SCOPE}",
  "explore_score_bonus_components": "${EXPLORE_SCORE_BONUS_COMPONENTS}",
  "explore_safety_filter": "${EXPLORE_SAFETY_FILTER}",
  "explore_safety_filter_enabled": "${EXPLORE_SAFETY_FILTER_ENABLED}",
  "explore_safety_filter_coef": "${EXPLORE_SAFETY_FILTER_COEF}",
  "explore_lprnd": "${EXPLORE_LPRND}",
  "explore_lprnd_enabled": "${EXPLORE_LPRND_ENABLED}",
  "explore_lprnd_coef": "${EXPLORE_LPRND_COEF}",
  "explore_lprnd_schedule": "${EXPLORE_LPRND_SCHEDULE}",
  "explore_lprnd_decay_steps": "${EXPLORE_LPRND_DECAY_STEPS}",
  "explore_lprnd_clip": "${EXPLORE_LPRND_CLIP}",
  "explore_lprnd_warmup": "${EXPLORE_LPRND_WARMUP}",
  "explore_agent57_lite": "${EXPLORE_AGENT57_LITE}",
  "explore_agent57_lite_enabled": "${EXPLORE_AGENT57_LITE_ENABLED}",
  "explore_agent57_k": "${EXPLORE_AGENT57_K}",
  "explore_agent57_arm_betas": "${EXPLORE_AGENT57_ARM_BETAS}",
  "explore_agent57_combine_mode": "${EXPLORE_AGENT57_COMBINE_MODE}",
  "explore_agent57_ngu_mod_clip": "${EXPLORE_AGENT57_NGU_MOD_CLIP}",
  "explore_agent57_ngu_episodic_source": "${EXPLORE_AGENT57_NGU_EPISODIC_SOURCE}",
  "explore_agent57_ngu_episodic_reducer": "${EXPLORE_AGENT57_NGU_EPISODIC_REDUCER}",
  "explore_agent57_ngu_life_mod_mode": "${EXPLORE_AGENT57_NGU_LIFE_MOD_MODE}",
  "explore_agent57_ngu_life_mod_std_clip": "${EXPLORE_AGENT57_NGU_LIFE_MOD_STD_CLIP}",
  "explore_agent57_max_bonus": "${EXPLORE_AGENT57_MAX_BONUS}",
  "explore_agent57_arm_temperatures": "${EXPLORE_AGENT57_ARM_TEMPERATURES}",
  "explore_agent57_arm_temperature_warmup_rollouts": "${EXPLORE_AGENT57_ARM_TEMPERATURE_WARMUP_ROLLOUTS}",
  "explore_agent57_arm_top_ps": "${EXPLORE_AGENT57_ARM_TOP_PS}",
  "explore_agent57_arm_top_ks": "${EXPLORE_AGENT57_ARM_TOP_KS}",
  "explore_agent57_controller": "${EXPLORE_AGENT57_CONTROLLER}",
  "explore_agent57_ucb_c": "${EXPLORE_AGENT57_UCB_C}",
  "explore_agent57_ucb_window": "${EXPLORE_AGENT57_UCB_WINDOW}",
  "explore_agent57_ucb_epsilon": "${EXPLORE_AGENT57_UCB_EPSILON}",
  "explore_agent57_ucb_min_per_arm": "${EXPLORE_AGENT57_UCB_MIN_PER_ARM}",
  "explore_agent57_ucb_value": "${EXPLORE_AGENT57_UCB_VALUE}",
  "explore_agent57_ucb_dataset_aware": "${EXPLORE_AGENT57_UCB_DATASET_AWARE}",
  "explore_agent57_ucb_random_seed": "${EXPLORE_AGENT57_UCB_RANDOM_SEED}",
  "explore_agent57_ucb_seed_salt": "${EXPLORE_AGENT57_UCB_SEED_SALT}",
  "explore_agent57_keep_baseline": "${EXPLORE_AGENT57_KEEP_BASELINE}",
  "episodic_memory_backend": "${EPISODIC_MEMORY_BACKEND}",
  "explore_agent57_episodic_backend": "${EXPLORE_AGENT57_EPISODIC_BACKEND}",
  "explore_agent57_episodic_capacity": "${EXPLORE_AGENT57_EPISODIC_CAPACITY}",
  "explore_agent57_episodic_count_decay": "${EXPLORE_AGENT57_EPISODIC_COUNT_DECAY}",
  "explore_agent57_episodic_clear_on_reset": "${EXPLORE_AGENT57_EPISODIC_CLEAR_ON_RESET}",
  "explore_agent57_episodic_simhash_bits": "${EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS}",
  "explore_agent57_episodic_bucket_capacity": "${EXPLORE_AGENT57_EPISODIC_BUCKET_CAPACITY}",
  "explore_agent57_episodic_k": "${EXPLORE_AGENT57_EPISODIC_K}",
  "explore_agent57_episodic_distance": "${EXPLORE_AGENT57_EPISODIC_DISTANCE}",
  "explore_agent57_episodic_vector_dim": "${EXPLORE_AGENT57_EPISODIC_VECTOR_DIM}",
  "explore_agent57_episodic_random_seed": "${EXPLORE_AGENT57_EPISODIC_RANDOM_SEED}",
  "explore_agent57_episodic_obs_mode": "${EXPLORE_AGENT57_EPISODIC_OBS_MODE}",
  "explore_agent57_episodic_include_turn": "${EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN}",
  "explore_agent57_episodic_multi_probe_radius": "${EXPLORE_AGENT57_EPISODIC_MULTI_PROBE_RADIUS}",
  "explore_agent57_episodic_novelty_floor": "${EXPLORE_AGENT57_EPISODIC_NOVELTY_FLOOR}",
  "explore_agent57_lifelong": "${EXPLORE_AGENT57_LIFELONG}",
  "explore_agent57_lifelong_enabled": "${EXPLORE_AGENT57_LIFELONG_ENABLED}",
  "explore_agent57_lifelong_coef": "${EXPLORE_AGENT57_LIFELONG_COEF}",
  "explore_agent57_lifelong_clip": "${EXPLORE_AGENT57_LIFELONG_CLIP}",
  "explore_agent57_lifelong_warmup": "${EXPLORE_AGENT57_LIFELONG_WARMUP}",
  "explore_agent57_lifelong_count_decay": "${EXPLORE_AGENT57_LIFELONG_COUNT_DECAY}",
  "explore_agent57_lifelong_capacity": "${EXPLORE_AGENT57_LIFELONG_CAPACITY}",
  "explore_agent57_lifelong_backend": "${EXPLORE_AGENT57_LIFELONG_BACKEND}",
  "explore_agent57_lifelong_key_version": "${EXPLORE_AGENT57_LIFELONG_KEY_VERSION}",
  "explore_agent57_lifelong_include_dataset": "${EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET}",
  "explore_agent57_lifelong_include_task": "${EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK}",
  "explore_agent57_lifelong_include_turn": "${EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN}",
  "explore_agent57_lifelong_obs_mode": "${EXPLORE_AGENT57_LIFELONG_OBS_MODE}",
  "explore_agent57_lifelong_hierarchical": "${EXPLORE_AGENT57_LIFELONG_HIERARCHICAL}",
  "explore_agent57_lifelong_task_weight": "${EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT}",
  "explore_agent57_lifelong_skill_weight": "${EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT}",
  "explore_agent57_lifelong_global_weight": "${EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT}",
  "explore_agent57_sqlite_busy_timeout_ms": "${EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS}",
  "explore_agent57_sqlite_wal": "${EXPLORE_AGENT57_SQLITE_WAL}",
  "explore_agent57_trust_gate": "${EXPLORE_AGENT57_TRUST_GATE}",
  "explore_agent57_trust_completed": "${EXPLORE_AGENT57_TRUST_COMPLETED}",
  "explore_agent57_trust_truncated": "${EXPLORE_AGENT57_TRUST_TRUNCATED}",
  "explore_agent57_trust_failed": "${EXPLORE_AGENT57_TRUST_FAILED}",
  "explore_agent57_trust_parse_error": "${EXPLORE_AGENT57_TRUST_PARSE_ERROR}",
  "explore_agent57_trust_warmup": "${EXPLORE_AGENT57_TRUST_WARMUP}",
  "explore_agent57_state_path": "${EXPLORE_AGENT57_STATE_PATH}",
  "explore_agent57_success_threshold": "${EXPLORE_AGENT57_SUCCESS_THRESHOLD}",
  "explore_advantage_bonus": "${EXPLORE_ADVANTAGE_BONUS}",
  "explore_advantage_bonus_enabled": "${EXPLORE_ADVANTAGE_BONUS_ENABLED}",
  "explore_advantage_bonus_mode": "${EXPLORE_ADVANTAGE_BONUS_MODE}",
  "explore_advantage_bonus_components": "${EXPLORE_ADVANTAGE_BONUS_COMPONENTS}",
  "explore_advantage_bonus_coef": "${EXPLORE_ADVANTAGE_BONUS_COEF}",
  "explore_advantage_bonus_clip": "${EXPLORE_ADVANTAGE_BONUS_CLIP}",
  "explore_advantage_intrinsic_key": "${EXPLORE_ADVANTAGE_INTRINSIC_KEY}",
  "explore_advantage_lambda": "${EXPLORE_ADVANTAGE_LAMBDA}",
  "explore_advantage_lambda_schedule": "${EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE}",
  "explore_advantage_lambda_decay_steps": "${EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS}",
  "explore_advantage_arm_weight_mode": "${EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE}",
  "explore_advantage_trust_key": "${EXPLORE_ADVANTAGE_TRUST_KEY}",
  "explore_advantage_truncated_intrinsic_scale": "${EXPLORE_ADVANTAGE_TRUNCATED_INTRINSIC_SCALE}",
  "explore_advantage_failed_intrinsic_scale": "${EXPLORE_ADVANTAGE_FAILED_INTRINSIC_SCALE}",
  "explore_truncation_penalty": "${EXPLORE_TRUNCATION_PENALTY}",
  "explore_advantage_truncation_penalty": "${EXPLORE_ADVANTAGE_TRUNCATION_PENALTY}",
  "slime_skip_zero_trainable_rollout": "${SLIME_SKIP_ZERO_TRAINABLE_ROLLOUT}",
  "slime_skip_zero_trainable_train": "${SLIME_SKIP_ZERO_TRAINABLE_TRAIN}",
  "explore_cde_actor": "${EXPLORE_CDE_ACTOR}",
  "explore_cde_actor_enabled": "${EXPLORE_CDE_ACTOR_ENABLED}",
  "explore_cde_actor_omega": "${EXPLORE_CDE_ACTOR_OMEGA}",
  "explore_cde_actor_kappa": "${EXPLORE_CDE_ACTOR_KAPPA}",
  "explore_cde_actor_alpha": "${EXPLORE_CDE_ACTOR_ALPHA}",
  "explore_cde_actor_reward_gate": "${EXPLORE_CDE_ACTOR_REWARD_GATE}",
  "explore_cde_actor_decay_steps": "${EXPLORE_CDE_ACTOR_DECAY_STEPS}",
  "explore_retry_attempts": "${EXPLORE_RETRY_ATTEMPTS}",
  "explore_retry_traj_gamma": "${EXPLORE_RETRY_TRAJ_GAMMA}",
  "clawsentry_url": "${CS_HTTP_URL}",
  "clawsentry_llm_provider": "${CS_LLM_PROVIDER}",
  "clawsentry_l3_enabled": "${CS_L3_ENABLED}",
  "clawsentry_evolving_enabled": "${CS_EVOLVING_ENABLED}",
  "terminal_structured_metrics": "${TERMINAL_STRUCTURED_METRICS}",
  "terminal_metrics_jsonl": "${TERMINAL_METRICS_JSONL}",
  "terminal_wandb_metric_profile": "${TERMINAL_WANDB_METRIC_PROFILE}",
  "train_python": "${TRAIN_PYTHON}",
  "a3s_code_repo_root": "${A3S_CODE_REPO_ROOT}",
  "a3s_code_config_path": "${A3S_CODE_CONFIG_PATH}",
  "a3s_code_cache_dir": "${A3S_CODE_CACHE_DIR}",
  "a3s_code_workspace_root": "${A3S_CODE_WORKSPACE_ROOT}",
  "a3s_code_max_tool_rounds": "${A3S_CODE_MAX_TOOL_ROUNDS}",
  "a3s_code_tool_timeout_ms": "${A3S_CODE_TOOL_TIMEOUT_MS}",
  "a3s_code_turn_timeout_sec": "${A3S_CODE_TURN_TIMEOUT_SEC}",
  "a3s_code_output_tokens": "${A3S_CODE_OUTPUT_TOKENS}",
  "a3s_code_planning_mode": "${A3S_CODE_PLANNING_MODE}",
  "a3s_code_external_tool_errors_as_results": "${A3S_CODE_EXTERNAL_TOOL_ERRORS_AS_RESULTS}",
  "a3s_code_local_workspace_guard": "${A3S_CODE_LOCAL_WORKSPACE_GUARD}",
  "claude_code_cli": "${CLAUDE_CODE_CLI}",
  "claude_code_llm_backend": "${CLAUDE_CODE_LLM_BACKEND}",
  "claude_code_model": "${CLAUDE_CODE_MODEL}",
  "claude_code_qwen_gateway_model": "${CLAUDE_CODE_QWEN_GATEWAY_MODEL}",
  "claude_code_workspace_root": "${CLAUDE_CODE_WORKSPACE_ROOT}",
  "claude_code_max_tool_rounds": "${CLAUDE_CODE_MAX_TOOL_ROUNDS}",
  "claude_code_tool_timeout_ms": "${CLAUDE_CODE_TOOL_TIMEOUT_MS}",
  "claude_code_turn_timeout_sec": "${CLAUDE_CODE_TURN_TIMEOUT_SEC}",
  "claude_code_output_format": "${CLAUDE_CODE_OUTPUT_FORMAT}",
  "claude_code_permission_mode": "${CLAUDE_CODE_PERMISSION_MODE}",
  "claude_code_allowed_tools": "${CLAUDE_CODE_ALLOWED_TOOLS}",
  "claude_code_mark_non_trainable": "${CLAUDE_CODE_MARK_NON_TRAINABLE}",
  "slime_ray_placement_gpu_probe": "${SLIME_RAY_PLACEMENT_GPU_PROBE}",
  "log_dir": "${RUN_LOG_DIR}"
}
CFGEOF

# ── Start Ray head ───────────────────────────────────────────────────
log "ray start --head ..."
ray start --head \
  --node-ip-address "${NODE_IP}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --temp-dir "${RAY_TMPDIR}"

log "Waiting for Ray dashboard http://${MASTER_ADDR}:8265 ..."
for i in {1..40}; do
  if curl -fsS --max-time 3 "http://${MASTER_ADDR}:8265/api/version" >/dev/null 2>&1; then
    log "Ray dashboard ready (attempt $i)"
    break
  fi
  sleep 3
done

# ── Build runtime env ────────────────────────────────────────────────
# Match v1 (run_swe_rl_8b_remote_1node.sh): only code dirs in PYTHONPATH.
# Do NOT inject conda site-packages — Ray workers use the default python3
# which already has Megatron/TE/sglang installed.
RUNTIME_PYTHONPATH="${MEGATRON_DIR}:${REPO_ROOT}:${SLIME_DIR}:${SCRIPT_DIR}"
if [[ "${HARNESS_OPTION}" == "a3s-code" ]]; then
  RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH}:${REPO_ROOT}/a3s-code-adapter"
fi

RUNTIME_ENV_XTRACE_WAS_ON=0
if [[ "${HARNESS_OPTION}" == "claude-code" && "$-" == *x* ]]; then
  RUNTIME_ENV_XTRACE_WAS_ON=1
  set +x
fi

A3S_RUNTIME_ENV_JSON=""
if [[ "${HARNESS_OPTION}" == "a3s-code" ]]; then
  A3S_RUNTIME_ENV_JSON=",
    \"A3S_CODE_REPO_ROOT\": \"${A3S_CODE_REPO_ROOT}\",
    \"A3S_CODE_CONFIG_PATH\": \"${A3S_CODE_CONFIG_PATH}\",
    \"A3S_CODE_CACHE_DIR\": \"${A3S_CODE_CACHE_DIR}\",
    \"A3S_CODE_WORKSPACE_ROOT\": \"${A3S_CODE_WORKSPACE_ROOT}\",
    \"A3S_CODE_EXTRA_SITE_PACKAGES\": \"${A3S_CODE_EXTRA_SITE_PACKAGES}\",
    \"A3S_CODE_TURN_TIMEOUT_SEC\": \"${A3S_CODE_TURN_TIMEOUT_SEC}\",
    \"A3S_CODE_TOOL_TIMEOUT_MS\": \"${A3S_CODE_TOOL_TIMEOUT_MS}\",
    \"A3S_CODE_MAX_TOOL_ROUNDS\": \"${A3S_CODE_MAX_TOOL_ROUNDS}\",
    \"A3S_CODE_MAX_PARSE_RETRIES\": \"${A3S_CODE_MAX_PARSE_RETRIES}\",
    \"A3S_CODE_OUTPUT_TOKENS\": \"${A3S_CODE_OUTPUT_TOKENS}\",
    \"A3S_CODE_PLANNING_MODE\": \"${A3S_CODE_PLANNING_MODE}\",
    \"A3S_CODE_THINKING_BUDGET\": \"${A3S_CODE_THINKING_BUDGET}\",
    \"A3S_CODE_EXTERNAL_TOOL_ERRORS_AS_RESULTS\": \"${A3S_CODE_EXTERNAL_TOOL_ERRORS_AS_RESULTS}\",
	    \"A3S_CODE_LOCAL_WORKSPACE_GUARD\": \"${A3S_CODE_LOCAL_WORKSPACE_GUARD}\""
fi

CLAUDE_RUNTIME_ENV_JSON=""
if [[ "${HARNESS_OPTION}" == "claude-code" ]]; then
  CLAUDE_RUNTIME_ENV_JSON=",
    \"CLAUDE_CODE_CLI\": \"${CLAUDE_CODE_CLI}\",
    \"CLAUDE_CODE_LLM_BACKEND\": \"${CLAUDE_CODE_LLM_BACKEND}\",
    \"CLAUDE_CODE_MODEL\": \"${CLAUDE_CODE_MODEL}\",
    \"CLAUDE_CODE_QWEN_GATEWAY_MODEL\": \"${CLAUDE_CODE_QWEN_GATEWAY_MODEL}\",
    \"CLAUDE_CODE_WORKSPACE_ROOT\": \"${CLAUDE_CODE_WORKSPACE_ROOT}\",
    \"CLAUDE_CODE_TURN_TIMEOUT_SEC\": \"${CLAUDE_CODE_TURN_TIMEOUT_SEC}\",
    \"CLAUDE_CODE_TOOL_TIMEOUT_MS\": \"${CLAUDE_CODE_TOOL_TIMEOUT_MS}\",
    \"CLAUDE_CODE_MAX_TOOL_ROUNDS\": \"${CLAUDE_CODE_MAX_TOOL_ROUNDS}\",
    \"CLAUDE_CODE_OUTPUT_FORMAT\": \"${CLAUDE_CODE_OUTPUT_FORMAT}\",
    \"CLAUDE_CODE_PERMISSION_MODE\": \"${CLAUDE_CODE_PERMISSION_MODE}\",
    \"CLAUDE_CODE_ALLOWED_TOOLS\": \"${CLAUDE_CODE_ALLOWED_TOOLS}\",
    \"CLAUDE_CODE_DISALLOWED_TOOLS\": \"${CLAUDE_CODE_DISALLOWED_TOOLS}\",
    \"CLAUDE_CODE_EXTRA_ARGS\": \"${CLAUDE_CODE_EXTRA_ARGS}\",
    \"CLAUDE_CODE_SYSTEM_PROMPT\": \"${CLAUDE_CODE_SYSTEM_PROMPT}\",
    \"CLAUDE_CODE_MCP_PYTHON\": \"${CLAUDE_CODE_MCP_PYTHON}\",
    \"CLAUDE_CODE_HTTP_MAX_RETRIES\": \"${CLAUDE_CODE_HTTP_MAX_RETRIES}\",
    \"CLAUDE_CODE_HTTP_RETRY_DELAY\": \"${CLAUDE_CODE_HTTP_RETRY_DELAY}\",
    \"CLAUDE_CODE_MARK_NON_TRAINABLE\": \"${CLAUDE_CODE_MARK_NON_TRAINABLE}\",
    \"ANTHROPIC_API_KEY\": \"${ANTHROPIC_API_KEY}\",
    \"ANTHROPIC_AUTH_TOKEN\": \"${ANTHROPIC_AUTH_TOKEN}\",
    \"ANTHROPIC_BASE_URL\": \"${ANTHROPIC_BASE_URL}\",
    \"ANTHROPIC_API_URL\": \"${ANTHROPIC_API_URL}\""
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"PYTHONPATH\": \"${RUNTIME_PYTHONPATH}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"PYTHONFAULTHANDLER\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SLIME_RAY_PLACEMENT_GPU_PROBE\": \"${SLIME_RAY_PLACEMENT_GPU_PROBE}\",
    \"SLIME_SKIP_ZERO_TRAINABLE_ROLLOUT\": \"${SLIME_SKIP_ZERO_TRAINABLE_ROLLOUT}\",
    \"SLIME_SKIP_ZERO_TRAINABLE_TRAIN\": \"${SLIME_SKIP_ZERO_TRAINABLE_TRAIN}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"USE_REMOTE_ENV\": \"${USE_REMOTE_ENV}\",
    \"ENV_SERVER_URL\": \"${ENV_SERVER_URL}\",
    \"ENV_HTTP_MAX_RETRIES\": \"${ENV_HTTP_MAX_RETRIES}\",
    \"ENV_ALLOCATE_MAX_RETRIES\": \"${ENV_ALLOCATE_MAX_RETRIES}\",
    \"ENV_ALLOCATE_RETRY_BASE_DELAY\": \"${ENV_ALLOCATE_RETRY_BASE_DELAY}\",
    \"ENV_ALLOCATE_RETRY_MAX_DELAY\": \"${ENV_ALLOCATE_RETRY_MAX_DELAY}\",
    \"ENV_ALLOCATE_RETRY_BACKOFF\": \"${ENV_ALLOCATE_RETRY_BACKOFF}\",
    \"ENV_ALLOCATE_RETRY_JITTER\": \"${ENV_ALLOCATE_RETRY_JITTER}\",
    \"HTTP_RETRY_LOG_EVERY_N\": \"${HTTP_RETRY_LOG_EVERY_N}\",
    \"HTTP_RETRY_LOG_RESPONSE_CHARS\": \"${HTTP_RETRY_LOG_RESPONSE_CHARS}\",
    \"TERMINAL_RL_GENERATE_FAILURE_TRACEBACK\": \"${TERMINAL_RL_GENERATE_FAILURE_TRACEBACK}\",
    \"ENV_EVALUATE_MAX_RETRIES\": \"${ENV_EVALUATE_MAX_RETRIES}\",
    \"ENV_CLOSE_MAX_RETRIES\": \"${ENV_CLOSE_MAX_RETRIES}\",
    \"ENV_EXEC_TOOL_MAX_RETRIES\": \"${ENV_EXEC_TOOL_MAX_RETRIES}\",
    \"ENV_ALLOCATE_HTTP_TIMEOUT\": \"${ENV_ALLOCATE_HTTP_TIMEOUT}\",
    \"ENV_RESET_HTTP_TIMEOUT\": \"${ENV_RESET_HTTP_TIMEOUT}\",
    \"ENV_CLOSE_HTTP_TIMEOUT\": \"${ENV_CLOSE_HTTP_TIMEOUT}\",
    \"ENV_REMOTE_MAX_ACTIVE_TASKS\": \"${ENV_REMOTE_MAX_ACTIVE_TASKS}\",
    \"ENV_REMOTE_MAX_ACTIVE_RUNS\": \"${ENV_REMOTE_MAX_ACTIVE_RUNS}\",
    \"ENV_REMOTE_MAX_RUNS_PER_TASK\": \"${ENV_REMOTE_MAX_RUNS_PER_TASK}\",
    \"ENV_REMOTE_ADMISSION_TIMEOUT\": \"${ENV_REMOTE_ADMISSION_TIMEOUT}\",
    \"ENV_REMOTE_ADMISSION_LOG_INTERVAL\": \"${ENV_REMOTE_ADMISSION_LOG_INTERVAL}\",
    \"ENV_REMOTE_MAX_CONCURRENT_CLOSES\": \"${ENV_REMOTE_MAX_CONCURRENT_CLOSES}\",
    \"AGENT_SAFETYBENCH_REMOTE_ENV\": \"${AGENT_SAFETYBENCH_REMOTE_ENV}\",
    \"AGENTHARM_REMOTE_ENV\": \"${AGENTHARM_REMOTE_ENV}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${NO_PROXY}\",
    \"CS_HTTP_URL\": \"${CS_HTTP_URL}\",
    \"CS_AUTH_TOKEN\": \"${CS_AUTH_TOKEN}\",
    \"SETA_SAFETY\": \"${SETA_SAFETY}\",
    \"SAFETY_BENCH_REWARD\": \"${SAFETY_BENCH_REWARD}\",
    \"AGENT_SAFETYBENCH_ROOT\": \"${AGENT_SAFETYBENCH_ROOT}\",
    \"AGENTHARM_REWARD\": \"${AGENTHARM_REWARD}\",
    \"AGENTHARM_ROOT\": \"${AGENTHARM_ROOT}\",
    \"SAFETY_REWARD_COEF\": \"${SAFETY_REWARD_COEF}\",
    \"SAFETY_REWARD_SUMMARY_WEIGHT\": \"${SAFETY_REWARD_SUMMARY_WEIGHT}\",
    \"SAFETY_REWARD_TIMEOUT\": \"${SAFETY_REWARD_TIMEOUT}\",
    \"SAFETY_REWARD_ZERO_THRESHOLD\": \"${SAFETY_REWARD_ZERO_THRESHOLD}\",
    \"TERMINAL_SAVE_TRAJ_DIR\": \"${TERMINAL_SAVE_TRAJ_DIR}\",
    \"TRAJECTORY_SAVE_INTERVAL\": \"${TRAJECTORY_SAVE_INTERVAL}\",
    \"TRAJECTORY_SAVE_INTERVAL_SETA\": \"${TRAJECTORY_SAVE_INTERVAL_SETA}\",
    \"TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH\": \"${TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH}\",
    \"TRAJECTORY_SAVE_INTERVAL_AGENTHARM\": \"${TRAJECTORY_SAVE_INTERVAL_AGENTHARM}\",
    \"TRAJECTORY_SAVE_POLICY\": \"${TRAJECTORY_SAVE_POLICY}\",
    \"TRAJECTORY_TASK_SAVE_INTERVAL\": \"${TRAJECTORY_TASK_SAVE_INTERVAL}\",
    \"TRAJECTORY_TASK_MAX_PER_STEP\": \"${TRAJECTORY_TASK_MAX_PER_STEP}\",
    \"TRAJECTORY_TASK_MAX_PER_TASK\": \"${TRAJECTORY_TASK_MAX_PER_TASK}\",
    \"TRAJECTORY_MAX_TOTAL\": \"${TRAJECTORY_MAX_TOTAL}\",
    \"TRAJECTORY_SAVE_REWARD_STRATA\": \"${TRAJECTORY_SAVE_REWARD_STRATA}\",
    \"TRAJECTORY_SAVE_LOG_DECISIONS\": \"${TRAJECTORY_SAVE_LOG_DECISIONS}\",
    \"MIX_MODE\": \"${MIX_MODE}\",
    \"RUN_DIR\": \"${RUN_DIR}\",
    \"RUN_ID\": \"${RUN_ID}\",
    \"RUN_NAME\": \"${RUN_NAME}\",
    \"RUN_LOG_DIR\": \"${RUN_LOG_DIR}\",
    \"TERMINAL_STRUCTURED_METRICS\": \"${TERMINAL_STRUCTURED_METRICS}\",
    \"TERMINAL_METRICS_JSONL\": \"${TERMINAL_METRICS_JSONL}\",
    \"TERMINAL_WANDB_METRIC_PROFILE\": \"${TERMINAL_WANDB_METRIC_PROFILE}\",
    \"HARNESS_OPTION\": \"${HARNESS_OPTION}\",
    \"DATASET\": \"${DATASET}\",
    \"ALGO\": \"${ALGO}\",
    \"DAPO_OVERLONG_BUFFER_ENABLE\": \"${DAPO_OVERLONG_BUFFER_ENABLE}\",
    \"DAPO_OVERLONG_BUFFER_LEN\": \"${DAPO_OVERLONG_BUFFER_LEN}\",
    \"DAPO_OVERLONG_PENALTY_FACTOR\": \"${DAPO_OVERLONG_PENALTY_FACTOR}\",
    \"DAPO_MAX_RESPONSE_LEN\": \"${ROLLOUT_MAX_RESPONSE_LEN}\",
    \"EXPLORATION_PROFILE\": \"${EXPLORATION_PROFILE}\",
    \"EXPLORE_ENTROPY_COEF\": \"${EXPLORE_ENTROPY_COEF}\",
    \"EXPLORE_THINK_MODE\": \"${EXPLORE_THINK_MODE}\",
    \"EXPLORE_TEMP_HIGH\": \"${EXPLORE_TEMP_HIGH}\",
    \"EXPLORE_INTRINSIC\": \"${EXPLORE_INTRINSIC}\",
    \"EXPLORE_INTRINSIC_ENABLED\": \"${EXPLORE_INTRINSIC_ENABLED}\",
    \"EXPLORE_INTRINSIC_COEF\": \"${EXPLORE_INTRINSIC_COEF}\",
    \"EXPLORE_INTRINSIC_SCHEDULE\": \"${EXPLORE_INTRINSIC_SCHEDULE}\",
    \"EXPLORE_INTRINSIC_DECAY_STEPS\": \"${EXPLORE_INTRINSIC_DECAY_STEPS}\",
    \"EXPLORE_INTRINSIC_REDUCER\": \"${EXPLORE_INTRINSIC_REDUCER}\",
    \"EXPLORE_INTRINSIC_GRANULARITY\": \"${EXPLORE_INTRINSIC_GRANULARITY}\",
    \"EXPLORE_INTRINSIC_SCOPE\": \"${EXPLORE_INTRINSIC_SCOPE}\",
    \"EXPLORE_SCORE_BONUS_COMPONENTS\": \"${EXPLORE_SCORE_BONUS_COMPONENTS}\",
    \"EXPLORE_SAFETY_FILTER\": \"${EXPLORE_SAFETY_FILTER}\",
    \"EXPLORE_SAFETY_FILTER_ENABLED\": \"${EXPLORE_SAFETY_FILTER_ENABLED}\",
    \"EXPLORE_SAFETY_FILTER_COEF\": \"${EXPLORE_SAFETY_FILTER_COEF}\",
    \"EXPLORE_LPRND\": \"${EXPLORE_LPRND}\",
    \"EXPLORE_LPRND_ENABLED\": \"${EXPLORE_LPRND_ENABLED}\",
    \"EXPLORE_LPRND_COEF\": \"${EXPLORE_LPRND_COEF}\",
    \"EXPLORE_LPRND_SCHEDULE\": \"${EXPLORE_LPRND_SCHEDULE}\",
    \"EXPLORE_LPRND_DECAY_STEPS\": \"${EXPLORE_LPRND_DECAY_STEPS}\",
    \"EXPLORE_LPRND_CLIP\": \"${EXPLORE_LPRND_CLIP}\",
    \"EXPLORE_LPRND_WARMUP\": \"${EXPLORE_LPRND_WARMUP}\",
    \"EXPLORE_AGENT57_LITE\": \"${EXPLORE_AGENT57_LITE}\",
    \"EXPLORE_AGENT57_LITE_ENABLED\": \"${EXPLORE_AGENT57_LITE_ENABLED}\",
    \"EXPLORE_AGENT57_K\": \"${EXPLORE_AGENT57_K}\",
    \"EXPLORE_AGENT57_ARM_BETAS\": \"${EXPLORE_AGENT57_ARM_BETAS}\",
    \"EXPLORE_AGENT57_COMBINE_MODE\": \"${EXPLORE_AGENT57_COMBINE_MODE}\",
    \"EXPLORE_AGENT57_NGU_MOD_CLIP\": \"${EXPLORE_AGENT57_NGU_MOD_CLIP}\",
    \"EXPLORE_AGENT57_NGU_EPISODIC_SOURCE\": \"${EXPLORE_AGENT57_NGU_EPISODIC_SOURCE}\",
    \"EXPLORE_AGENT57_NGU_EPISODIC_REDUCER\": \"${EXPLORE_AGENT57_NGU_EPISODIC_REDUCER}\",
    \"EXPLORE_AGENT57_NGU_LIFE_MOD_MODE\": \"${EXPLORE_AGENT57_NGU_LIFE_MOD_MODE}\",
    \"EXPLORE_AGENT57_NGU_LIFE_MOD_STD_CLIP\": \"${EXPLORE_AGENT57_NGU_LIFE_MOD_STD_CLIP}\",
    \"EXPLORE_AGENT57_MAX_BONUS\": \"${EXPLORE_AGENT57_MAX_BONUS}\",
    \"EXPLORE_AGENT57_ARM_TEMPERATURES\": \"${EXPLORE_AGENT57_ARM_TEMPERATURES}\",
    \"EXPLORE_AGENT57_ARM_TEMPERATURE_WARMUP_ROLLOUTS\": \"${EXPLORE_AGENT57_ARM_TEMPERATURE_WARMUP_ROLLOUTS}\",
    \"EXPLORE_AGENT57_ARM_TOP_PS\": \"${EXPLORE_AGENT57_ARM_TOP_PS}\",
    \"EXPLORE_AGENT57_ARM_TOP_KS\": \"${EXPLORE_AGENT57_ARM_TOP_KS}\",
    \"EXPLORE_AGENT57_CONTROLLER\": \"${EXPLORE_AGENT57_CONTROLLER}\",
    \"EXPLORE_AGENT57_UCB_C\": \"${EXPLORE_AGENT57_UCB_C}\",
    \"EXPLORE_AGENT57_UCB_WINDOW\": \"${EXPLORE_AGENT57_UCB_WINDOW}\",
    \"EXPLORE_AGENT57_UCB_EPSILON\": \"${EXPLORE_AGENT57_UCB_EPSILON}\",
    \"EXPLORE_AGENT57_UCB_MIN_PER_ARM\": \"${EXPLORE_AGENT57_UCB_MIN_PER_ARM}\",
    \"EXPLORE_AGENT57_UCB_VALUE\": \"${EXPLORE_AGENT57_UCB_VALUE}\",
    \"EXPLORE_AGENT57_UCB_DATASET_AWARE\": \"${EXPLORE_AGENT57_UCB_DATASET_AWARE}\",
    \"EXPLORE_AGENT57_UCB_RANDOM_SEED\": \"${EXPLORE_AGENT57_UCB_RANDOM_SEED}\",
    \"EXPLORE_AGENT57_UCB_SEED_SALT\": \"${EXPLORE_AGENT57_UCB_SEED_SALT}\",
    \"EXPLORE_AGENT57_KEEP_BASELINE\": \"${EXPLORE_AGENT57_KEEP_BASELINE}\",
    \"EPISODIC_MEMORY_BACKEND\": \"${EPISODIC_MEMORY_BACKEND}\",
    \"EXPLORE_AGENT57_EPISODIC_BACKEND\": \"${EXPLORE_AGENT57_EPISODIC_BACKEND}\",
    \"EXPLORE_AGENT57_EPISODIC_CAPACITY\": \"${EXPLORE_AGENT57_EPISODIC_CAPACITY}\",
    \"EXPLORE_AGENT57_EPISODIC_COUNT_DECAY\": \"${EXPLORE_AGENT57_EPISODIC_COUNT_DECAY}\",
    \"EXPLORE_AGENT57_EPISODIC_CLEAR_ON_RESET\": \"${EXPLORE_AGENT57_EPISODIC_CLEAR_ON_RESET}\",
    \"EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS\": \"${EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS}\",
    \"EXPLORE_AGENT57_EPISODIC_BUCKET_CAPACITY\": \"${EXPLORE_AGENT57_EPISODIC_BUCKET_CAPACITY}\",
    \"EXPLORE_AGENT57_EPISODIC_K\": \"${EXPLORE_AGENT57_EPISODIC_K}\",
    \"EXPLORE_AGENT57_EPISODIC_DISTANCE\": \"${EXPLORE_AGENT57_EPISODIC_DISTANCE}\",
    \"EXPLORE_AGENT57_EPISODIC_VECTOR_DIM\": \"${EXPLORE_AGENT57_EPISODIC_VECTOR_DIM}\",
    \"EXPLORE_AGENT57_EPISODIC_RANDOM_SEED\": \"${EXPLORE_AGENT57_EPISODIC_RANDOM_SEED}\",
    \"EXPLORE_AGENT57_EPISODIC_OBS_MODE\": \"${EXPLORE_AGENT57_EPISODIC_OBS_MODE}\",
    \"EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN\": \"${EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN}\",
    \"EXPLORE_AGENT57_EPISODIC_MULTI_PROBE_RADIUS\": \"${EXPLORE_AGENT57_EPISODIC_MULTI_PROBE_RADIUS}\",
    \"EXPLORE_AGENT57_EPISODIC_NOVELTY_FLOOR\": \"${EXPLORE_AGENT57_EPISODIC_NOVELTY_FLOOR}\",
    \"EXPLORE_AGENT57_LIFELONG\": \"${EXPLORE_AGENT57_LIFELONG}\",
    \"EXPLORE_AGENT57_LIFELONG_ENABLED\": \"${EXPLORE_AGENT57_LIFELONG_ENABLED}\",
    \"EXPLORE_AGENT57_LIFELONG_COEF\": \"${EXPLORE_AGENT57_LIFELONG_COEF}\",
    \"EXPLORE_AGENT57_LIFELONG_CLIP\": \"${EXPLORE_AGENT57_LIFELONG_CLIP}\",
    \"EXPLORE_AGENT57_LIFELONG_WARMUP\": \"${EXPLORE_AGENT57_LIFELONG_WARMUP}\",
    \"EXPLORE_AGENT57_LIFELONG_COUNT_DECAY\": \"${EXPLORE_AGENT57_LIFELONG_COUNT_DECAY}\",
    \"EXPLORE_AGENT57_LIFELONG_CAPACITY\": \"${EXPLORE_AGENT57_LIFELONG_CAPACITY}\",
    \"EXPLORE_AGENT57_LIFELONG_BACKEND\": \"${EXPLORE_AGENT57_LIFELONG_BACKEND}\",
    \"EXPLORE_AGENT57_LIFELONG_KEY_VERSION\": \"${EXPLORE_AGENT57_LIFELONG_KEY_VERSION}\",
    \"EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET\": \"${EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET}\",
    \"EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK\": \"${EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK}\",
    \"EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN\": \"${EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN}\",
    \"EXPLORE_AGENT57_LIFELONG_OBS_MODE\": \"${EXPLORE_AGENT57_LIFELONG_OBS_MODE}\",
    \"EXPLORE_AGENT57_LIFELONG_HIERARCHICAL\": \"${EXPLORE_AGENT57_LIFELONG_HIERARCHICAL}\",
    \"EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT\": \"${EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT}\",
    \"EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT\": \"${EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT}\",
    \"EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT\": \"${EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT}\",
    \"EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS\": \"${EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS}\",
    \"EXPLORE_AGENT57_SQLITE_WAL\": \"${EXPLORE_AGENT57_SQLITE_WAL}\",
    \"EXPLORE_AGENT57_TRUST_GATE\": \"${EXPLORE_AGENT57_TRUST_GATE}\",
    \"EXPLORE_AGENT57_TRUST_COMPLETED\": \"${EXPLORE_AGENT57_TRUST_COMPLETED}\",
    \"EXPLORE_AGENT57_TRUST_TRUNCATED\": \"${EXPLORE_AGENT57_TRUST_TRUNCATED}\",
    \"EXPLORE_AGENT57_TRUST_FAILED\": \"${EXPLORE_AGENT57_TRUST_FAILED}\",
    \"EXPLORE_AGENT57_TRUST_PARSE_ERROR\": \"${EXPLORE_AGENT57_TRUST_PARSE_ERROR}\",
    \"EXPLORE_AGENT57_TRUST_WARMUP\": \"${EXPLORE_AGENT57_TRUST_WARMUP}\",
    \"EXPLORE_AGENT57_STATE_PATH\": \"${EXPLORE_AGENT57_STATE_PATH}\",
    \"EXPLORE_AGENT57_SUCCESS_THRESHOLD\": \"${EXPLORE_AGENT57_SUCCESS_THRESHOLD}\",
    \"EXPLORE_ADVANTAGE_BONUS\": \"${EXPLORE_ADVANTAGE_BONUS}\",
    \"EXPLORE_ADVANTAGE_BONUS_ENABLED\": \"${EXPLORE_ADVANTAGE_BONUS_ENABLED}\",
    \"EXPLORE_ADVANTAGE_BONUS_MODE\": \"${EXPLORE_ADVANTAGE_BONUS_MODE}\",
    \"EXPLORE_ADVANTAGE_BONUS_COMPONENTS\": \"${EXPLORE_ADVANTAGE_BONUS_COMPONENTS}\",
    \"EXPLORE_ADVANTAGE_BONUS_COEF\": \"${EXPLORE_ADVANTAGE_BONUS_COEF}\",
    \"EXPLORE_ADVANTAGE_BONUS_CLIP\": \"${EXPLORE_ADVANTAGE_BONUS_CLIP}\",
    \"EXPLORE_ADVANTAGE_INTRINSIC_KEY\": \"${EXPLORE_ADVANTAGE_INTRINSIC_KEY}\",
    \"EXPLORE_ADVANTAGE_LAMBDA\": \"${EXPLORE_ADVANTAGE_LAMBDA}\",
    \"EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE\": \"${EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE}\",
    \"EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS\": \"${EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS}\",
    \"EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE\": \"${EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE}\",
    \"EXPLORE_ADVANTAGE_TRUST_KEY\": \"${EXPLORE_ADVANTAGE_TRUST_KEY}\",
    \"EXPLORE_ADVANTAGE_TRUNCATED_INTRINSIC_SCALE\": \"${EXPLORE_ADVANTAGE_TRUNCATED_INTRINSIC_SCALE}\",
    \"EXPLORE_ADVANTAGE_FAILED_INTRINSIC_SCALE\": \"${EXPLORE_ADVANTAGE_FAILED_INTRINSIC_SCALE}\",
    \"EXPLORE_TRUNCATION_PENALTY\": \"${EXPLORE_TRUNCATION_PENALTY}\",
    \"EXPLORE_ADVANTAGE_TRUNCATION_PENALTY\": \"${EXPLORE_ADVANTAGE_TRUNCATION_PENALTY}\",
    \"EXPLORE_CDE_ACTOR\": \"${EXPLORE_CDE_ACTOR}\",
    \"EXPLORE_CDE_ACTOR_ENABLED\": \"${EXPLORE_CDE_ACTOR_ENABLED}\",
    \"EXPLORE_CDE_ACTOR_OMEGA\": \"${EXPLORE_CDE_ACTOR_OMEGA}\",
    \"EXPLORE_CDE_ACTOR_KAPPA\": \"${EXPLORE_CDE_ACTOR_KAPPA}\",
    \"EXPLORE_CDE_ACTOR_ALPHA\": \"${EXPLORE_CDE_ACTOR_ALPHA}\",
    \"EXPLORE_CDE_ACTOR_REWARD_GATE\": \"${EXPLORE_CDE_ACTOR_REWARD_GATE}\",
    \"EXPLORE_CDE_ACTOR_DECAY_STEPS\": \"${EXPLORE_CDE_ACTOR_DECAY_STEPS}\",
    \"EXPLORE_RETRY_ATTEMPTS\": \"${EXPLORE_RETRY_ATTEMPTS}\",
    \"EXPLORE_RETRY_TRAJ_GAMMA\": \"${EXPLORE_RETRY_TRAJ_GAMMA}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-offline}\"
    ${A3S_RUNTIME_ENV_JSON}
    ${CLAUDE_RUNTIME_ENV_JSON}
  }
}"

if [[ "${RUNTIME_ENV_XTRACE_WAS_ON}" == "1" ]]; then
  set -x
fi

RAY_JOB_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-terminal_rl_8b_${NUM_GPUS}gpu_$(date +%Y%m%d_%H%M%S)}"
CASE_STUDY_ON_EXIT="${CASE_STUDY_ON_EXIT:-0}"
CASE_STUDY_ON_FAILURE="${CASE_STUDY_ON_FAILURE:-0}"
CASE_STUDY_CONFIG="${CASE_STUDY_CONFIG:-${SCRIPT_DIR}/scripts/case_study_samples.yaml}"

run_case_study_if_requested() {
  local phase="$1"
  if [[ "${CASE_STUDY_ON_EXIT}" != "1" ]]; then
    return 0
  fi
  if [[ "${phase}" != "success" && "${CASE_STUDY_ON_FAILURE}" != "1" ]]; then
    return 0
  fi
  if [[ ! -d "${RUN_DIR}/trajectories" ]]; then
    log "Case-study skipped: ${RUN_DIR}/trajectories not found"
    return 0
  fi
  log "Running case-study analysis (${phase})"
  if ! CASE_STUDY_CONFIG="${CASE_STUDY_CONFIG}" \
       bash "${SCRIPT_DIR}/scripts/run_case_study.sh" "${RUN_DIR}"; then
    log "Case-study analysis failed; training exit status is unchanged"
  fi
}

log "Submitting Ray job ${RAY_JOB_SUBMISSION_ID}"
RAY_SUBMIT_XTRACE_WAS_ON=0
if [[ "${HARNESS_OPTION}" == "claude-code" && "$-" == *x* ]]; then
  RAY_SUBMIT_XTRACE_WAS_ON=1
  set +x
fi
ray job submit --address="http://${MASTER_ADDR}:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_PYTHON}" -u "${SLIME_DIR}/train_async.py" \
  "${TRAIN_ARGS[@]}"
if [[ "${RAY_SUBMIT_XTRACE_WAS_ON}" == "1" ]]; then
  set -x
fi

set +e
ray job logs --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record
RAY_LOG_EXIT=$?
RAY_STATUS_OUTPUT=$(ray job status --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)
echo "${RAY_STATUS_OUTPUT}"
set -e

# ── Checkpoint cleanup: keep only the latest MAX_CKPT_KEEP per-run ───
# (issue #3 §3: ckpt accumulation is the #1 cause of ENOSPC on shared FS.)
CKPT_DIR="${SAVE_CKPT}"
if [[ -d "${CKPT_DIR}" ]] && (( MAX_CKPT_KEEP > 0 )); then
  CKPT_DIRS=($(ls -1d "${CKPT_DIR}"/iter_* 2>/dev/null | sort -t_ -k2 -n))
  TOTAL_CKPTS=${#CKPT_DIRS[@]}
  if (( TOTAL_CKPTS > MAX_CKPT_KEEP )); then
    NUM_TO_DELETE=$(( TOTAL_CKPTS - MAX_CKPT_KEEP ))
    log "Ckpt cleanup: found ${TOTAL_CKPTS}, keeping ${MAX_CKPT_KEEP}, deleting ${NUM_TO_DELETE}"
    for (( i=0; i<NUM_TO_DELETE; i++ )); do
      log "  Removing: ${CKPT_DIRS[$i]}"
      rm -rf "${CKPT_DIRS[$i]}"
    done
  fi
fi

RAY_STATUS_LOWER=$(echo "${RAY_STATUS_OUTPUT}" | tr '[:upper:]' '[:lower:]')
if [[ "${RAY_STATUS_LOWER}" == *"succeeded"* ]]; then
  run_case_study_if_requested success
  log "Ray job succeeded"
  exit 0
fi

# ── Failure auto-capture ─────────────────────────────────────────────
# Generate two condensed artifacts under the run-contained log mirror:
#   gpu_tail.log : last 300 lines (often enough to see the actual stack)
#   gpu_err.log  : grep-filtered "real" error lines (CUDA/OOM/Exception/etc.)
log "Ray job failed (logs exit: ${RAY_LOG_EXIT}). Writing condensed artifacts..."
tail -n 300 "${RUN_LOG}" > "${GPU_TAIL_LOG}" 2>/dev/null || true
grep -E "Error|Exception|Traceback|CUDA|OOM|invalid device|FAILED|CheckpointException|ENOSPC|PermissionError|Connect call failed|ConnectorError|500|502" \
     "${RUN_LOG}" 2>/dev/null \
  | grep -v "FutureWarning" \
  | grep -v "DeprecationWarning" \
  | tail -n 200 \
  > "${GPU_ERR_LOG}" 2>/dev/null || true

cat <<EOF
========================================
  Run failed. Inspect:
    full   : ${GPU_RUN_LOG}
    errors : ${GPU_ERR_LOG}
    tail   : ${GPU_TAIL_LOG}
    latest : ${TMP_DOC_LATEST}/
========================================
EOF
run_case_study_if_requested failure
exit 1
