#!/usr/bin/env bash
# Full SWE-bench Verified evaluation for the pre-RL Qwen3-8B checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TERMINAL_RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TERMINAL_RL_DIR}/.." && pwd)"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[ERROR] ${name} is required." >&2
    exit 2
  fi
}

pin_formal_env() {
  local name="$1"
  local expected="$2"
  if [[ -n "${!name:-}" && "${!name}" != "${expected}" ]]; then
    echo "[ERROR] Formal profile requires ${name}=${expected}; got ${!name}." >&2
    exit 2
  fi
  export "${name}=${expected}"
}

if [[ -n "${EVAL_LIMIT:-}" ]]; then
  echo "[ERROR] This formal launcher requires all 500 SWE-bench Verified instances; unset EVAL_LIMIT." >&2
  exit 2
fi
if [[ -n "${SWEBENCH_EXPECTED_INSTANCES:-}" && "${SWEBENCH_EXPECTED_INSTANCES}" != "500" ]]; then
  echo "[ERROR] Formal SWE-bench Verified instance count is fixed at 500." >&2
  exit 2
fi
if [[ -n "${SWEBENCH_EXPECTED_DATASET_SHA256:-}" && "${SWEBENCH_EXPECTED_DATASET_SHA256}" != "4282529dbcc1b9253fa91da35b9f1768a2002b391cc90ac6a4e64575d59cfbf3" ]]; then
  echo "[ERROR] Formal SWE-bench Verified dataset SHA256 is fixed to the official converted dataset." >&2
  exit 2
fi
if [[ -n "${SWEBENCH_EXPECTED_OFFICIAL_DATASET_SHA256:-}" && "${SWEBENCH_EXPECTED_OFFICIAL_DATASET_SHA256}" != "f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c" ]]; then
  echo "[ERROR] Formal official SWE-bench dataset SHA256 is fixed." >&2
  exit 2
fi
if [[ -n "${SWEBENCH_DEFER_GRADING:-}" && "${SWEBENCH_DEFER_GRADING}" != "1" ]]; then
  echo "[ERROR] Formal SWE-bench Verified uses prediction-only generation; SWEBENCH_DEFER_GRADING must be 1." >&2
  exit 2
fi

require_env WORKER_URLS
require_env HF_CKPT
require_env REF_LOAD
TRAIN_PYTHON="${TRAIN_PYTHON:-$(command -v python3 || true)}"
require_env TRAIN_PYTHON

export WORKER_URLS
export TRAIN_PYTHON
export EVAL_SUITE=sweverified
export EVAL_CKPT=init
export FORMAL_SWEBENCH_VERIFIED=1
export SWEBENCH_DEFER_GRADING=1
export SWEBENCH_EXPECTED_INSTANCES=500
export SWEBENCH_EXPECTED_DATASET_SHA256="4282529dbcc1b9253fa91da35b9f1768a2002b391cc90ac6a4e64575d59cfbf3"
export SWEBENCH_EXPECTED_OFFICIAL_DATASET_SHA256="f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c"
export HF_CKPT
export REF_LOAD
export INIT_CKPT="${INIT_CKPT:-${REF_LOAD}}"

CANONICAL_CONFIG_PATH="${TERMINAL_RL_DIR}/configs/rollout_qwen3_think.yaml"
if [[ -n "${CUSTOM_CONFIG_PATH:-}" && "${CUSTOM_CONFIG_PATH}" != "${CANONICAL_CONFIG_PATH}" ]]; then
  echo "[ERROR] Formal profile requires CUSTOM_CONFIG_PATH=${CANONICAL_CONFIG_PATH}." >&2
  exit 2
fi
export CUSTOM_CONFIG_PATH="${CANONICAL_CONFIG_PATH}"

# Qwen3 official thinking-mode recommendations plus the fixed Terminal-RL
# scaffold used for comparable SWE-bench Verified runs.
pin_formal_env SWEBENCH_AGENT_PROFILE qwen3_official_think_64k_terminal_ref_v1
pin_formal_env SWEBENCH_MODEL_NAME_OR_PATH Qwen/Qwen3-8B
pin_formal_env HARNESS_OPTION camel-agent
pin_formal_env EVAL_TEMPERATURE 0.6
pin_formal_env EVAL_TOP_P 0.95
pin_formal_env EVAL_TOP_K 20
pin_formal_env EVAL_MIN_P 0
pin_formal_env EVAL_N_SAMPLES 1
pin_formal_env EVAL_MAX_PROMPT_LEN 32768
pin_formal_env EVAL_MAX_RESPONSE_LEN 32768
pin_formal_env EVAL_MAX_CONTEXT_LEN 65536
pin_formal_env TERMINAL_MAX_TOTAL_TOKENS 65536
pin_formal_env MAX_TURN 200
pin_formal_env SGLANG_CONTEXT_LENGTH 65536
pin_formal_env SGLANG_JSON_MODEL_OVERRIDE_ARGS '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}}'
pin_formal_env EVAL_SEED 1234
pin_formal_env ROLLOUT_SEED 42
pin_formal_env EVAL_DETERMINISTIC 1
pin_formal_env SGLANG_REQUEST_TIMEOUT 1800

# Two TP=2 SGLang engines keep all four H20 GPUs active while the worker runs
# up to four independent Docker tasks.
pin_formal_env NUM_GPUS 4
pin_formal_env ROLLOUT_GPUS 4
pin_formal_env ROLLOUT_NUM_GPUS_PER_ENGINE 2
# eval-only declares a one-rank dummy Megatron actor for argument validation;
# its TP must remain 1 and is unrelated to the two-GPU SGLang engine TP above.
pin_formal_env ACTOR_NUM_NODES 1
pin_formal_env ACTOR_NUM_GPUS_PER_NODE 1
pin_formal_env MEGATRON_TP_SIZE 1
pin_formal_env EVAL_MAX_CONCURRENCY 4
pin_formal_env SWEBENCH_WORKER_MAX_CONCURRENT_BUILDS 1

cd "${REPO_ROOT}"
exec bash "${TERMINAL_RL_DIR}/terminal-rl_qwen3-8b_eval_pu.sh"
