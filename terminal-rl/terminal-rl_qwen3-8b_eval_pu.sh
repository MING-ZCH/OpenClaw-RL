#!/usr/bin/env bash
# Terminal-RL Qwen3-8B eval-only launcher.
#
# Runs slime/eval_only.py with SGLang rollout engines only. No actor/training
# workers are started.

set -euo pipefail
set -x

log() { echo "[$(date +'%F %T')] $*"; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] missing cmd: $1"; exit 1; }; }

LIGHTRFT_PY312_BIN="${LIGHTRFT_PY312_BIN:-}"
if [[ -n "${LIGHTRFT_PY312_BIN}" ]]; then
  export PATH="${LIGHTRFT_PY312_BIN}:${PATH}"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export REPO_ROOT
export SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/slime}"
export MEGATRON_DIR="${MEGATRON_DIR:-${REPO_ROOT}/Megatron-LM}"

TRAIN_PYTHON="${TRAIN_PYTHON:-$(command -v python3 || true)}"
HF_CKPT="${HF_CKPT:-}"
REF_LOAD="${REF_LOAD:-}"
INIT_CKPT="${INIT_CKPT:-${REF_LOAD}}"
STEP119_CKPT="${STEP119_CKPT:-}"

for required_name in TRAIN_PYTHON HF_CKPT REF_LOAD; do
  if [[ -z "${!required_name}" ]]; then
    echo "[ERROR] ${required_name} is required." >&2
    exit 2
  fi
done
if [[ ! -x "${TRAIN_PYTHON}" ]]; then
  echo "[ERROR] TRAIN_PYTHON is not executable: ${TRAIN_PYTHON}" >&2
  exit 2
fi

EVAL_CKPT="${EVAL_CKPT:-init}"
EVAL_SUITE="${EVAL_SUITE:-mock}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-1}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
EVAL_TOP_K="${EVAL_TOP_K:-20}"
EVAL_MIN_P="${EVAL_MIN_P:-0}"
EVAL_SEED="${EVAL_SEED:-1234}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-0}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"
EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-16384}"
EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-$((EVAL_MAX_PROMPT_LEN + EVAL_MAX_RESPONSE_LEN))}"
MAX_TURN="${MAX_TURN:-10}"
TERMINAL_MAX_TOTAL_TOKENS="${TERMINAL_MAX_TOTAL_TOKENS:-32768}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-}"
SGLANG_JSON_MODEL_OVERRIDE_ARGS="${SGLANG_JSON_MODEL_OVERRIDE_ARGS:-}"
SWEBENCH_AGENT_PROFILE="${SWEBENCH_AGENT_PROFILE:-custom}"
SWEBENCH_MODEL_NAME_OR_PATH="${SWEBENCH_MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-}"
EVAL_DRY_RUN="${EVAL_DRY_RUN:-0}"
FORMAL_SWEBENCH_VERIFIED="${FORMAL_SWEBENCH_VERIFIED:-0}"
SWEBENCH_DEFER_GRADING="${SWEBENCH_DEFER_GRADING:-0}"
OFFICIAL_SWEBENCH_VERIFIED_INSTANCES=500
OFFICIAL_SWEBENCH_VERIFIED_SHA256="4282529dbcc1b9253fa91da35b9f1768a2002b391cc90ac6a4e64575d59cfbf3"
OFFICIAL_SWEBENCH_VERIFIED_OFFICIAL_SHA256="f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c"
HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"
SETA_SAFETY="${SETA_SAFETY:-none}"
SAFETY_BENCH_REWARD="${SAFETY_BENCH_REWARD:-rule}"
AGENTHARM_REWARD="${AGENTHARM_REWARD:-rule}"
SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0}"

DETECTED_GPUS=0
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS="$(nvidia-smi -L 2>/dev/null | awk 'END {print NR + 0}')"
fi
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS}}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-${NUM_GPUS}}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${ROLLOUT_GPUS}}"
# eval_only.py still validates Megatron actor topology even though
# --debug-rollout-only does not instantiate an actor. Keep this independent
# from SGLang's tensor parallel size (ROLLOUT_NUM_GPUS_PER_ENGINE).
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-1}"
MEGATRON_TP_SIZE="${MEGATRON_TP_SIZE:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"

if (( NUM_GPUS <= 0 || ROLLOUT_GPUS <= 0 )); then
  echo "[ERROR] Eval requires visible rollout GPUs or explicit positive NUM_GPUS/ROLLOUT_GPUS." >&2
  exit 2
fi
ACTOR_WORLD_SIZE=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
if (( MEGATRON_TP_SIZE <= 0 || ACTOR_WORLD_SIZE % MEGATRON_TP_SIZE != 0 )); then
  echo "[ERROR] Megatron eval topology is invalid: actor_world_size=${ACTOR_WORLD_SIZE}, megatron_tp=${MEGATRON_TP_SIZE}." >&2
  echo "[ERROR] SGLang TP is configured separately with ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}." >&2
  exit 2
fi
if (( ROLLOUT_NUM_GPUS_PER_ENGINE <= 0 || ROLLOUT_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE != 0 )); then
  echo "[ERROR] SGLang eval topology is invalid: rollout_gpus=${ROLLOUT_GPUS}, gpus_per_engine=${ROLLOUT_NUM_GPUS_PER_ENGINE}." >&2
  exit 2
fi

case "${EVAL_CKPT}" in
  init)
    CKPT_LABEL="init"
    LOAD_CKPT="${LOAD_CKPT:-${INIT_CKPT}}"
    CKPT_STEP_ARGS=()
    ;;
  step119)
    if [[ -z "${STEP119_CKPT}" ]]; then
      echo "[ERROR] EVAL_CKPT=step119 requires STEP119_CKPT=/path/to/checkpoint"
      exit 1
    fi
    CKPT_LABEL="step119"
    LOAD_CKPT="${LOAD_CKPT:-${STEP119_CKPT}}"
    CKPT_STEP_ARGS=(--ckpt-step 119)
    ;;
  custom)
    if [[ -z "${LOAD_CKPT:-}" ]]; then
      echo "[ERROR] EVAL_CKPT=custom requires LOAD_CKPT=/path/to/ckpt"
      exit 1
    fi
    CKPT_LABEL="custom-$(basename "${LOAD_CKPT}")"
    CKPT_STEP_ARGS=()
    if [[ -n "${CKPT_STEP:-}" ]]; then
      CKPT_STEP_ARGS=(--ckpt-step "${CKPT_STEP}")
    fi
    ;;
  *)
    echo "[ERROR] Unknown EVAL_CKPT=${EVAL_CKPT}. Use: init|step119|custom"
    exit 1
    ;;
esac

case "${EVAL_SUITE}" in
  mock|all|seta|safety|agentharm|sweverified|swe-verified) ;;
  *)
    echo "[ERROR] Unknown EVAL_SUITE=${EVAL_SUITE}. Use: mock|all|seta|safety|agentharm|sweverified"
    exit 1
    ;;
esac
if [[ "${EVAL_SUITE}" == "swe-verified" ]]; then
  EVAL_SUITE="sweverified"
fi
if [[ "${EVAL_SUITE}" == "sweverified" && "${EVAL_N_SAMPLES}" != "1" ]]; then
  echo "[ERROR] Official SWE-bench Verified evaluation requires EVAL_N_SAMPLES=1"
  exit 1
fi
if [[ "${FORMAL_SWEBENCH_VERIFIED}" == "1" ]]; then
  EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-4}"
  if [[ "${EVAL_SUITE}" != "sweverified" || -n "${EVAL_LIMIT}" ]]; then
    echo "[ERROR] FORMAL_SWEBENCH_VERIFIED=1 requires EVAL_SUITE=sweverified and an empty EVAL_LIMIT." >&2
    exit 2
  fi
  if [[ "${NUM_GPUS}" != "4" || "${ROLLOUT_GPUS}" != "4" || "${ROLLOUT_NUM_GPUS_PER_ENGINE}" != "2" || "${EVAL_MAX_CONCURRENCY}" != "4" ]]; then
    echo "[ERROR] Formal SWE-bench Verified topology must be 4 rollout GPUs with two TP=2 engines and eval concurrency 4." >&2
    exit 2
  fi
  if [[ "${EVAL_DRY_RUN}" != "1" && "${DETECTED_GPUS}" -lt 4 ]]; then
    echo "[ERROR] Formal SWE-bench Verified requires at least 4 visible GPUs; detected ${DETECTED_GPUS}." >&2
    exit 2
  fi
  FORMAL_PROFILE_EXPECTATIONS=(
    "SWEBENCH_AGENT_PROFILE=qwen3_official_think_64k_terminal_ref_v1"
    "SWEBENCH_MODEL_NAME_OR_PATH=Qwen/Qwen3-8B"
    "HARNESS_OPTION=camel-agent"
    "EVAL_TEMPERATURE=0.6"
    "EVAL_TOP_P=0.95"
    "EVAL_TOP_K=20"
    "EVAL_MIN_P=0"
    "EVAL_N_SAMPLES=1"
    "EVAL_MAX_PROMPT_LEN=32768"
    "EVAL_MAX_RESPONSE_LEN=32768"
    "EVAL_MAX_CONTEXT_LEN=65536"
    "TERMINAL_MAX_TOTAL_TOKENS=65536"
    "MAX_TURN=200"
    "SGLANG_CONTEXT_LENGTH=65536"
    'SGLANG_JSON_MODEL_OVERRIDE_ARGS={"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}}'
    "EVAL_SEED=1234"
    "ROLLOUT_SEED=42"
    "EVAL_DETERMINISTIC=1"
  )
  for expectation in "${FORMAL_PROFILE_EXPECTATIONS[@]}"; do
    name="${expectation%%=*}"
    expected="${expectation#*=}"
    if [[ "${!name}" != "${expected}" ]]; then
      echo "[ERROR] Formal SWE-bench profile requires ${name}=${expected}; got ${!name}." >&2
      exit 2
    fi
  done
  if [[ -n "${SWEBENCH_EXPECTED_INSTANCES:-}" && "${SWEBENCH_EXPECTED_INSTANCES}" != "${OFFICIAL_SWEBENCH_VERIFIED_INSTANCES}" ]]; then
    echo "[ERROR] Formal SWE-bench Verified count is pinned to ${OFFICIAL_SWEBENCH_VERIFIED_INSTANCES}; got ${SWEBENCH_EXPECTED_INSTANCES}." >&2
    exit 2
  fi
  if [[ -n "${SWEBENCH_EXPECTED_DATASET_SHA256:-}" && "${SWEBENCH_EXPECTED_DATASET_SHA256}" != "${OFFICIAL_SWEBENCH_VERIFIED_SHA256}" ]]; then
    echo "[ERROR] Formal SWE-bench Verified dataset SHA256 is pinned to the official converted dataset." >&2
    exit 2
  fi
  if [[ -n "${SWEBENCH_EXPECTED_OFFICIAL_DATASET_SHA256:-}" && "${SWEBENCH_EXPECTED_OFFICIAL_DATASET_SHA256}" != "${OFFICIAL_SWEBENCH_VERIFIED_OFFICIAL_SHA256}" ]]; then
    echo "[ERROR] Formal official SWE-bench dataset SHA256 is pinned." >&2
    exit 2
  fi
  SWEBENCH_EXPECTED_INSTANCES="${OFFICIAL_SWEBENCH_VERIFIED_INSTANCES}"
  SWEBENCH_EXPECTED_DATASET_SHA256="${OFFICIAL_SWEBENCH_VERIFIED_SHA256}"
  SWEBENCH_EXPECTED_OFFICIAL_DATASET_SHA256="${OFFICIAL_SWEBENCH_VERIFIED_OFFICIAL_SHA256}"
  export \
    SWEBENCH_EXPECTED_INSTANCES \
    SWEBENCH_EXPECTED_DATASET_SHA256 \
    SWEBENCH_EXPECTED_OFFICIAL_DATASET_SHA256
fi
if [[ "${SWEBENCH_DEFER_GRADING}" != "0" && "${SWEBENCH_DEFER_GRADING}" != "1" ]]; then
  echo "[ERROR] SWEBENCH_DEFER_GRADING must be 0 or 1." >&2
  exit 2
fi
if [[ "${SWEBENCH_DEFER_GRADING}" == "1" && "${EVAL_SUITE}" != "sweverified" ]]; then
  echo "[ERROR] SWEBENCH_DEFER_GRADING=1 requires EVAL_SUITE=sweverified." >&2
  exit 2
fi
if [[ "${EVAL_SUITE}" == "sweverified" && "${SWEBENCH_DEFER_GRADING}" != "1" ]]; then
  echo "[ERROR] SWE-bench Verified workers only export predictions; set SWEBENCH_DEFER_GRADING=1." >&2
  exit 2
fi
export SWEBENCH_DEFER_GRADING

SAFETY_DATA="${SCRIPT_DIR}/dataset/agent_safetybench_convert/train.jsonl"
AGENTHARM_DATA="${SCRIPT_DIR}/dataset/agentharm_convert/val.jsonl"
SETA_DATA="${SCRIPT_DIR}/dataset/seta_env_convert/train.jsonl"
SWEVERIFIED_DATA="${SWEVERIFIED_DATA:-${SCRIPT_DIR}/dataset/sweverified_convert/test.jsonl}"
SWEVERIFIED_OFFICIAL_DATA="${SWEVERIFIED_OFFICIAL_DATA:-${SCRIPT_DIR}/dataset/sweverified_convert/official_test.jsonl}"

EVAL_PROMPT_DATA=()
INCLUDES_SETA=0
INCLUDES_SAFETY=0
INCLUDES_AGENTHARM=0
INCLUDES_SWEVERIFIED=0
case "${EVAL_SUITE}" in
  mock)
    EVAL_PROMPT_DATA=(safety "${SAFETY_DATA}" agentharm "${AGENTHARM_DATA}")
    INCLUDES_SAFETY=1
    INCLUDES_AGENTHARM=1
    ;;
  all)
    EVAL_PROMPT_DATA=(safety "${SAFETY_DATA}" agentharm "${AGENTHARM_DATA}" seta "${SETA_DATA}")
    INCLUDES_SAFETY=1
    INCLUDES_AGENTHARM=1
    INCLUDES_SETA=1
    ;;
  safety)
    EVAL_PROMPT_DATA=(safety "${SAFETY_DATA}")
    INCLUDES_SAFETY=1
    ;;
  agentharm)
    EVAL_PROMPT_DATA=(agentharm "${AGENTHARM_DATA}")
    INCLUDES_AGENTHARM=1
    ;;
  seta)
    EVAL_PROMPT_DATA=(seta "${SETA_DATA}")
    INCLUDES_SETA=1
    ;;
  sweverified)
    EVAL_PROMPT_DATA=(sweverified "${SWEVERIFIED_DATA}")
    INCLUDES_SWEVERIFIED=1
    ;;
esac

for ((i=1; i<${#EVAL_PROMPT_DATA[@]}; i+=2)); do
  if [[ ! -f "${EVAL_PROMPT_DATA[$i]}" ]]; then
    echo "[ERROR] eval dataset not found: ${EVAL_PROMPT_DATA[$i]}"
    exit 1
  fi
done

SWEBENCH_DATASET_ROWS=""
SWEBENCH_DATASET_UNIQUE_IDS=""
SWEBENCH_DATASET_SHA256=""
SWEBENCH_OFFICIAL_DATA_SHA256=""
if [[ "${INCLUDES_SWEVERIFIED}" == "1" ]]; then
  if ! SWEBENCH_PREFLIGHT_OUTPUT="$(
    "${TRAIN_PYTHON}" - \
      "${SWEVERIFIED_DATA}" \
      "${SWEVERIFIED_OFFICIAL_DATA}" \
      "${FORMAL_SWEBENCH_VERIFIED}" \
      "${SWEBENCH_EXPECTED_INSTANCES:-500}" \
      "${SWEBENCH_EXPECTED_DATASET_SHA256:-}" \
      "${SWEBENCH_EXPECTED_OFFICIAL_DATASET_SHA256:-}" 2>&1 <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
official_path = Path(sys.argv[2])
formal = sys.argv[3] == "1"
expected = int(sys.argv[4])
expected_sha256 = sys.argv[5].strip().lower()
expected_official_sha256 = sys.argv[6].strip().lower()
rows = []
ids = []
expected_harness_version = "4.1.0"
expected_harness_commit = "f7bbbb2ccdf479001d6467c9e34af59e44a840f9"
expected_dataset = "princeton-nlp/SWE-bench_Verified"
expected_revision = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
expected_format = "sweverified-terminal-rl-v1"
for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    row = json.loads(line)
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise SystemExit(f"row {line_no}: missing metadata")
    instance_id = str(metadata.get("swe_instance_id") or "").strip()
    if not instance_id:
        raise SystemExit(f"row {line_no}: missing swe_instance_id")
    if str(metadata.get("suite") or "").lower() != "sweverified":
        raise SystemExit(f"row {line_no}: suite is not sweverified")
    if not str(metadata.get("task_path") or "").startswith("sweverified_env/"):
        raise SystemExit(f"row {line_no}: invalid task_path")
    if metadata.get("swebench_harness_version") != expected_harness_version:
        raise SystemExit(f"row {line_no}: unexpected SWE-bench harness version")
    if metadata.get("swebench_harness_commit") != expected_harness_commit:
        raise SystemExit(f"row {line_no}: unexpected SWE-bench harness commit")
    if metadata.get("source_dataset") != expected_dataset:
        raise SystemExit(f"row {line_no}: unexpected source dataset")
    if metadata.get("source_revision") != expected_revision:
        raise SystemExit(f"row {line_no}: unexpected source revision")
    if metadata.get("task_format_version") != expected_format:
        raise SystemExit(f"row {line_no}: unexpected task format")
    ids.append(instance_id)
    rows.append(row)

unique_ids = set(ids)
if len(unique_ids) != len(ids):
    raise SystemExit(f"duplicate SWE-bench instance IDs: rows={len(ids)} unique={len(unique_ids)}")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if formal and len(rows) != expected:
    raise SystemExit(f"formal SWE-bench Verified requires {expected} rows, found {len(rows)}")
if formal and len(unique_ids) != expected:
    raise SystemExit(f"formal SWE-bench Verified requires {expected} unique IDs, found {len(unique_ids)}")
if expected_sha256 and digest != expected_sha256:
    raise SystemExit(
        "SWE-bench Verified dataset fingerprint mismatch: "
        f"actual={digest} expected={expected_sha256}"
    )
if not official_path.is_file():
    raise SystemExit(f"missing pinned official SWE-bench dataset: {official_path}")
official_rows = []
official_ids = []
required_official_fields = {
    "repo",
    "instance_id",
    "base_commit",
    "patch",
    "test_patch",
    "problem_statement",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
}
for line_no, line in enumerate(
    official_path.read_text(encoding="utf-8").splitlines(), 1
):
    if not line.strip():
        continue
    row = json.loads(line)
    missing = required_official_fields - set(row)
    if missing:
        raise SystemExit(
            f"official row {line_no}: missing fields {sorted(missing)}"
        )
    official_ids.append(str(row["instance_id"]))
    official_rows.append(row)
if len(set(official_ids)) != len(official_ids):
    raise SystemExit("duplicate IDs in pinned official SWE-bench dataset")
if set(official_ids) != unique_ids:
    raise SystemExit(
        "converted and official SWE-bench datasets contain different instance IDs"
    )
if formal and len(official_rows) != expected:
    raise SystemExit(
        f"formal pinned official dataset requires {expected} rows, "
        f"found {len(official_rows)}"
    )
official_digest = hashlib.sha256(official_path.read_bytes()).hexdigest()
if expected_official_sha256 and official_digest != expected_official_sha256:
    raise SystemExit(
        "pinned official SWE-bench dataset fingerprint mismatch: "
        f"actual={official_digest} expected={expected_official_sha256}"
    )
print(len(rows), len(unique_ids), digest, official_digest)
PY
  )"; then
    echo "[ERROR] SWE-bench Verified dataset preflight failed: ${SWEBENCH_PREFLIGHT_OUTPUT}" >&2
    exit 2
  fi
  read -r \
    SWEBENCH_DATASET_ROWS \
    SWEBENCH_DATASET_UNIQUE_IDS \
    SWEBENCH_DATASET_SHA256 \
    SWEBENCH_OFFICIAL_DATA_SHA256 <<< "${SWEBENCH_PREFLIGHT_OUTPUT}"
fi

export AGENT_SAFETYBENCH_REMOTE_ENV=0
export AGENTHARM_REMOTE_ENV=0
export AGENT_SAFETYBENCH_ROOT="${AGENT_SAFETYBENCH_ROOT:-}"
export AGENTHARM_ROOT="${AGENTHARM_ROOT:-}"
if [[ "${INCLUDES_SAFETY}" == "1" && -z "${AGENT_SAFETYBENCH_ROOT}" ]]; then
  echo "[ERROR] EVAL_SUITE=${EVAL_SUITE} requires AGENT_SAFETYBENCH_ROOT." >&2
  exit 2
fi
if [[ "${INCLUDES_AGENTHARM}" == "1" && -z "${AGENTHARM_ROOT}" ]]; then
  echo "[ERROR] EVAL_SUITE=${EVAL_SUITE} requires AGENTHARM_ROOT." >&2
  exit 2
fi
export SAFETY_BENCH_REWARD AGENTHARM_REWARD SETA_SAFETY SAFETY_REWARD_COEF

INCLUDES_REMOTE_ENV=0
if [[ "${INCLUDES_SETA}" == "1" || "${INCLUDES_SWEVERIFIED}" == "1" ]]; then
  INCLUDES_REMOTE_ENV=1
fi

if [[ "${INCLUDES_REMOTE_ENV}" == "1" && -z "${WORKER_URLS:-}" ]]; then
  echo "[ERROR] EVAL_SUITE=${EVAL_SUITE} requires WORKER_URLS=http://<worker-host>:<worker-port>"
  exit 1
fi

if [[ "${INCLUDES_REMOTE_ENV}" == "1" ]]; then
  # Eval rollout launches one async task per prompt. Keep remote-env eval below
  # the Docker worker capacity; otherwise queued prompts can hit admission
  # timeout before they ever start.
  if [[ "${INCLUDES_SWEVERIFIED}" == "1" ]]; then
    # Verified eval uses one run for each distinct task. This must not exceed
    # WORKER_MAX_TASKS (the documented worker default is 4).
    EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-4}"
    # The official instance images can take several minutes to pull. These
    # task-level values are sent in /reset and therefore must match the worker
    # limits instead of falling back to generate.py's 300-second defaults.
    export TERMINAL_ENSURE_IMAGE_TIMEOUT="${TERMINAL_ENSURE_IMAGE_TIMEOUT:-3600}"
    export TERMINAL_RESET_SESSION_TIMEOUT="${TERMINAL_RESET_SESSION_TIMEOUT:-900}"
    export TERMINAL_CLOSE_SESSION_TIMEOUT="${TERMINAL_CLOSE_SESSION_TIMEOUT:-120}"
    export TERMINAL_EVAL_TIMEOUT="${TERMINAL_EVAL_TIMEOUT:-1800}"
    export SWEBENCH_WORKER_MAX_CONCURRENT_BUILDS="${SWEBENCH_WORKER_MAX_CONCURRENT_BUILDS:-1}"

    # /reset performs image preparation and container startup sequentially.
    # Keep the router above that full budget, and above evaluate()'s timeout,
    # so it never replays a request merely because an inner timeout is expiring.
    if [[ ! -x "${TRAIN_PYTHON}" ]]; then
      echo "[ERROR] TRAIN_PYTHON is not executable: ${TRAIN_PYTHON}" >&2
      exit 1
    fi
    REQUIRED_ROUTER_FORWARD_TIMEOUT="$(
      "${TRAIN_PYTHON}" - \
        "${TERMINAL_ENSURE_IMAGE_TIMEOUT}" \
        "${TERMINAL_RESET_SESSION_TIMEOUT}" \
        "${TERMINAL_EVAL_TIMEOUT}" \
        "${EVAL_MAX_CONCURRENCY}" \
        "${SWEBENCH_WORKER_MAX_CONCURRENT_BUILDS}" <<'PY'
import math
import sys

try:
    ensure_image, reset_session, evaluate = map(float, sys.argv[1:4])
    concurrency, concurrent_builds = map(int, sys.argv[4:])
except (TypeError, ValueError) as exc:
    raise SystemExit(f"invalid SWE-bench timeout: {exc}") from exc
if min(ensure_image, reset_session, evaluate) <= 0:
    raise SystemExit("SWE-bench timeouts must all be positive")
if concurrency <= 0 or concurrent_builds <= 0:
    raise SystemExit("SWE-bench concurrency/build concurrency must be positive")
build_waves = math.ceil(concurrency / concurrent_builds)
queued_reset = build_waves * ensure_image + reset_session
print(math.ceil(max(queued_reset, evaluate + 30.0) + 300.0))
PY
    )"
    export ROUTER_FORWARD_TIMEOUT="${ROUTER_FORWARD_TIMEOUT:-${REQUIRED_ROUTER_FORWARD_TIMEOUT}}"
    export ROUTER_FORWARD_RETRIES="${ROUTER_FORWARD_RETRIES:-1}"
    export ENV_HTTP_MAX_RETRIES="${ENV_HTTP_MAX_RETRIES:-10}"
    export ENV_ALLOCATE_MAX_RETRIES="${ENV_ALLOCATE_MAX_RETRIES:-100}"
    export ENV_RESET_MAX_RETRIES="${ENV_RESET_MAX_RETRIES:-1}"
    export ENV_EXEC_TOOL_MAX_RETRIES="${ENV_EXEC_TOOL_MAX_RETRIES:-1}"
    export ENV_EVALUATE_MAX_RETRIES="${ENV_EVALUATE_MAX_RETRIES:-1}"
    export ENV_CLOSE_MAX_RETRIES="${ENV_CLOSE_MAX_RETRIES:-3}"
    for retry_var in \
      ENV_HTTP_MAX_RETRIES \
      ENV_ALLOCATE_MAX_RETRIES \
      ENV_RESET_MAX_RETRIES \
      ENV_EXEC_TOOL_MAX_RETRIES \
      ENV_EVALUATE_MAX_RETRIES \
      ENV_CLOSE_MAX_RETRIES; do
      retry_value="${!retry_var}"
      if [[ ! "${retry_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] ${retry_var} must be a positive integer, got '${retry_value}'" >&2
        exit 2
      fi
    done
    "${TRAIN_PYTHON}" - \
      "${ROUTER_FORWARD_TIMEOUT}" \
      "${REQUIRED_ROUTER_FORWARD_TIMEOUT}" <<'PY'
import sys

try:
    configured, required = map(float, sys.argv[1:])
except (TypeError, ValueError) as exc:
    raise SystemExit(f"invalid ROUTER_FORWARD_TIMEOUT: {exc}") from exc
if configured < required:
    raise SystemExit(
        "ROUTER_FORWARD_TIMEOUT is too small for SWE-bench Verified: "
        f"configured={configured:g}s required>={required:g}s"
    )
PY
    CLIENT_HTTP_TIMEOUT="$("${TRAIN_PYTHON}" - "${ROUTER_FORWARD_TIMEOUT}" <<'PY'
import math
import sys

print(math.ceil(float(sys.argv[1]) + 60.0))
PY
    )"
    export ENV_RESET_HTTP_TIMEOUT="${ENV_RESET_HTTP_TIMEOUT:-${CLIENT_HTTP_TIMEOUT}}"
    export ENV_ALLOCATE_HTTP_TIMEOUT="${ENV_ALLOCATE_HTTP_TIMEOUT:-${CLIENT_HTTP_TIMEOUT}}"
    export ENV_EXEC_TOOL_HTTP_TIMEOUT="${ENV_EXEC_TOOL_HTTP_TIMEOUT:-${CLIENT_HTTP_TIMEOUT}}"
    export ENV_EVALUATE_HTTP_TIMEOUT="${ENV_EVALUATE_HTTP_TIMEOUT:-${CLIENT_HTTP_TIMEOUT}}"
    export ENV_CLOSE_HTTP_TIMEOUT="${ENV_CLOSE_HTTP_TIMEOUT:-${CLIENT_HTTP_TIMEOUT}}"
    export ENV_HEARTBEAT_HTTP_TIMEOUT="${ENV_HEARTBEAT_HTTP_TIMEOUT:-300}"
    "${TRAIN_PYTHON}" - \
      "${CLIENT_HTTP_TIMEOUT}" \
      "${ENV_ALLOCATE_HTTP_TIMEOUT}" \
      "${ENV_RESET_HTTP_TIMEOUT}" \
      "${ENV_EXEC_TOOL_HTTP_TIMEOUT}" \
      "${ENV_EVALUATE_HTTP_TIMEOUT}" \
      "${ENV_CLOSE_HTTP_TIMEOUT}" \
      "${ENV_HEARTBEAT_HTTP_TIMEOUT}" <<'PY'
import sys

labels = ("allocate", "reset", "exec_tool", "evaluate", "close")
try:
    required = float(sys.argv[1])
    configured = list(map(float, sys.argv[2:7]))
    heartbeat = float(sys.argv[7])
except (TypeError, ValueError) as exc:
    raise SystemExit(f"invalid terminal env client timeout: {exc}") from exc
for label, value in zip(labels, configured, strict=True):
    if value < required:
        raise SystemExit(
            f"ENV_{label.upper()}_HTTP_TIMEOUT is too small: "
            f"configured={value:g}s required>={required:g}s"
        )
if heartbeat <= 0:
    raise SystemExit("ENV_HEARTBEAT_HTTP_TIMEOUT must be positive")
PY
  else
    EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-8}"
  fi
  export ENV_REMOTE_MAX_ACTIVE_TASKS="${ENV_REMOTE_MAX_ACTIVE_TASKS:-${EVAL_MAX_CONCURRENCY}}"
  export ENV_REMOTE_ADMISSION_TIMEOUT="${ENV_REMOTE_ADMISSION_TIMEOUT:-21600}"
  export ENV_REMOTE_ADMISSION_LOG_INTERVAL="${ENV_REMOTE_ADMISSION_LOG_INTERVAL:-60}"
  if [[ "${INCLUDES_SWEVERIFIED}" != "1" ]]; then
    export ENV_RESET_HTTP_TIMEOUT="${ENV_RESET_HTTP_TIMEOUT:-900}"
  fi
  export ROUTER_PRESSURE_COOLDOWN="${ROUTER_PRESSURE_COOLDOWN:-5}"
else
  EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-0}"
fi
export EVAL_ROLLOUT_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
RUN_ID="${RUN_ID:-eval_qwen3-8b_${CKPT_LABEL}_${EVAL_SUITE}_${RUN_TIMESTAMP}}"
RUN_DIR="${RUNS_ROOT}/${RUN_ID}"
RUN_LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${RUN_LOG_DIR}" "${RUN_DIR}/config" "${RUN_DIR}/trajectories"
SWEBENCH_RUN_OFFICIAL_DATA=""
if [[ "${INCLUDES_SWEVERIFIED}" == "1" ]]; then
  SWEBENCH_RUN_OFFICIAL_DATA="${RUN_DIR}/config/swebench_verified_official_test.jsonl"
  install -m 0644 "${SWEVERIFIED_OFFICIAL_DATA}" "${SWEBENCH_RUN_OFFICIAL_DATA}"
fi

if [[ "${RESET_RUN_OUTPUTS:-1}" == "1" ]]; then
  rm -f \
    "${RUN_DIR}/eval_summary.json" \
    "${RUN_DIR}/eval_summary.tsv" \
    "${RUN_LOG_DIR}/metrics.jsonl" \
    "${RUN_LOG_DIR}/eval.log" \
    "${RUN_LOG_DIR}/ray_job.log" \
    "${RUN_LOG_DIR}/router.log" \
    "${RUN_LOG_DIR}/router.pid"
  find "${RUN_DIR}/trajectories" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi

if [[ -n "${EVAL_LIMIT}" ]]; then
  LIMITED_EVAL_PROMPT_DATA=()
  for ((i=0; i<${#EVAL_PROMPT_DATA[@]}; i+=2)); do
    name="${EVAL_PROMPT_DATA[$i]}"
    src="${EVAL_PROMPT_DATA[$((i+1))]}"
    dst="${RUN_DIR}/config/${name}.limit${EVAL_LIMIT}.jsonl"
    "${TRAIN_PYTHON}" - "$src" "$dst" "$EVAL_LIMIT" <<'PY'
import sys

src, dst, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as fin, open(dst, "w") as fout:
    for idx, line in enumerate(fin):
        if idx >= limit:
            break
        fout.write(line)
PY
    LIMITED_EVAL_PROMPT_DATA+=("${name}" "${dst}")
  done
  EVAL_PROMPT_DATA=("${LIMITED_EVAL_PROMPT_DATA[@]}")
fi

BASE_CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_qwen3_think.yaml}"
CUSTOM_CONFIG_PATH="${RUN_DIR}/config/rollout_qwen3_think_eval.yaml"
if [[ "${FORMAL_SWEBENCH_VERIFIED}" == "1" ]]; then
  CANONICAL_CUSTOM_CONFIG_PATH="${SCRIPT_DIR}/configs/rollout_qwen3_think.yaml"
  if [[ "$(readlink -f "${BASE_CUSTOM_CONFIG_PATH}")" != "$(readlink -f "${CANONICAL_CUSTOM_CONFIG_PATH}")" ]]; then
    echo "[ERROR] Formal SWE-bench profile requires ${CANONICAL_CUSTOM_CONFIG_PATH}." >&2
    exit 2
  fi
fi
"${TRAIN_PYTHON}" - \
  "$BASE_CUSTOM_CONFIG_PATH" \
  "$CUSTOM_CONFIG_PATH" \
  "$MAX_TURN" \
  "$HARNESS_OPTION" \
  "$TERMINAL_MAX_TOTAL_TOKENS" <<'PY'
import sys
import yaml

src, dst = sys.argv[1:3]
max_turn = int(sys.argv[3])
harness = sys.argv[4]
max_total_tokens = int(sys.argv[5])
with open(src) as f:
    cfg = yaml.safe_load(f) or {}
cfg["max_iteration"] = max_turn
cfg["harness_option"] = harness
cfg["terminal_agent_type"] = "camel_agent"
cfg["max_total_tokens"] = max_total_tokens
cfg["non_think_mode"] = False
cfg["trajectory_save_interval"] = 1
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=True)
PY

export TERMINAL_STRUCTURED_METRICS="${TERMINAL_STRUCTURED_METRICS:-1}"
export TERMINAL_METRICS_JSONL="${TERMINAL_METRICS_JSONL:-${RUN_LOG_DIR}/metrics.jsonl}"
export TERMINAL_SAVE_TRAJ_DIR="${TERMINAL_SAVE_TRAJ_DIR:-${RUN_DIR}/trajectories}"
export SWEBENCH_MODEL_NAME_OR_PATH
export SWEBENCH_RESULTS_DIR="${SWEBENCH_RESULTS_DIR:-${RUN_DIR}/swebench_official}"
if [[ "${EVAL_SUITE}" == "sweverified" ]]; then
  export SWEBENCH_EVAL_DATA_PATH="${EVAL_PROMPT_DATA[1]}"
fi
export RUN_DIR RUN_ID RUN_LOG_DIR HARNESS_OPTION
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
export SLIME_RAY_PLACEMENT_GPU_PROBE="${SLIME_RAY_PLACEMENT_GPU_PROBE:-0}"
export SGLANG_REQUEST_TIMEOUT="${SGLANG_REQUEST_TIMEOUT:-1800}"

CODE_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
if ! HF_MODEL_PREFLIGHT_OUTPUT="$(
  "${TRAIN_PYTHON}" - "${HF_CKPT}" "${REPO_ROOT}" "${FORMAL_SWEBENCH_VERIFIED}" 2>&1 <<'PY'
import hashlib
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1]).resolve()
repo_root = Path(sys.argv[2]).resolve()
formal = sys.argv[3] == "1"
expected_revision = "b968826d9c46dd6066d109eabc6255188de91218"
expected_artifacts = {
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
    "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
config_path = model_dir / "config.json"
if not config_path.is_file():
    raise SystemExit(f"missing HF config: {config_path}")
config = json.loads(config_path.read_text(encoding="utf-8"))
if formal:
    expected = {
        "model_type": "qwen3",
        "num_hidden_layers": 36,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "max_position_embeddings": 40960,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(
                f"HF_CKPT is not the expected Qwen3-8B architecture: "
                f"{key}={config.get(key)!r}, expected={value!r}"
            )
    generation_path = model_dir / "generation_config.json"
    if not generation_path.is_file():
        raise SystemExit(f"missing official generation config: {generation_path}")
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation_expected = {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }
    for key, value in generation_expected.items():
        if generation.get(key) != value:
            raise SystemExit(
                f"official Qwen3-8B generation config mismatch: "
                f"{key}={generation.get(key)!r}, expected={value!r}"
            )

def digest_entries(entries):
    hasher = hashlib.sha256()
    for name, value in entries:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(value)
        hasher.update(b"\0")
    return hasher.hexdigest()

def file_sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

metadata_dir = model_dir / ".cache" / "huggingface" / "download"
revisions = set()
model_entries = []
if metadata_dir.is_dir():
    for path in sorted(metadata_dir.glob("*.metadata")):
        content = path.read_bytes()
        model_entries.append((str(path.relative_to(model_dir)), content))
        lines = content.decode("utf-8", errors="replace").splitlines()
        if lines:
            revisions.add(lines[0].strip())
revision = next(iter(revisions)) if len(revisions) == 1 else "unknown"
if formal:
    if revision != expected_revision:
        raise SystemExit(
            "HF_CKPT revision mismatch for official Qwen3-8B: "
            f"actual={revision} expected={expected_revision}"
        )
    model_entries = []
    for name, expected_digest in sorted(expected_artifacts.items()):
        path = model_dir / name
        if not path.is_file():
            raise SystemExit(f"official Qwen3-8B artifact is missing: {path}")
        actual_digest = file_sha256(path)
        if actual_digest != expected_digest:
            raise SystemExit(
                "official Qwen3-8B artifact fingerprint mismatch: "
                f"file={name} actual={actual_digest} expected={expected_digest}"
            )
        model_entries.append((name, actual_digest.encode("ascii")))
else:
    for path in sorted(model_dir.glob("model-*.safetensors")):
        model_entries.append((path.name, str(path.stat().st_size).encode("ascii")))
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        path = model_dir / name
        if path.is_file():
            model_entries.append((name, path.read_bytes()))

code_paths = [
    "terminal-rl/env_client.py",
    "terminal-rl/generate.py",
    "terminal-rl/router_server.py",
    "terminal-rl/remote/pool_server.py",
    "terminal-rl/remote/swe_task_utils.py",
    "terminal-rl/remote/terminal_env.py",
    "terminal-rl/swebench_report.py",
    "terminal-rl/scripts/run_sweverified_qwen3_8b_base_think_eval.sh",
    "terminal-rl/terminal-rl_qwen3-8b_eval_pu.sh",
]
code_entries = []
for relative in code_paths:
    path = repo_root / relative
    if not path.is_file():
        raise SystemExit(f"missing runtime code file: {path}")
    code_entries.append((relative, path.read_bytes()))

print(
    revision,
    digest_entries(model_entries),
    hashlib.sha256(config_path.read_bytes()).hexdigest(),
    digest_entries(code_entries),
)
PY
)"; then
  echo "[ERROR] Model/runtime provenance preflight failed: ${HF_MODEL_PREFLIGHT_OUTPUT}" >&2
  exit 2
fi
read -r HF_MODEL_REVISION HF_MODEL_MANIFEST_SHA256 HF_CONFIG_SHA256 CODE_RUNTIME_SHA256 <<< "${HF_MODEL_PREFLIGHT_OUTPUT}"
if [[ -z "${HF_MODEL_REVISION}" || -z "${HF_MODEL_MANIFEST_SHA256}" || -z "${HF_CONFIG_SHA256}" || -z "${CODE_RUNTIME_SHA256}" ]]; then
  echo "[ERROR] Model/runtime provenance preflight returned incomplete output: ${HF_MODEL_PREFLIGHT_OUTPUT}" >&2
  exit 2
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
  MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
export MASTER_ADDR
NODE_IP="${MASTER_ADDR}"

if [[ "${INCLUDES_REMOTE_ENV}" == "1" ]]; then
  export USE_REMOTE_ENV=1
  export ENV_SERVER_BIND_HOST="${ENV_SERVER_BIND_HOST:-0.0.0.0}"
  export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18080}"
  export ENV_SERVER_HOST="${ENV_SERVER_HOST:-${MASTER_ADDR}}"
  export ENV_SERVER_URL="${ENV_SERVER_URL:-http://${ENV_SERVER_HOST}:${ENV_SERVER_PORT}}"
else
  export USE_REMOTE_ENV=0
  export ENV_SERVER_URL="${ENV_SERVER_URL:-http://${MASTER_ADDR}:18080}"
fi

ALL_WORKER_HOSTS=""
if [[ -n "${WORKER_URLS:-}" ]]; then
  ALL_WORKER_HOSTS="$(echo "${WORKER_URLS}" | tr ',' '\n' | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
fi
NO_PROXY_VALUE="${NO_PROXY:-localhost,127.0.0.1}"
append_no_proxy_host() {
  local host="$1"
  [[ -z "${host}" ]] && return
  case ",${NO_PROXY_VALUE}," in
    *",${host},"*) ;;
    *) NO_PROXY_VALUE="${NO_PROXY_VALUE},${host}" ;;
  esac
}
append_no_proxy_host "${MASTER_ADDR}"
append_no_proxy_host "${ENV_SERVER_HOST:-}"
if [[ -n "${ALL_WORKER_HOSTS}" ]]; then
  IFS=',' read -r -a _NO_PROXY_WORKER_HOSTS <<< "${ALL_WORKER_HOSTS}"
  for _host in "${_NO_PROXY_WORKER_HOSTS[@]}"; do
    append_no_proxy_host "${_host}"
  done
fi
export NO_PROXY="${NO_PROXY_VALUE}"
export no_proxy="${NO_PROXY}"

ROUTER_PID=""
cleanup() {
  set +e
  if [[ -n "${ROUTER_PID}" ]] && kill -0 "${ROUTER_PID}" 2>/dev/null; then
    kill "${ROUTER_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

RUN_LOG="${RUN_LOG_DIR}/eval.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "========================================"
echo "  Terminal-RL Eval: ${RUN_ID}"
echo "  Suite:   ${EVAL_SUITE}"
echo "  Ckpt:    ${EVAL_CKPT} load=${LOAD_CKPT} step=${CKPT_STEP_ARGS[*]:-latest}"
echo "  Served:  HF_CKPT=${HF_CKPT} (--debug-rollout-only SGLang source)"
echo "  Model:   revision=${HF_MODEL_REVISION} manifest=${HF_MODEL_MANIFEST_SHA256}"
echo "  Profile: ${SWEBENCH_AGENT_PROFILE} sampling=${EVAL_TEMPERATURE}/${EVAL_TOP_P}/${EVAL_TOP_K}/${EVAL_MIN_P} context=${EVAL_MAX_PROMPT_LEN}+${EVAL_MAX_RESPONSE_LEN}/${SGLANG_CONTEXT_LENGTH:-model-default}"
echo "  Data:    ${EVAL_PROMPT_DATA[*]}"
echo "  Topology: Megatron actor=${ACTOR_WORLD_SIZE}/TP${MEGATRON_TP_SIZE}; SGLang=${ROLLOUT_GPUS} GPUs, $((ROLLOUT_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE)) engines x TP${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "  Metrics: ${TERMINAL_METRICS_JSONL}"
echo "========================================"

require_cmd curl

if [[ "${EVAL_SUITE}" == "sweverified" ]]; then
  log "SWE-bench timeouts: image=${TERMINAL_ENSURE_IMAGE_TIMEOUT}s reset_session=${TERMINAL_RESET_SESSION_TIMEOUT}s eval=${TERMINAL_EVAL_TIMEOUT}s router=${ROUTER_FORWARD_TIMEOUT}s client=${ENV_RESET_HTTP_TIMEOUT}s heartbeat=${ENV_HEARTBEAT_HTTP_TIMEOUT}s retries=${ROUTER_FORWARD_RETRIES} eval_concurrency=${EVAL_MAX_CONCURRENCY} worker_builds=${SWEBENCH_WORKER_MAX_CONCURRENT_BUILDS}"
  log "Terminal env retries: http=${ENV_HTTP_MAX_RETRIES} allocate=${ENV_ALLOCATE_MAX_RETRIES} reset=${ENV_RESET_MAX_RETRIES} exec_tool=${ENV_EXEC_TOOL_MAX_RETRIES} evaluate=${ENV_EVALUATE_MAX_RETRIES} close=${ENV_CLOSE_MAX_RETRIES}"
  log "SWE-bench grading: deferred=${SWEBENCH_DEFER_GRADING} (1=prediction export now, pinned official harness later)"
  log "SWE-bench dataset: rows=${SWEBENCH_DATASET_ROWS} unique_ids=${SWEBENCH_DATASET_UNIQUE_IDS} converted_sha256=${SWEBENCH_DATASET_SHA256} official_sha256=${SWEBENCH_OFFICIAL_DATA_SHA256} formal=${FORMAL_SWEBENCH_VERIFIED}"
fi

if [[ "${EVAL_DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] CPU path smoke only; skip process cleanup, router, Ray, and SGLang."
  echo "[DRY_RUN] REPO_ROOT=${REPO_ROOT}"
  echo "[DRY_RUN] SCRIPT_DIR=${SCRIPT_DIR}"
  echo "[DRY_RUN] TRAIN_PYTHON=${TRAIN_PYTHON}"
  echo "[DRY_RUN] CUSTOM_CONFIG_PATH=${CUSTOM_CONFIG_PATH}"
  echo "[DRY_RUN] LOAD_CKPT=${LOAD_CKPT}"
  echo "[DRY_RUN] HF_CKPT=${HF_CKPT}"
  echo "[DRY_RUN] REF_LOAD=${REF_LOAD}"
  echo "[DRY_RUN] Megatron actor world=${ACTOR_WORLD_SIZE} tp=${MEGATRON_TP_SIZE}"
  echo "[DRY_RUN] SGLang rollout_gpus=${ROLLOUT_GPUS} gpus_per_engine=${ROLLOUT_NUM_GPUS_PER_ENGINE} engines=$((ROLLOUT_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE))"
  echo "[DRY_RUN] profile=${SWEBENCH_AGENT_PROFILE} max_turn=${MAX_TURN} max_total_tokens=${TERMINAL_MAX_TOTAL_TOKENS}"
  echo "[DRY_RUN] SGLang context_length=${SGLANG_CONTEXT_LENGTH:-model-default} model_override=${SGLANG_JSON_MODEL_OVERRIDE_ARGS:-none}"
  for ((i=1; i<${#EVAL_PROMPT_DATA[@]}; i+=2)); do
    echo "[DRY_RUN] dataset ${EVAL_PROMPT_DATA[$((i-1))]}=${EVAL_PROMPT_DATA[$i]}"
  done
  [[ -x "${TRAIN_PYTHON}" ]] || { echo "[ERROR] TRAIN_PYTHON is not executable: ${TRAIN_PYTHON}"; exit 1; }
  [[ -f "${CUSTOM_CONFIG_PATH}" ]] || { echo "[ERROR] custom config not generated: ${CUSTOM_CONFIG_PATH}"; exit 1; }
  [[ -e "${LOAD_CKPT}" ]] || { echo "[ERROR] LOAD_CKPT not found: ${LOAD_CKPT}"; exit 1; }
  [[ -e "${HF_CKPT}" ]] || { echo "[ERROR] HF_CKPT not found: ${HF_CKPT}"; exit 1; }
  [[ -e "${REF_LOAD}" ]] || { echo "[ERROR] REF_LOAD not found: ${REF_LOAD}"; exit 1; }
  echo "[DRY_RUN] path smoke passed. Run dir: ${RUN_DIR}"
  exit 0
fi

if [[ "${CLEANUP_PROCESSES:-1}" == "1" ]]; then
  CLEANUP_TIMEOUT_SECS="${CLEANUP_TIMEOUT_SECS:-30}"
  pkill -9 sglang || true
  if command -v timeout >/dev/null 2>&1; then
    timeout --kill-after=5 "${CLEANUP_TIMEOUT_SECS}" ray stop --force || true
  else
    ray stop --force || true
  fi
  pkill -9 ray || true
  pkill -9 -f "terminal-rl.router_server" || true
  sleep 2
fi

if [[ "${INCLUDES_REMOTE_ENV}" == "1" ]]; then
  ROUTER_LOG="${RUN_LOG_DIR}/router.log"
  ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
  ROUTER_PORT="${ROUTER_PORT:-${ENV_SERVER_PORT}}"
  CHECK_HOST="${CHECK_HOST:-127.0.0.1}"
  CHECK_WAIT_SECS="${CHECK_WAIT_SECS:-300}"
  log "Starting router on ${ROUTER_HOST}:${ROUTER_PORT} -> ${WORKER_URLS}"
  (
    cd "${REPO_ROOT}"
    "${TRAIN_PYTHON}" -m terminal-rl.router_server \
      --host "${ROUTER_HOST}" --port "${ROUTER_PORT}" --workers "${WORKER_URLS}" \
      > "${ROUTER_LOG}" 2>&1 &
    echo $! > "${RUN_LOG_DIR}/router.pid"
  )
  ROUTER_PID="$(cat "${RUN_LOG_DIR}/router.pid")"
  ROUTER_READY=0
  for ((i=1; i<=CHECK_WAIT_SECS; i++)); do
    if curl -fsS --noproxy '*' "http://${CHECK_HOST}:${ROUTER_PORT}/readyz" >/dev/null 2>&1; then
      log "router and at least one env worker ready (attempt ${i})"
      ROUTER_READY=1
      break
    fi
    if ! kill -0 "${ROUTER_PID}" 2>/dev/null; then
      echo "[ERROR] router process exited before readiness. Log tail:"
      tail -n 120 "${ROUTER_LOG}" || true
      exit 1
    fi
    sleep 1
  done
  if [[ "${ROUTER_READY}" != "1" ]]; then
    echo "[ERROR] router did not become ready within ${CHECK_WAIT_SECS}s. Log tail:"
    tail -n 120 "${ROUTER_LOG}" || true
    exit 1
  fi

  if [[ "${EVAL_SUITE}" == "sweverified" ]]; then
    ROUTER_STATUS_PATH="${RUN_LOG_DIR}/router_status_preflight.json"
    if ! curl -fsS --noproxy '*' \
      "http://${CHECK_HOST}:${ROUTER_PORT}/status" > "${ROUTER_STATUS_PATH}"; then
      echo "[ERROR] failed to query router worker status after readiness" >&2
      exit 1
    fi
    "${TRAIN_PYTHON}" - \
      "${ROUTER_STATUS_PATH}" \
      "${SWEBENCH_WORKER_MAX_CONCURRENT_BUILDS}" \
      "${EVAL_MAX_CONCURRENCY}" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
expected_builds = int(sys.argv[2])
expected_tasks = int(sys.argv[3])
payload = json.loads(status_path.read_text(encoding="utf-8"))
workers = [item for item in payload.get("workers", []) if item.get("ok") is True]
if not workers:
    raise SystemExit("router status contains no healthy SWE-bench worker")
for worker in workers:
    pool = worker.get("pool") or {}
    image = pool.get("image_preparation") or {}
    actual_builds = int(image.get("max_concurrent", -1))
    max_tasks = int(pool.get("max_tasks", -1))
    if actual_builds != expected_builds:
        raise SystemExit(
            "worker build concurrency mismatch: "
            f"worker={worker.get('url')} actual={actual_builds} "
            f"expected={expected_builds}"
        )
    if max_tasks < expected_tasks:
        raise SystemExit(
            "worker task capacity is below formal eval concurrency: "
            f"worker={worker.get('url')} max_tasks={max_tasks} "
            f"required>={expected_tasks}"
        )
print(
    f"validated {len(workers)} worker(s): "
    f"max_concurrent_builds={expected_builds} max_tasks>={expected_tasks}"
)
PY
  fi
fi

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
if [[ "${NVLINK_COUNT:-0}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --load "${LOAD_CKPT}"
  "${CKPT_STEP_ARGS[@]}"
  --rotary-base 1000000
)

EVAL_ARGS=(
  --eval-prompt-data "${EVAL_PROMPT_DATA[@]}"
  --eval-input-key task
  --n-samples-per-eval-prompt "${EVAL_N_SAMPLES}"
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  --eval-max-prompt-len "${EVAL_MAX_PROMPT_LEN}"
  --eval-temperature "${EVAL_TEMPERATURE}"
  --eval-top-p "${EVAL_TOP_P}"
  --eval-top-k "${EVAL_TOP_K}"
  --eval-min-p "${EVAL_MIN_P}"
)

ROLLOUT_ARGS=(
  --prompt-data "${EVAL_PROMPT_DATA[1]}"
  --input-key task
  --reward-key score
  --num-rollout 1
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt 1
  --rollout-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${EVAL_MAX_CONTEXT_LEN}"
  --rollout-temperature "${EVAL_TEMPERATURE}"
  --rollout-top-p "${EVAL_TOP_P}"
  --rollout-top-k "${EVAL_TOP_K}"
  --num-steps-per-rollout 1
  --balance-data
)

PERF_ARGS=(
  --tensor-model-parallel-size "${MEGATRON_TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 1024
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.6}"
)
if [[ -n "${SGLANG_CONTEXT_LENGTH}" ]]; then
  SGLANG_ARGS+=(--sglang-context-length "${SGLANG_CONTEXT_LENGTH}")
fi
if [[ -n "${SGLANG_JSON_MODEL_OVERRIDE_ARGS}" ]]; then
  SGLANG_ARGS+=(
    --sglang-json-model-override-args "${SGLANG_JSON_MODEL_OVERRIDE_ARGS}"
  )
fi
if [[ "${EVAL_DETERMINISTIC}" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-enable-deterministic-inference)
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-backend flash
  --no-gradient-accumulation-fusion
  --debug-rollout-only
  --seed "${EVAL_SEED}"
  --rollout-seed "${ROLLOUT_SEED}"
)

CUSTOM_ARGS=(
  # Keep eval_function_path at slime's rollout-level default. generate.generate
  # is a per-sample async function consumed by sglang_rollout via this hook.
  --custom-generate-function-path generate.generate
  --custom-eval-rollout-log-function-path rollout_log.eval_rollout_log
  --custom-config-path "${CUSTOM_CONFIG_PATH}"
)

EVAL_ONLY_ARGS=(
  --actor-num-nodes "${ACTOR_NUM_NODES}"
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
  --rollout-num-gpus "${ROLLOUT_GPUS}"
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${EVAL_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${CUSTOM_ARGS[@]}"
)

SGLANG_JSON_MODEL_OVERRIDE_CONFIG="${SGLANG_JSON_MODEL_OVERRIDE_ARGS:-null}"
cat > "${RUN_DIR}/config/run_config.json" <<CFGEOF
{
  "run_id": "${RUN_ID}",
  "eval_ckpt": "${EVAL_CKPT}",
  "served_model_source": "hf_checkpoint",
  "hf_checkpoint": "${HF_CKPT}",
  "hf_model_revision": "${HF_MODEL_REVISION}",
  "hf_model_manifest_sha256": "${HF_MODEL_MANIFEST_SHA256}",
  "hf_config_sha256": "${HF_CONFIG_SHA256}",
  "code_revision": "${CODE_REVISION}",
  "code_runtime_sha256": "${CODE_RUNTIME_SHA256}",
  "load_ckpt": "${LOAD_CKPT}",
  "eval_suite": "${EVAL_SUITE}",
  "eval_prompt_data": "${EVAL_PROMPT_DATA[*]}",
  "eval_limit": "${EVAL_LIMIT}",
  "num_gpus": "${NUM_GPUS}",
  "rollout_gpus": "${ROLLOUT_GPUS}",
  "rollout_num_gpus_per_engine": "${ROLLOUT_NUM_GPUS_PER_ENGINE}",
  "sglang_engine_count": "$((ROLLOUT_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE))",
  "actor_world_size": "${ACTOR_WORLD_SIZE}",
  "megatron_tensor_model_parallel_size": "${MEGATRON_TP_SIZE}",
  "swebench_agent_profile": "${SWEBENCH_AGENT_PROFILE}",
  "harness_option": "${HARNESS_OPTION}",
  "max_turn": "${MAX_TURN}",
  "terminal_max_total_tokens": "${TERMINAL_MAX_TOTAL_TOKENS}",
  "eval_n_samples": "${EVAL_N_SAMPLES}",
  "eval_temperature": "${EVAL_TEMPERATURE}",
  "eval_top_p": "${EVAL_TOP_P}",
  "eval_top_k": "${EVAL_TOP_K}",
  "eval_min_p": "${EVAL_MIN_P}",
  "eval_seed": "${EVAL_SEED}",
  "rollout_seed": "${ROLLOUT_SEED}",
  "deterministic_inference": "${EVAL_DETERMINISTIC}",
  "sglang_request_timeout": "${SGLANG_REQUEST_TIMEOUT}",
  "thinking_mode": true,
  "model_name_or_path": "${SWEBENCH_MODEL_NAME_OR_PATH}",
  "eval_max_prompt_len": "${EVAL_MAX_PROMPT_LEN}",
  "eval_max_response_len": "${EVAL_MAX_RESPONSE_LEN}",
  "eval_max_context_len": "${EVAL_MAX_CONTEXT_LEN}",
  "sglang_context_length": "${SGLANG_CONTEXT_LENGTH}",
  "sglang_json_model_override_args": ${SGLANG_JSON_MODEL_OVERRIDE_CONFIG},
  "eval_max_concurrency": "${EVAL_MAX_CONCURRENCY}",
  "swebench_worker_max_concurrent_builds": "${SWEBENCH_WORKER_MAX_CONCURRENT_BUILDS:-}",
  "env_remote_max_active_tasks": "${ENV_REMOTE_MAX_ACTIVE_TASKS:-}",
  "env_remote_admission_timeout": "${ENV_REMOTE_ADMISSION_TIMEOUT:-}",
  "env_allocate_http_timeout": "${ENV_ALLOCATE_HTTP_TIMEOUT:-}",
  "env_heartbeat_http_timeout": "${ENV_HEARTBEAT_HTTP_TIMEOUT:-}",
  "env_reset_http_timeout": "${ENV_RESET_HTTP_TIMEOUT:-}",
  "env_exec_tool_http_timeout": "${ENV_EXEC_TOOL_HTTP_TIMEOUT:-}",
  "env_evaluate_http_timeout": "${ENV_EVALUATE_HTTP_TIMEOUT:-}",
  "env_close_http_timeout": "${ENV_CLOSE_HTTP_TIMEOUT:-}",
  "env_http_max_retries": "${ENV_HTTP_MAX_RETRIES:-}",
  "env_allocate_max_retries": "${ENV_ALLOCATE_MAX_RETRIES:-}",
  "env_reset_max_retries": "${ENV_RESET_MAX_RETRIES:-}",
  "env_exec_tool_max_retries": "${ENV_EXEC_TOOL_MAX_RETRIES:-}",
  "env_evaluate_max_retries": "${ENV_EVALUATE_MAX_RETRIES:-}",
  "env_close_max_retries": "${ENV_CLOSE_MAX_RETRIES:-}",
  "terminal_ensure_image_timeout": "${TERMINAL_ENSURE_IMAGE_TIMEOUT:-}",
  "terminal_reset_session_timeout": "${TERMINAL_RESET_SESSION_TIMEOUT:-}",
  "terminal_close_session_timeout": "${TERMINAL_CLOSE_SESSION_TIMEOUT:-}",
  "terminal_eval_timeout": "${TERMINAL_EVAL_TIMEOUT:-}",
  "router_forward_timeout": "${ROUTER_FORWARD_TIMEOUT:-}",
  "router_forward_retries": "${ROUTER_FORWARD_RETRIES:-}",
  "router_pressure_cooldown": "${ROUTER_PRESSURE_COOLDOWN:-}",
  "formal_swebench_verified": "${FORMAL_SWEBENCH_VERIFIED}",
  "swebench_defer_grading": "${SWEBENCH_DEFER_GRADING}",
  "swebench_dataset_rows": "${SWEBENCH_DATASET_ROWS}",
  "swebench_dataset_unique_ids": "${SWEBENCH_DATASET_UNIQUE_IDS}",
  "swebench_dataset_sha256": "${SWEBENCH_DATASET_SHA256}",
  "swebench_official_dataset_path": "${SWEBENCH_RUN_OFFICIAL_DATA}",
  "swebench_official_dataset_sha256": "${SWEBENCH_OFFICIAL_DATA_SHA256}",
  "metrics_jsonl": "${TERMINAL_METRICS_JSONL}"
}
CFGEOF
"${TRAIN_PYTHON}" -m json.tool "${RUN_DIR}/config/run_config.json" >/dev/null

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_terminal_rl_eval}"
mkdir -p "${RAY_TMPDIR}"

log "ray start --head ..."
ray start --head \
  --node-ip-address "${NODE_IP}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" \
  --temp-dir "${RAY_TMPDIR}"

for i in {1..40}; do
  if curl -fsS --max-time 3 "http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT:-8265}/api/version" >/dev/null 2>&1; then
    log "Ray dashboard ready (attempt $i)"
    break
  fi
  sleep 3
done

RUNTIME_PYTHONPATH="${MEGATRON_DIR}:${REPO_ROOT}:${SLIME_DIR}:${SCRIPT_DIR}"
RUNTIME_ENV_JSON="$("${TRAIN_PYTHON}" - <<PY
import json
import os

keys = [
    "PATH", "LD_LIBRARY_PATH", "MASTER_ADDR", "PYTORCH_CUDA_ALLOC_CONF",
    "USE_REMOTE_ENV", "ENV_SERVER_URL", "AGENT_SAFETYBENCH_REMOTE_ENV",
    "AGENTHARM_REMOTE_ENV", "AGENT_SAFETYBENCH_ROOT", "AGENTHARM_ROOT",
    "SAFETY_BENCH_REWARD", "AGENTHARM_REWARD", "SETA_SAFETY",
    "SAFETY_REWARD_COEF", "TERMINAL_STRUCTURED_METRICS",
    "TERMINAL_METRICS_JSONL", "TERMINAL_SAVE_TRAJ_DIR", "RUN_DIR",
    "RUN_ID", "RUN_LOG_DIR", "HARNESS_OPTION", "NO_PROXY", "no_proxy",
    "SLIME_RAY_PLACEMENT_GPU_PROBE", "EVAL_ROLLOUT_MAX_CONCURRENCY",
    "ENV_REMOTE_MAX_ACTIVE_TASKS", "ENV_REMOTE_ADMISSION_TIMEOUT",
    "ENV_REMOTE_ADMISSION_LOG_INTERVAL", "ENV_ALLOCATE_HTTP_TIMEOUT",
    "ENV_HEARTBEAT_HTTP_TIMEOUT",
    "ENV_RESET_HTTP_TIMEOUT", "ENV_EXEC_TOOL_HTTP_TIMEOUT",
    "ENV_CLOSE_HTTP_TIMEOUT", "ENV_HTTP_MAX_RETRIES",
    "ENV_ALLOCATE_MAX_RETRIES",
    "ENV_RESET_MAX_RETRIES", "ENV_EXEC_TOOL_MAX_RETRIES", "ENV_CLOSE_MAX_RETRIES",
    "ENV_EVALUATE_MAX_RETRIES", "ENV_EVALUATE_HTTP_TIMEOUT",
    "TERMINAL_ENSURE_IMAGE_TIMEOUT", "TERMINAL_RESET_SESSION_TIMEOUT",
    "TERMINAL_CLOSE_SESSION_TIMEOUT", "TERMINAL_EVAL_TIMEOUT",
    "ROUTER_FORWARD_TIMEOUT", "ROUTER_FORWARD_RETRIES",
    "SWEBENCH_MODEL_NAME_OR_PATH", "SWEBENCH_RESULTS_DIR",
    "SWEBENCH_DEFER_GRADING",
    "SWEBENCH_EVAL_DATA_PATH", "SGLANG_REQUEST_TIMEOUT",
]
env = {}
for key in keys:
    value = os.environ.get(key)
    if value is not None and value != "":
        env[key] = value
env.update({
    "PYTHONPATH": "${RUNTIME_PYTHONPATH}",
    "PYTHONUNBUFFERED": "1",
    "PYTHONFAULTHANDLER": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
})
print(json.dumps({"env_vars": env}))
PY
)"

RAY_JOB_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-terminal_rl_eval_${CKPT_LABEL}_${EVAL_SUITE}_$(date +%Y%m%d_%H%M%S)}"
RAY_ADDR="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT:-8265}"

write_eval_summary() {
  "${TRAIN_PYTHON}" - \
    "${TERMINAL_METRICS_JSONL}" \
    "${RUN_DIR}/eval_summary.json" \
    "${RUN_DIR}/eval_summary.tsv" \
    "${RUN_DIR}/config/run_config.json" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
json_out = Path(sys.argv[2])
tsv_out = Path(sys.argv[3])
config_path = Path(sys.argv[4])


def norm(value):
    return str(value or "").strip().lower().replace("-", "_")


def benchmark_name(record):
    dataset = norm(record.get("dataset"))
    sources = {norm(item) for item in (record.get("source_datasets") or [])}
    labels = {dataset, *sources}
    if dataset == "mixed_all":
        return "mixed-all"
    if any(label == "seta" or label.startswith("seta_") for label in labels):
        return "seta"
    if any(label == "agentharm" or label.startswith("agentharm") or label == "ah" for label in labels):
        return "agentharm"
    if labels.intersection({"safety", "security", "agent_safetybench", "asb", "mcpsafety", "harmbench"}):
        return "agent_safetybench"
    if labels.intersection({"sweverified", "swe_verified"}):
        return "sweverified"
    return dataset or "unknown"


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


config = {}
if config_path.exists():
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        config = {"config_parse_error": str(exc)}

records = []
if metrics_path.exists():
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("schema") != "terminal_rl.per_dataset_metrics.v1":
            continue
        if record.get("phase") != "eval":
            continue
        benchmark = benchmark_name(record)
        raw_reward_scale = record.get("raw_reward_scale")
        raw_reward_semantics = record.get("raw_reward_semantics")
        if benchmark in {"agent_safetybench", "agentharm"} and raw_reward_scale in {None, "", "unknown"}:
            raw_reward_scale = "direct_safety_score"
            raw_reward_semantics = "dataset reward-model score, not a 0/1 pass rate"
        if benchmark == "sweverified":
            raw_reward_scale = "deferred_official_grading"
            raw_reward_semantics = (
                "prediction export only; use the pinned official harness report "
                "for resolved scores"
            )
        row = {
            "benchmark": benchmark,
            "dataset": record.get("dataset"),
            "source_datasets": record.get("source_datasets") or [],
            "reward_task": record.get("reward/task"),
            "reward_total": record.get("reward/total"),
            "reward_raw": record.get("reward/raw"),
            "test_acc": record.get("test_acc"),
            "sample_count": record.get("sample_count"),
            "trainable_count": record.get("trainable_count"),
            "completed": record.get("completed"),
            "failed": record.get("failed"),
            "aborted": record.get("aborted"),
            "truncated": record.get("truncated"),
            "truncated_fraction": record.get("truncated_fraction"),
            "response_length": record.get("response_length"),
            "raw_reward_scale": raw_reward_scale,
            "raw_reward_semantics": raw_reward_semantics,
        }
        records.append(row)

payload = {
    "run_id": config.get("run_id"),
    "eval_ckpt": config.get("eval_ckpt"),
    "load_ckpt": config.get("load_ckpt"),
    "eval_suite": config.get("eval_suite"),
    "eval_limit": config.get("eval_limit"),
    "metrics_jsonl": str(metrics_path),
    "benchmarks": records,
}

json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

headers = [
    "benchmark",
    "dataset",
    "source_datasets",
    "reward_task",
    "reward_total",
    "reward_raw",
    "test_acc",
    "sample_count",
    "trainable_count",
    "completed",
    "failed",
    "aborted",
    "truncated",
    "truncated_fraction",
    "response_length",
    "raw_reward_scale",
]
lines = ["\t".join(headers)]
for row in records:
    values = []
    for key in headers:
        value = row.get(key)
        if key == "source_datasets":
            value = ",".join(value or [])
        values.append(fmt(value))
    lines.append("\t".join(values))
tsv_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote eval summary: {json_out}")
print(f"Wrote eval summary: {tsv_out}")
if records:
    print("benchmark\tdataset\treward_task\tsample_count\tfailed\ttruncated")
    for row in records:
        print(
            "\t".join(
                [
                    fmt(row.get("benchmark")),
                    fmt(row.get("dataset")),
                    fmt(row.get("reward_task")),
                    fmt(row.get("sample_count")),
                    fmt(row.get("failed")),
                    fmt(row.get("truncated")),
                ]
            )
        )
else:
    print(f"WARNING: no eval metric records found in {metrics_path}", file=sys.stderr)
PY
}

log "Submitting Ray job ${RAY_JOB_SUBMISSION_ID}"
ray job submit --address="${RAY_ADDR}" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_PYTHON}" -u "${SLIME_DIR}/eval_only.py" \
  "${EVAL_ONLY_ARGS[@]}"

set +e
ray job logs --address="${RAY_ADDR}" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record | tee -a "${RUN_LOG_DIR}/ray_job.log"
RAY_LOG_EXIT=$?
RAY_STATUS_OUTPUT="$(ray job status --address="${RAY_ADDR}" "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)"
echo "${RAY_STATUS_OUTPUT}"
set -e

if echo "${RAY_STATUS_OUTPUT}" | tr '[:upper:]' '[:lower:]' | grep -q "succeeded"; then
  if [[ "${EVAL_SUITE}" == "sweverified" ]]; then
    "${TRAIN_PYTHON}" - \
      "${SWEBENCH_RESULTS_DIR}" \
      "${SWEBENCH_EVAL_DATA_PATH}" \
      "${SWEBENCH_DEFER_GRADING}" <<'PY'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
dataset_path = Path(sys.argv[2])
grading_deferred = sys.argv[3] == "1"
expected = sum(1 for line in dataset_path.open(encoding="utf-8") if line.strip())
summary_path = results_dir / "score_summary.json"
if not summary_path.is_file():
    raise SystemExit(f"missing SWE-bench score summary: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if int(summary.get("total", -1)) != expected:
    raise SystemExit(
        f"SWE-bench report denominator mismatch: report={summary.get('total')} expected={expected}"
    )
if int(summary.get("submitted", -1)) != expected:
    raise SystemExit(
        "SWE-bench prediction coverage mismatch: "
        f"submitted={summary.get('submitted')} expected={expected}; "
        "this is not a complete 500-instance evaluation"
    )
if int(summary.get("incomplete", -1)) != 0:
    raise SystemExit(
        "SWE-bench prediction set is missing official instance IDs: "
        f"count={summary.get('incomplete')}"
    )
if int(summary.get("unexpected", -1)) != 0:
    raise SystemExit(
        "SWE-bench prediction set contains non-dataset instance IDs: "
        f"count={summary.get('unexpected')} ids={summary.get('unexpected_ids', [])[:20]}"
    )
if int(summary.get("technical_failures", -1)) != 0:
    ids = summary.get("technical_failure_ids", [])
    raise SystemExit(
        "SWE-bench prediction generation had infrastructure failures: "
        f"count={summary.get('technical_failures')} ids={ids[:20]}"
    )
if not grading_deferred:
    raise SystemExit("SWE-bench Verified must use deferred official grading")
pending = int(summary.get("pending_official_grading", -1))
pending_ids = set(summary.get("pending_official_grading_ids", []))
expected_ids = set()
for line in dataset_path.open(encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    metadata = row.get("metadata") or {}
    instance_id = metadata.get("swe_instance_id") or metadata.get("task_name")
    if instance_id:
        expected_ids.add(str(instance_id))
if summary.get("grading_deferred") is not True or pending != expected:
    raise SystemExit(
        "SWE-bench deferred grading coverage mismatch: "
        f"pending={pending} expected={expected}"
    )
if pending_ids != expected_ids:
    raise SystemExit(
        "SWE-bench deferred grading ID set mismatch: "
        f"missing={sorted(expected_ids - pending_ids)[:20]} "
        f"unexpected={sorted(pending_ids - expected_ids)[:20]}"
    )
if summary.get("authoritative_score") is not None:
    raise SystemExit("prediction export must not fabricate an official score")
print(
    "SWE-bench Verified prediction generation complete: "
    f"{summary['submitted']}/{summary['total']} patches exported; "
    "authoritative grading is pending"
)
print(f"Official-format artifacts: {results_dir}")
print("Run the pinned official SWE-bench harness on predictions.jsonl for the authoritative score.")
PY
  fi
  if ! write_eval_summary; then
    log "Eval succeeded, but summary generation failed. Metrics: ${TERMINAL_METRICS_JSONL}"
  fi
  if [[ "${EVAL_SUITE}" == "sweverified" ]]; then
    if [[ "${SWEBENCH_DEFER_GRADING}" == "1" ]]; then
      log "SWE-bench prediction generation succeeded; pinned authoritative harness grading is required. Run dir: ${RUN_DIR}"
    else
      log "SWE-bench prediction generation and integrated precheck succeeded; authoritative harness grading is still required. Run dir: ${RUN_DIR}"
    fi
  else
    log "Eval succeeded. Run dir: ${RUN_DIR}"
  fi
  exit 0
fi

log "Eval failed (logs exit: ${RAY_LOG_EXIT}). Inspect ${RUN_LOG}"
exit 1
