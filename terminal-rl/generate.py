from __future__ import annotations

import hashlib
import fcntl
import json
import logging
import math
import os
import re
import shutil
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from agent.prm_agent import TerminalPRMAgent
from clawsentry_client import ClawSentryClient
from custom_types import (
    Interaction,
    RunContext,
    TaskSpec,
    TaskTimeouts,
    TurnContext,
    TurnResult,
)
from inference_client import SGLangTurnClient
from agent_runner import create_agent_runner, normalize_harness_option
from env_client import TerminalEnvClient
from agent57_episodic_memory import create_episodic_memory_backend
from explore_agent57_lite import (
    coarse_observation_fingerprint as _agent57_coarse_observation_fingerprint,
    coarse_observation_label as _agent57_coarse_observation_label,
    compute_ngu_lite_bonus as _agent57_compute_ngu_lite_bonus,
    compute_lifelong_bonus as _agent57_compute_lifelong_bonus,
    config_from_env as _agent57_config_from_env,
    exit_code_bucket as _agent57_exit_code_bucket,
    record_arm_event as _agent57_record_arm_event,
)
from safety_reward import (
    DEFAULT_ZERO_THRESHOLD as _SAFETY_ZERO_THRESHOLD,
    broadcast_to_turns as _safety_broadcast,
    per_turn_score as _safety_per_turn_score,
    trajectory_score as _safety_trajectory_score,
)

logger = logging.getLogger(__name__)

_DIRECT_SCORE_DATA_SOURCES = {"agent_safetybench", "agentharm"}
_AGENT57_CONFIG = _agent57_config_from_env()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.4f", name, raw, default)
        return default


async def _await_with_optional_timeout(awaitable, timeout: float, *, op_name: str):
    if timeout <= 0:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{op_name} timed out after {timeout:.1f}s") from exc


_REMOTE_ENV_CONDITION: asyncio.Condition | None = None
_REMOTE_ENV_ACTIVE_BY_TASK: dict[str, int] = {}
_REMOTE_ENV_ACTIVE_TOTAL = 0
_REMOTE_ENV_CLOSE_SEMAPHORE: asyncio.Semaphore | None = None
_REMOTE_ENV_CLOSE_LIMIT: int | None = None
_REMOTE_ENV_CLOSE_SEMAPHORE_LOCK: asyncio.Lock | None = None  # P1 fix: Add lock for semaphore recreation


def _uses_local_agent_safetybench_env(task_meta: Dict[str, Any] | None) -> bool:
    return (
        isinstance(task_meta, dict)
        and task_meta.get("data_source") == "agent_safetybench"
        and os.getenv("AGENT_SAFETYBENCH_REMOTE_ENV", "0") != "1"
    )


def _uses_local_agentharm_env(task_meta: Dict[str, Any] | None) -> bool:
    return (
        isinstance(task_meta, dict)
        and task_meta.get("data_source") == "agentharm"
        and os.getenv("AGENTHARM_REMOTE_ENV", "0") != "1"
    )


def _uses_remote_terminal_env(task_meta: Dict[str, Any] | None) -> bool:
    return not (
        _uses_local_agent_safetybench_env(task_meta)
        or _uses_local_agentharm_env(task_meta)
    )


_TASK_CIRCUIT: dict[str, dict[str, Any]] = {}


def _task_circuit_enabled() -> bool:
    return _env_bool("ENV_TASK_CIRCUIT_BREAKER_ENABLED", True)


def _task_circuit_threshold() -> int:
    return max(1, _env_int("ENV_TASK_CIRCUIT_BREAKER_THRESHOLD", 2))


def _task_circuit_cooldown() -> float:
    return max(0.0, _env_float("ENV_TASK_CIRCUIT_BREAKER_COOLDOWN", 1800.0))


def _task_circuit_failure_is_relevant(exc: BaseException) -> bool:
    text = str(exc)
    return any(
        marker in text
        for marker in (
            "TASK_BUILD_FAILED",
            "WORKER_RESET_TIMEOUT",
            "env reset timed out",
            "reset timed out",
            "Docker image build failed",
            "dockerfile parse error",
            "RESET_IN_PROGRESS",
            "WORKER_RESET_CANCELLED",
            "WORKER_RESET_STALE",
        )
    )


def _task_circuit_open_reason(task_key: str) -> str | None:
    if not _task_circuit_enabled():
        return None
    state = _TASK_CIRCUIT.get(task_key)
    if not state:
        return None
    opened_until = float(state.get("opened_until", 0.0) or 0.0)
    if opened_until <= time.time():
        _TASK_CIRCUIT.pop(task_key, None)
        return None
    return str(state.get("reason") or "recent env failures")


def _task_circuit_record_success(task_key: str) -> None:
    if task_key:
        _TASK_CIRCUIT.pop(task_key, None)


def _task_circuit_record_failure(task_key: str, exc: BaseException) -> None:
    if not task_key or not _task_circuit_enabled():
        return
    if not _task_circuit_failure_is_relevant(exc):
        return
    now = time.time()
    state = _TASK_CIRCUIT.setdefault(
        task_key,
        {"count": 0, "opened_until": 0.0, "reason": ""},
    )
    state["count"] = int(state.get("count", 0) or 0) + 1
    reason = f"{type(exc).__name__}: {str(exc)[:300]}"
    state["reason"] = reason
    immediate = "TASK_BUILD_FAILED" in str(exc) or "dockerfile parse error" in str(exc)
    if immediate or int(state["count"]) >= _task_circuit_threshold():
        state["opened_until"] = now + _task_circuit_cooldown()
        logger.warning(
            "Opening task circuit breaker task_key=%s count=%s cooldown=%.1fs reason=%s",
            task_key,
            state["count"],
            _task_circuit_cooldown(),
            reason,
        )


def _remote_env_condition() -> asyncio.Condition:
    global _REMOTE_ENV_CONDITION
    if _REMOTE_ENV_CONDITION is None:
        _REMOTE_ENV_CONDITION = asyncio.Condition()
    return _REMOTE_ENV_CONDITION


def _remote_env_close_semaphore() -> asyncio.Semaphore | None:
    global _REMOTE_ENV_CLOSE_LIMIT, _REMOTE_ENV_CLOSE_SEMAPHORE, _REMOTE_ENV_CLOSE_SEMAPHORE_LOCK
    limit = _env_int("ENV_REMOTE_MAX_CONCURRENT_CLOSES", 8)
    if limit <= 0:
        return None
    # P1 fix: Use lock to prevent race condition during semaphore recreation
    if _REMOTE_ENV_CLOSE_SEMAPHORE_LOCK is None:
        _REMOTE_ENV_CLOSE_SEMAPHORE_LOCK = asyncio.Lock()
    # Note: This is not truly async-safe since we can't await here, but it prevents
    # the worst case of two semaphores coexisting. For full safety, callers should
    # cache the semaphore result at module init.
    if _REMOTE_ENV_CLOSE_SEMAPHORE is None or _REMOTE_ENV_CLOSE_LIMIT != limit:
        _REMOTE_ENV_CLOSE_LIMIT = limit
        _REMOTE_ENV_CLOSE_SEMAPHORE = asyncio.Semaphore(limit)
    return _REMOTE_ENV_CLOSE_SEMAPHORE


async def _acquire_remote_env_admission(
    task_key: str,
    *,
    log_tag: str,
) -> str | None:
    global _REMOTE_ENV_ACTIVE_TOTAL
    max_active_tasks = _env_int("ENV_REMOTE_MAX_ACTIVE_TASKS", 12)
    max_active_runs = _env_int("ENV_REMOTE_MAX_ACTIVE_RUNS", 0)
    max_runs_per_task = _env_int("ENV_REMOTE_MAX_RUNS_PER_TASK", 8)
    if max_active_tasks <= 0 and max_active_runs <= 0 and max_runs_per_task <= 0:
        return None

    timeout = _env_float("ENV_REMOTE_ADMISSION_TIMEOUT", 900.0)
    log_interval = max(5.0, _env_float("ENV_REMOTE_ADMISSION_LOG_INTERVAL", 30.0))
    condition = _remote_env_condition()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout if timeout > 0 else None
    last_log = 0.0

    async with condition:
        while True:
            active_for_task = _REMOTE_ENV_ACTIVE_BY_TASK.get(task_key, 0)
            active_tasks = len(_REMOTE_ENV_ACTIVE_BY_TASK)
            reasons: list[str] = []
            if (
                max_active_tasks > 0
                and active_for_task <= 0
                and active_tasks >= max_active_tasks
            ):
                reasons.append(f"active_tasks={active_tasks}/{max_active_tasks}")
            if max_active_runs > 0 and _REMOTE_ENV_ACTIVE_TOTAL >= max_active_runs:
                reasons.append(
                    f"active_runs={_REMOTE_ENV_ACTIVE_TOTAL}/{max_active_runs}"
                )
            if max_runs_per_task > 0 and active_for_task >= max_runs_per_task:
                reasons.append(
                    f"runs_per_task={active_for_task}/{max_runs_per_task}"
                )

            if not reasons:
                _REMOTE_ENV_ACTIVE_BY_TASK[task_key] = active_for_task + 1
                _REMOTE_ENV_ACTIVE_TOTAL += 1
                return task_key

            now = loop.time()
            if deadline is not None and now >= deadline:
                raise TimeoutError(
                    f"{log_tag} remote env admission timed out for task_key={task_key} "
                    f"after {timeout:.1f}s ({', '.join(reasons)})"
                )

            if now - last_log >= log_interval:
                logger.info(
                    "%s Waiting for remote env admission task_key=%s (%s)",
                    log_tag,
                    task_key,
                    ", ".join(reasons),
                )
                last_log = now

            wait_timeout = log_interval
            if deadline is not None:
                wait_timeout = min(wait_timeout, max(0.1, deadline - now))
            try:
                await asyncio.wait_for(condition.wait(), timeout=wait_timeout)
            except asyncio.TimeoutError:
                pass


async def _release_remote_env_admission(task_key: str | None) -> None:
    global _REMOTE_ENV_ACTIVE_TOTAL
    if not task_key:
        return
    condition = _remote_env_condition()
    async with condition:
        active_for_task = _REMOTE_ENV_ACTIVE_BY_TASK.get(task_key, 0)
        if active_for_task <= 1:
            _REMOTE_ENV_ACTIVE_BY_TASK.pop(task_key, None)
        else:
            _REMOTE_ENV_ACTIVE_BY_TASK[task_key] = active_for_task - 1
        _REMOTE_ENV_ACTIVE_TOTAL = max(0, _REMOTE_ENV_ACTIVE_TOTAL - 1)
        # P1 fix: Use notify(1) instead of notify_all() to reduce wake-up storm
        # Only one waiter can proceed anyway since we released exactly one slot
        condition.notify(1)


# ── Exploration: count-based intrinsic reward (MERCI simplified) ──────────────
_EXPLORE_INTRINSIC_ENABLED = _env_bool("EXPLORE_INTRINSIC_ENABLED", False)
_EXPLORE_INTRINSIC_COEF = _env_float("EXPLORE_INTRINSIC_COEF", 0.1)
_EXPLORE_INTRINSIC_SCHEDULE = os.getenv("EXPLORE_INTRINSIC_SCHEDULE", "constant").strip().lower()
_EXPLORE_INTRINSIC_DECAY_STEPS = _env_int("EXPLORE_INTRINSIC_DECAY_STEPS", 0)
_EXPLORE_INTRINSIC_REDUCER = os.getenv("EXPLORE_INTRINSIC_REDUCER", "sum").strip().lower()
if _EXPLORE_INTRINSIC_REDUCER not in {"sum", "mean"}:
    _EXPLORE_INTRINSIC_REDUCER = "sum"
_EXPLORE_SCORE_BONUS_COMPONENTS = os.getenv("EXPLORE_SCORE_BONUS_COMPONENTS", "legacy").strip().lower()
# Granularity for novelty hashing:
#   "raw"        = full command string (default, matches v1)
#   "signature"  = tool-call signature (cmd name + first 2 args), Agent57-style
#                  sub-goal/skill granularity per the LaMer/Agent57 analysis.
_EXPLORE_INTRINSIC_GRANULARITY = os.getenv("EXPLORE_INTRINSIC_GRANULARITY", "raw").strip().lower()
_EXPLORE_INTRINSIC_SCOPE = os.getenv("EXPLORE_INTRINSIC_SCOPE", "process").strip().lower()
_EXPLORE_AGENT57_EPISODIC_OBS_MODE = (
    os.getenv(
        "EXPLORE_AGENT57_EPISODIC_OBS_MODE",
        os.getenv("EXPLORE_AGENT57_LIFELONG_OBS_MODE", "fingerprint"),
    )
    .strip()
    .lower()
)
if _EXPLORE_AGENT57_EPISODIC_OBS_MODE not in {"fingerprint", "label", "none"}:
    _EXPLORE_AGENT57_EPISODIC_OBS_MODE = "fingerprint"
_EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN = _env_bool(
    "EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN",
    True,
)
_CMD_COUNTER: Dict[str, int] = {}  # process-level counter for command novelty
_AGENT57_LAST_EPISODIC_STATS: Dict[str, float] = {}
_AGENT57_LAST_EPISODIC_BY_TURN: Dict[int, float] = {}

# ── Exploration: LP-RND lifelong novelty (草案 C, zero-extra-param) ───────────
# Reuses the rollout_log_probs already computed by slime (no extra forward pass).
# Bonus is proportional to how surprised the *current* policy is by the trajectory:
# higher mean negative-logprob → more novel → larger bonus, clipped to [0, L].
# This is the LLM analog of RND: "how surprising is this trajectory under the
# current rollout policy?" implemented without maintaining a separate net.
_EXPLORE_LPRND_ENABLED = _env_bool("EXPLORE_LPRND_ENABLED", False)
_EXPLORE_LPRND_COEF = _env_float("EXPLORE_LPRND_COEF", 0.05)
_EXPLORE_LPRND_SCHEDULE = os.getenv("EXPLORE_LPRND_SCHEDULE", "constant").strip().lower()
_EXPLORE_LPRND_DECAY_STEPS = _env_int("EXPLORE_LPRND_DECAY_STEPS", 0)
_EXPLORE_LPRND_CLIP = _env_float("EXPLORE_LPRND_CLIP", 3.0)
_EXPLORE_LPRND_WARMUP = _env_int("EXPLORE_LPRND_WARMUP", 32)
# Running stats for normalization (process-level, updated online).
_LPRND_STATS = {"warmup": 0, "n": 0, "mean": 0.0, "m2": 0.0}

# ── T2PO-style turn uncertainty diagnostics ─────────────────────────────────
# Logging-only. This does not alter sampling or rewards. T2PO's original turn
# score uses logits entropy + max-logprob during generation; OpenClaw currently
# persists sampled-token log-probs, so this records a mean-logprob proxy.
_TURN_UNCERTAINTY_SCHEMA = "openclaw.t2po_turn_uncertainty"
_TURN_UNCERTAINTY_SCHEMA_VERSION = 1
_TURN_UNCERTAINTY_ENABLED = _env_bool("T2PO_TURN_UNCERTAINTY_LOGGING", True)
_TURN_UNCERTAINTY_WARMUP_TOKENS = max(
    0, _env_int("T2PO_TURN_UNCERTAINTY_WARMUP_TOKENS", 0)
)
_TURN_UNCERTAINTY_FINGERPRINT_TOKENS = max(
    1, _env_int("T2PO_TURN_UNCERTAINTY_FINGERPRINT_TOKENS", 32)
)
_TURN_LOW_PROGRESS_THRESHOLD = max(
    0.0, _env_float("T2PO_TURN_LOW_PROGRESS_THRESHOLD", 0.3)
)

# ── Exploration: CDE actor curiosity bonus (RLVR PPL bonus) ──────────────────
# Optional actor-side Curiosity-Driven Exploration bonus:
#   B_actor(q,o) = -mean_t log pi(o_t | o_<t, q)
#   r_hat = r + omega * min(|r| / kappa, alpha * B_actor)
#
# The cap is based on the pre-exploration task reward magnitude. That keeps this
# as a supplement to verifiable rewards and prevents empty/infra-failed rollouts
# with score=0 from receiving curiosity reward.
_EXPLORE_CDE_ACTOR_ENABLED = (
    os.getenv("EXPLORE_CDE_ACTOR_ENABLED", os.getenv("EXPLORE_CDE_ACTOR", "0")).strip().lower()
    in {"1", "true", "yes", "on"}
)
_EXPLORE_CDE_ACTOR_OMEGA = _env_float("EXPLORE_CDE_ACTOR_OMEGA", 0.05)
_EXPLORE_CDE_ACTOR_KAPPA = _env_float("EXPLORE_CDE_ACTOR_KAPPA", 2.0)
_EXPLORE_CDE_ACTOR_ALPHA = _env_float("EXPLORE_CDE_ACTOR_ALPHA", 0.1)
_EXPLORE_CDE_ACTOR_DECAY_STEPS = _env_int("EXPLORE_CDE_ACTOR_DECAY_STEPS", 0)
_EXPLORE_CDE_ACTOR_REWARD_GATE = os.getenv(
    "EXPLORE_CDE_ACTOR_REWARD_GATE", "nonzero"
).strip().lower()

# ── Exploration: multi-attempt reflection (LaMer-style) ───────────────────────
# When EXPLORE_RETRY_ATTEMPTS > 1, a failed rollout is annotated with a reflection
# block + traj_gamma reward shaping. Note: actual env restart is handled inside
# agent_runner; we surface the knobs here so the wrapper script and rollout_log
# can pick them up. Defaults (1 / 1.0) preserve baseline behavior.
_EXPLORE_RETRY_ATTEMPTS = _env_int("EXPLORE_RETRY_ATTEMPTS", 1)
_EXPLORE_RETRY_TRAJ_GAMMA = _env_float("EXPLORE_RETRY_TRAJ_GAMMA", 1.0)

# ── Exploration: safety pre-filter ────────────────────────────────────────────
_EXPLORE_SAFETY_FILTER_ENABLED = _env_bool("EXPLORE_SAFETY_FILTER_ENABLED", False)
_EXPLORE_SAFETY_FILTER_COEF = _env_float("EXPLORE_SAFETY_FILTER_COEF", -0.5)
_DANGER_RE = re.compile(
    r"rm\s+-[rfRF]+\s+/(?:\s|$)|"          # rm -rf /
    r"(?:curl|wget)[^|;]+\|\s*(?:bash|sh)|"  # curl|bash, wget|sh
    r"chmod\s+(?:0?7{2,3})\s+/|"             # chmod 777 /
    r">\s*/etc/(?:passwd|shadow|sudoers)|"
    r"cat\s+/etc/shadow|"
    r"eval\s+.*(?:rm\s+-[rfRF]+\s+/|curl|wget)|"
    r"(?:`|\$\()[^`)]*(?:rm\s+-[rfRF]+\s+/|curl|wget)[^`)]*(?:`|\))|"
    r":\(\)\s*\{\s*:\|:&\s*\}\s*;:",         # fork bomb
    re.IGNORECASE,
)


def _cmd_signature(cmd: str) -> str:
    """Skill-level signature of a command (cmd name + first 2 args) for novelty hashing.

    'signature' granularity reduces hash-collision-by-paraphrase: e.g., `ls -la /tmp`
    and `ls -al /tmp/` map to the same skill bucket, while `ls -la /etc` is distinct.
    This is the sub-goal granularity proposed in the Agent57→Agentic-RL migration analysis.
    """
    import shlex
    if not cmd or not cmd.strip():
        return "__empty__"

    def _normalize_part(part: str) -> str:
        part = part.strip()
        if len(part) > 2 and part.startswith("-") and not part.startswith("--"):
            # Normalize common short-flag permutations: -al and -la -> -al.
            return "-" + "".join(sorted(part[1:]))
        if part != "/" and "/" in part:
            return part.rstrip("/")
        return part

    try:
        parts = [_normalize_part(p) for p in shlex.split(cmd)[:3]]
        return "|".join(parts) if parts else "__empty__"
    except Exception:
        return cmd[:80]


def _stable_json(value: Any, limit: int = 512) -> str:
    try:
        text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text[:limit]


def _explore_len_bucket(text: str) -> str:
    size = len(text)
    if size == 0:
        return "len0"
    if size < 80:
        return "lenS"
    if size < 512:
        return "lenM"
    if size < 2048:
        return "lenL"
    return "lenXL"


def _explore_path_signature(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"/+", "/", text)
    if text != "/":
        text = text.rstrip("/")
    return text[:160]


def _explore_structured_tool_signature(
    tool_name: str,
    args: Any,
) -> tuple[str, str]:
    """Return a compact signature/family for non-command structured tools.

    Avoid hashing full payloads such as file contents into exploration keys. The
    key should capture the operation and target, not every byte written.
    """
    tool = str(tool_name or "tool").strip() or "tool"
    if not isinstance(args, dict):
        args_text = _stable_json(args)
        return f"{tool}|{args_text[:160]}", f"{tool}:structured"

    path_value = None
    for key in (
        "file_path",
        "path",
        "target_path",
        "filename",
        "dest",
        "destination",
        "repo_path",
    ):
        if args.get(key):
            path_value = args.get(key)
            break
    if path_value is not None:
        path = _explore_path_signature(path_value)
        ext = Path(path).suffix[:16] or "noext"
        parts = [tool, f"path:{path}", f"ext:{ext}"]
        if "content" in args:
            parts.append(f"content:{_explore_len_bucket(str(args.get('content') or ''))}")
        return "|".join(parts), f"{tool}:file"

    stable_keys = []
    for key in ("query", "url", "package", "name", "id"):
        value = args.get(key)
        if value:
            stable_keys.append(f"{key}:{str(value)[:80]}")
    if stable_keys:
        return "|".join([tool, *stable_keys]), f"{tool}:structured"

    return f"{tool}|schema:{','.join(sorted(str(k) for k in args.keys()))[:120]}", f"{tool}:structured"


def _explore_turn_bucket(turn_idx: Any) -> str:
    try:
        idx = int(turn_idx)
    except (TypeError, ValueError):
        return "turn_unknown"
    if idx <= 0:
        return "turn0"
    if idx <= 2:
        return "turn1_2"
    if idx <= 5:
        return "turn3_5"
    return "turn6p"


def _explore_observation_bucket(value: Any, mode: str) -> str:
    if mode == "none":
        return "obs_ignored"
    if mode == "label":
        return _agent57_coarse_observation_label(value)
    return _agent57_coarse_observation_fingerprint(value)


def _iter_explore_actions(turn_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract action strings used by intrinsic reward and safety diagnostics.

    Older code looked only at turn["command"], but current terminal-rl trajectories
    store most actions as structured tool_calls. Missing those calls makes command
    novelty and danger filtering silently no-op for real rollouts.
    """
    actions: List[Dict[str, Any]] = []
    for tr in turn_records or []:
        turn_idx = tr.get("turn_idx")
        legacy_cmd = str(tr.get("command", "") or "").strip()
        if legacy_cmd:
            result = tr.get("result") or tr.get("observation") or tr.get("output")
            actions.append(
                {
                    "tool_name": "shell",
                    "raw": legacy_cmd,
                    "signature": f"shell|{_cmd_signature(legacy_cmd)}",
                    "danger_text": legacy_cmd,
                    "turn_idx": str(turn_idx) if turn_idx is not None else "",
                    "turn_bucket": _explore_turn_bucket(turn_idx),
                    "result": result,
                    "obs_bucket": _explore_observation_bucket(
                        result,
                        _EXPLORE_AGENT57_EPISODIC_OBS_MODE,
                    ),
                    "exit_bucket": _agent57_exit_code_bucket(result),
                }
            )

        for call in tr.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool_name") or call.get("name") or "tool").strip() or "tool"
            args = call.get("args")
            if args is None:
                args = call.get("arguments")
            command_text = ""
            if isinstance(args, dict):
                for key in ("command", "cmd", "script", "code"):
                    value = args.get(key)
                    if value:
                        command_text = str(value).strip()
                        break
            elif args is not None:
                command_text = str(args).strip()

            args_text = _stable_json(args)
            raw = f"{tool_name}:{command_text or args_text}"
            if command_text:
                signature = f"{tool_name}|{_cmd_signature(command_text)}"
                action_family = ""
            else:
                signature, action_family = _explore_structured_tool_signature(
                    tool_name,
                    args,
                )
            result = call.get("result")
            if result is None:
                result = call.get("observation") or call.get("output")
            actions.append(
                {
                    "tool_name": tool_name,
                    "raw": raw,
                    "signature": signature,
                    "action_family": action_family,
                    "danger_text": command_text or args_text,
                    "turn_idx": str(turn_idx) if turn_idx is not None else "",
                    "turn_bucket": _explore_turn_bucket(turn_idx),
                    "result": result,
                    "obs_bucket": _explore_observation_bucket(
                        result,
                        _EXPLORE_AGENT57_EPISODIC_OBS_MODE,
                    ),
                    "exit_bucket": _agent57_exit_code_bucket(result),
                }
            )
    return actions


def _explore_agent57_episodic_state(action: Dict[str, Any]) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "tool": str(action.get("tool_name") or "tool"),
        "signature": str(action.get("signature") or action.get("raw") or "unknown"),
    }
    if _EXPLORE_AGENT57_EPISODIC_OBS_MODE != "none":
        state["observation"] = str(action.get("obs_bucket") or "no_result")
        state["exit"] = str(action.get("exit_bucket") or "exit_unknown")
    if _EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN:
        state["turn"] = str(action.get("turn_bucket") or "turn_unknown")
    return state


def _explore_intrinsic_bonus(turn_records: List[Dict[str, Any]]) -> float:
    """Sum of 1/sqrt(count) bonuses for unique commands (MERCI-style).

    Granularity controlled by EXPLORE_INTRINSIC_GRANULARITY env var:
      - "raw"       : full command text (default, v1 behavior)
      - "signature" : cmd name + first 2 args (skill-level, Agent57-style)

    Scope controlled by EXPLORE_INTRINSIC_SCOPE:
      - "process" : historical behavior, process-local counter across rollouts
      - "episode" : reset counts per rollout; lower-risk under multi-process Ray
    """
    if not _EXPLORE_INTRINSIC_ENABLED or not turn_records:
        return 0.0
    total = 0.0
    action_count = 0
    episode_counter: Dict[str, int] = {}
    for action in _iter_explore_actions(turn_records):
        action_count += 1
        if _EXPLORE_INTRINSIC_GRANULARITY == "signature":
            key_src = action["signature"]
        else:
            key_src = action["raw"]
        key = hashlib.md5(key_src.encode()).hexdigest()[:10]
        if _EXPLORE_INTRINSIC_SCOPE == "episode":
            # Bug fix / robustness: process-level counters diverge across Ray
            # rollout workers. Episode scope gives deterministic within-rollout
            # novelty and is the default for the robust_dapo_lite preset.
            episode_counter[key] = episode_counter.get(key, 0) + 1
            total += 1.0 / math.sqrt(episode_counter[key])
        else:
            _CMD_COUNTER[key] = _CMD_COUNTER.get(key, 0) + 1
            total += 1.0 / math.sqrt(_CMD_COUNTER[key])
    if _EXPLORE_INTRINSIC_REDUCER == "mean" and action_count > 0:
        return total / action_count
    return total


def _explore_episode_signature_novelty(
    turn_records: List[Dict[str, Any]],
    *,
    reducer: str = "sum",
) -> float:
    """Episode-local novelty used by Agent57 NGU-lite product mode."""
    global _AGENT57_LAST_EPISODIC_STATS, _AGENT57_LAST_EPISODIC_BY_TURN
    _AGENT57_LAST_EPISODIC_STATS = {}
    _AGENT57_LAST_EPISODIC_BY_TURN = {}
    if not turn_records:
        return 0.0
    total = 0.0
    action_count = 0
    episode_counter: Dict[str, int] = {}
    turn_total: Dict[int, float] = {}
    turn_count: Dict[int, int] = {}
    episodic_memory = create_episodic_memory_backend(_AGENT57_CONFIG.episodic_backend)
    empty_bucket_count = 0.0
    exact_repeat_count = 0.0
    candidate_count_total = 0.0
    probe_count_total = 0.0
    for action in _iter_explore_actions(turn_records):
        action_count += 1
        try:
            turn_idx = int(action.get("turn_idx", -1))
        except (TypeError, ValueError):
            turn_idx = -1
        if episodic_memory is not None:
            state = _explore_agent57_episodic_state(action)
            novelty = float(episodic_memory.compute_novelty(state))
            total += novelty
            turn_total[turn_idx] = turn_total.get(turn_idx, 0.0) + novelty
            turn_count[turn_idx] = turn_count.get(turn_idx, 0) + 1
            query_stats_fn = getattr(episodic_memory, "last_query_stats", None)
            query_stats = query_stats_fn() if callable(query_stats_fn) else {}
            empty_bucket_count += float(query_stats.get("empty_bucket", 0.0) or 0.0)
            exact_repeat_count += float(query_stats.get("exact_repeat", 0.0) or 0.0)
            candidate_count_total += float(query_stats.get("candidate_count", 0.0) or 0.0)
            probe_count_total += float(query_stats.get("probe_count", 0.0) or 0.0)
            episodic_memory.add(state)
            continue
        key_src = action["signature"]
        key = hashlib.md5(key_src.encode()).hexdigest()[:10]
        episode_counter[key] = episode_counter.get(key, 0) + 1
        novelty = 1.0 / math.sqrt(episode_counter[key])
        total += novelty
        turn_total[turn_idx] = turn_total.get(turn_idx, 0.0) + novelty
        turn_count[turn_idx] = turn_count.get(turn_idx, 0) + 1
    value = total / action_count if reducer == "mean" and action_count > 0 else total
    if action_count > 0:
        if reducer == "mean":
            _AGENT57_LAST_EPISODIC_BY_TURN = {
                idx: turn_total[idx] / max(1, turn_count.get(idx, 0))
                for idx in turn_total
                if idx >= 0
            }
        else:
            _AGENT57_LAST_EPISODIC_BY_TURN = {
                idx: turn_total[idx]
                for idx in turn_total
                if idx >= 0
            }
        _AGENT57_LAST_EPISODIC_STATS = {
            "explore_agent57_episodic_action_count": float(action_count),
            "explore_agent57_episodic_empty_bucket_count": float(empty_bucket_count),
            "explore_agent57_episodic_empty_bucket_rate": float(empty_bucket_count / action_count),
            "explore_agent57_episodic_exact_repeat_count": float(exact_repeat_count),
            "explore_agent57_episodic_candidate_count_mean": float(candidate_count_total / action_count),
            "explore_agent57_episodic_probe_count_mean": float(probe_count_total / action_count),
        }
    return value


def _explore_score_bonus_from_components(
    components_raw: str,
    *,
    intrinsic: float,
    safety: float,
    lprnd: float,
    agent57: float,
    cde_actor: float,
) -> float:
    """Select which exploration components are injected into reward["score"]."""
    raw = (components_raw or "").strip().lower()
    if raw in {"", "none", "off", "0"}:
        return 0.0
    values = {
        "intrinsic": intrinsic,
        "explore_intrinsic_scaled": intrinsic,
        "safety": safety,
        "explore_safety_penalty": safety,
        "lprnd": lprnd,
        "explore_lprnd": lprnd,
        "agent57": agent57,
        "ngu": agent57,
        "explore_agent57_ngu_bonus": agent57,
        "cde": cde_actor,
        "cde_actor": cde_actor,
        "explore_cde_actor_bonus": cde_actor,
    }
    if raw == "legacy":
        return intrinsic + safety + lprnd + agent57 + cde_actor
    total = 0.0
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        total += values.get(key, 0.0)
    return total


def _explore_safety_penalty(turn_records: List[Dict[str, Any]]) -> float:
    """Negative penalty if any turn matched a danger pattern."""
    if not _EXPLORE_SAFETY_FILTER_ENABLED or not turn_records:
        return 0.0
    pen = 0.0
    for action in _iter_explore_actions(turn_records):
        danger_text = action.get("danger_text", "")
        if danger_text and _DANGER_RE.search(danger_text):
            pen += _EXPLORE_SAFETY_FILTER_COEF
    return pen


def _explore_lprnd_bonus(interactions) -> float:
    """LP-RND lifelong novelty: reuse rollout_log_probs as the 'surprise' signal.

    The intuition (from the Agent57→Agentic-RL analysis, 草案 C):
      r_t^life = clip( (-mean_logprob - mu) / sigma, 0, L )

    Higher negative-logprob = trajectory is more surprising under current policy =
    indicates exploration into previously-low-density regions. Running stats keep
    the bonus normalized so it doesn't dominate task reward as training progresses.

    Zero extra parameters: relies entirely on log-probs already computed by slime.
    Returns 0.0 when disabled or during EXPLORE_LPRND_WARMUP.
    """
    if not _EXPLORE_LPRND_ENABLED or not interactions:
        return 0.0
    # Average negative logprob across all generated tokens in this rollout.
    total_logp, total_tok = 0.0, 0
    for it in interactions:
        lp = list(getattr(it, "output_token_logprobs", []) or [])
        if not lp:
            continue
        total_logp += sum(lp)
        total_tok += len(lp)
    if total_tok == 0:
        return 0.0
    surprise = -(total_logp / total_tok)  # mean negative logprob, in nats

    s = _LPRND_STATS
    if s["warmup"] < _EXPLORE_LPRND_WARMUP:
        # Bug fix: the previous implementation updated Welford statistics during
        # warmup and then returned 0. That made early high-entropy rollouts the
        # normalization baseline, suppressing the novelty signal later. Warmup
        # now only counts trajectories; normalization starts afterward.
        s["warmup"] += 1
        return 0.0

    # Welford running stats after warmup.
    s["n"] += 1
    delta = surprise - s["mean"]
    s["mean"] += delta / s["n"]
    s["m2"] += delta * (surprise - s["mean"])
    if s["n"] < 2:
        return 0.0
    var = s["m2"] / max(1, s["n"] - 1)
    std = max(math.sqrt(var), 1e-6)
    z = (surprise - s["mean"]) / std
    return max(0.0, min(_EXPLORE_LPRND_CLIP, z))


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _token_fingerprint(token_ids: list[int], limit: int) -> str | None:
    if not token_ids:
        return None
    try:
        payload = json.dumps(
            [int(x) for x in token_ids[:limit]],
            separators=(",", ":"),
        ).encode("utf-8")
    except Exception:
        return None
    return hashlib.sha256(payload).hexdigest()[:16]


def _turn_uncertainty_metrics(
    interaction: Interaction,
    *,
    previous_turn_score: float | None = None,
) -> dict[str, Any]:
    """Build T2PO-style turn diagnostics from sampled-token log-probs."""
    if not _TURN_UNCERTAINTY_ENABLED:
        return {}

    output_ids = list(interaction.output_token_ids or [])
    raw_logprobs = list(interaction.output_token_logprobs or [])
    nums = [_finite_float(v) for v in raw_logprobs]
    nums = [v for v in nums if v is not None]

    record: dict[str, Any] = {
        "schema": _TURN_UNCERTAINTY_SCHEMA,
        "schema_version": _TURN_UNCERTAINTY_SCHEMA_VERSION,
        "source": "rollout_logprobs",
        "score_kind": "mean_sampled_token_logprob_proxy",
        "turn_idx": int(interaction.turn_idx),
        "available": False,
        "n_input_tokens": len(interaction.input_ids or []),
        "n_output_tokens": len(output_ids),
        "n_logprob_tokens": len(nums),
        "ignored_prefix_tokens": min(_TURN_UNCERTAINTY_WARMUP_TOKENS, len(nums)),
        "fingerprint": _token_fingerprint(
            output_ids, _TURN_UNCERTAINTY_FINGERPRINT_TOKENS
        ),
        "fingerprint_tokens": _TURN_UNCERTAINTY_FINGERPRINT_TOKENS,
        "finish_reason": interaction.finish_reason,
        "latency_ms": float(interaction.latency_ms or 0.0),
        "low_progress_threshold": _TURN_LOW_PROGRESS_THRESHOLD,
    }

    if not nums:
        record["missing_reason"] = "missing_output_token_logprobs"
        return record

    scored = nums[_TURN_UNCERTAINTY_WARMUP_TOKENS:]
    if not scored:
        record["missing_reason"] = "all_tokens_skipped_by_warmup"
        return record

    count = len(scored)
    mean_logprob = sum(scored) / count
    variance = sum((x - mean_logprob) ** 2 for x in scored) / count
    mean_neg_logprob = -mean_logprob
    turn_score = mean_logprob

    record.update(
        {
            "available": True,
            "n_scored_tokens": count,
            "turn_level_score": turn_score,
            "turn_level_uncertainty": mean_neg_logprob,
            "mean_logprob": mean_logprob,
            "std_logprob": math.sqrt(max(variance, 0.0)),
            "min_logprob": min(scored),
            "max_logprob": max(scored),
            "mean_neg_logprob": mean_neg_logprob,
            "sum_neg_logprob": -sum(scored),
            "log_ppl": mean_neg_logprob,
            "ppl": math.exp(min(mean_neg_logprob, 50.0)),
            "first_scored_logprob": scored[0],
            "last_scored_logprob": scored[-1],
        }
    )

    if previous_turn_score is not None and math.isfinite(previous_turn_score):
        delta = turn_score - previous_turn_score
        abs_delta = abs(delta)
        record["score_delta_from_prev"] = delta
        record["abs_score_delta_from_prev"] = abs_delta
        record["low_progress_from_prev"] = (
            abs_delta > 0.0 and abs_delta < _TURN_LOW_PROGRESS_THRESHOLD
        )
    else:
        record["score_delta_from_prev"] = None
        record["abs_score_delta_from_prev"] = None
        record["low_progress_from_prev"] = False

    return record


def _summarize_turn_uncertainty(
    records: list[dict[str, Any]],
    *,
    run_ctx: RunContext,
) -> dict[str, Any]:
    if not _TURN_UNCERTAINTY_ENABLED:
        return {}

    all_records = [r for r in records if isinstance(r, dict) and r]
    available = [r for r in all_records if r.get("available")]

    def collect(key: str) -> list[float]:
        vals: list[float] = []
        for rec in available:
            num = _finite_float(rec.get(key))
            if num is not None:
                vals.append(num)
        return vals

    def stats(values: list[float]) -> dict[str, float | int] | None:
        if not values:
            return None
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        return {
            "count": len(values),
            "mean": mean,
            "std": math.sqrt(max(var, 0.0)),
            "min": min(values),
            "max": max(values),
        }

    scores = collect("turn_level_score")
    uncertainties = collect("turn_level_uncertainty")
    deltas = collect("abs_score_delta_from_prev")
    score_stats = stats(scores)
    uncertainty_stats = stats(uncertainties)
    delta_stats = stats(deltas)
    low_progress_count = sum(1 for r in available if r.get("low_progress_from_prev"))

    summary: dict[str, Any] = {
        "schema": _TURN_UNCERTAINTY_SCHEMA,
        "schema_version": _TURN_UNCERTAINTY_SCHEMA_VERSION,
        "source": "rollout_logprobs",
        "score_kind": "mean_sampled_token_logprob_proxy",
        "uid": run_ctx.uid,
        "group_index": run_ctx.group_index,
        "sample_index": run_ctx.sample_index,
        "rollout_id": run_ctx.rollout_id,
        "train_step": run_ctx.train_step,
        "rollout_step": run_ctx.rollout_step,
        "turn_count": len(all_records),
        "available_turn_count": len(available),
        "missing_turn_count": len(all_records) - len(available),
        "warmup_tokens": _TURN_UNCERTAINTY_WARMUP_TOKENS,
        "low_progress_threshold": _TURN_LOW_PROGRESS_THRESHOLD,
        "low_progress_turn_count": low_progress_count,
        "low_progress_fraction": (
            low_progress_count / len(available) if available else None
        ),
    }

    if score_stats:
        summary.update(
            {
                "mean_turn_level_score": score_stats["mean"],
                "std_turn_level_score": score_stats["std"],
                "min_turn_level_score": score_stats["min"],
                "max_turn_level_score": score_stats["max"],
            }
        )
    if uncertainty_stats:
        summary.update(
            {
                "mean_turn_level_uncertainty": uncertainty_stats["mean"],
                "std_turn_level_uncertainty": uncertainty_stats["std"],
                "min_turn_level_uncertainty": uncertainty_stats["min"],
                "max_turn_level_uncertainty": uncertainty_stats["max"],
            }
        )
    if delta_stats:
        summary.update(
            {
                "mean_abs_score_delta": delta_stats["mean"],
                "min_abs_score_delta": delta_stats["min"],
                "max_abs_score_delta": delta_stats["max"],
            }
        )

    return summary


def _explore_schedule_multiplier(schedule: str, train_step: Any, decay_steps: int) -> float:
    """SPEAR-style curriculum multiplier for auxiliary exploration rewards."""
    mode = (schedule or "constant").strip().lower()
    if mode in {"constant", "none", "off"}:
        return 1.0
    if decay_steps <= 0 or train_step is None:
        return 1.0
    try:
        step = max(0.0, float(train_step))
    except (TypeError, ValueError):
        return 1.0
    progress = min(1.0, step / max(1.0, float(decay_steps)))
    if mode == "cosine":
        return max(0.0, (math.cos(progress * math.pi) + 1.0) / 2.0)
    if mode == "linear":
        return max(0.0, 1.0 - progress)
    logger.warning("Unknown exploration schedule=%r; using constant", schedule)
    return 1.0


def _explore_actor_log_ppl(interactions) -> float:
    """Mean negative actor logprob over generated tokens, i.e. log perplexity."""
    total_logp, total_tok = 0.0, 0
    for it in interactions or []:
        lp = list(getattr(it, "output_token_logprobs", []) or [])
        if not lp:
            continue
        total_logp += sum(lp)
        total_tok += len(lp)
    if total_tok <= 0:
        return 0.0
    return max(0.0, -(total_logp / total_tok))


def _explore_decayed_weight(weight: float, train_step: Any, decay_steps: int) -> float:
    if decay_steps <= 0 or train_step is None:
        return max(0.0, float(weight))
    try:
        step = max(0.0, float(train_step))
    except (TypeError, ValueError):
        return max(0.0, float(weight))
    progress = min(1.0, step / max(1.0, float(decay_steps)))
    return max(0.0, float(weight) * (1.0 - progress))


def _explore_cde_actor_metrics(
    interactions,
    base_score_mean: float,
    train_step: Any,
) -> Dict[str, float]:
    """Actor-side CDE/PPL curiosity metrics for optional reward shaping.

    This intentionally implements only the actor bonus from the CDE paper. The
    critic bonus requires a multi-head critic/value path, which terminal-rl's
    current GRPO/DAPO rollout path does not have.
    """
    metrics = {
        "log_ppl": 0.0,
        "omega": 0.0,
        "alpha": _EXPLORE_CDE_ACTOR_ALPHA,
        "kappa": _EXPLORE_CDE_ACTOR_KAPPA,
        "decay_steps": float(_EXPLORE_CDE_ACTOR_DECAY_STEPS),
        "base_score_mean": 0.0,
        "base_score_magnitude": 0.0,
        "cap": 0.0,
        "scaled": 0.0,
        "clipped": 0.0,
        "bonus": 0.0,
        "eligible": 0.0,
    }
    if not _EXPLORE_CDE_ACTOR_ENABLED:
        return metrics

    log_ppl = _explore_actor_log_ppl(interactions)
    omega = _explore_decayed_weight(
        _EXPLORE_CDE_ACTOR_OMEGA,
        train_step,
        _EXPLORE_CDE_ACTOR_DECAY_STEPS,
    )
    base_mean = float(base_score_mean)
    base_magnitude = abs(base_mean)
    gate = _EXPLORE_CDE_ACTOR_REWARD_GATE
    if gate in {"positive", "pos"}:
        eligible = base_mean > 0.0
    elif gate in {"nonnegative", "non-negative"}:
        eligible = base_mean >= 0.0
    elif gate in {"none", "off", "always", "all"}:
        eligible = True
    else:
        # Paper-faithful default: any non-zero verifiable reward magnitude can
        # bound curiosity. For safety-heavy runs, use gate=positive to avoid
        # softening unsafe negative rewards.
        eligible = base_magnitude > 0.0

    cap = base_magnitude / max(_EXPLORE_CDE_ACTOR_KAPPA, 1e-6) if eligible else 0.0
    scaled = max(0.0, _EXPLORE_CDE_ACTOR_ALPHA * log_ppl)
    clipped = min(cap, scaled)
    metrics.update(
        {
            "log_ppl": log_ppl,
            "omega": omega,
            "base_score_mean": base_mean,
            "base_score_magnitude": base_magnitude,
            "cap": cap,
            "scaled": scaled,
            "clipped": clipped,
            "bonus": omega * clipped,
            "eligible": 1.0 if eligible else 0.0,
        }
    )
    return metrics


def _explore_debug_metrics(
    *,
    status: Sample.Status,
    base_score_mean: float,
    total_bonus: float,
    intrinsic_scaled: float,
    safety_penalty: float,
    lprnd_bonus: float,
    agent57_bonus: float,
    cde_actor: Dict[str, float],
    turn_records: List[Dict[str, Any]],
    parse_error_count: int,
) -> Dict[str, Any]:
    """Structured exploration/exploitation diagnostics for logs and trajectory audits."""
    tool_call_count = 0
    action_count = 0
    danger_command_count = 0
    actions = _iter_explore_actions(turn_records)
    for tr in turn_records or []:
        tool_calls = tr.get("tool_calls") or []
        if isinstance(tool_calls, list):
            tool_call_count += len(tool_calls)
    for action in actions:
        action_count += 1
        danger_text = action.get("danger_text", "")
        if danger_text and _DANGER_RE.search(danger_text):
            danger_command_count += 1

    base_abs = abs(float(base_score_mean))
    bonus_to_base = abs(float(total_bonus)) / max(base_abs, 1e-6)
    curiosity_pressure = (
        max(0.0, intrinsic_scaled)
        + max(0.0, lprnd_bonus)
        + max(0.0, agent57_bonus)
        + max(0.0, float(cde_actor.get("bonus", 0.0)))
    )
    safety_pressure = max(0.0, -float(safety_penalty)) + float(danger_command_count)
    reward_hacking_risk = bool(base_score_mean <= 0.0 and total_bonus > 0.0)
    over_exploration_risk = bool(bonus_to_base > 0.5 and base_score_mean <= 0.0)
    safety_tension = bool(safety_pressure > 0.0)

    status_value = getattr(status, "value", str(status)).lower()
    if status_value in {"failed", "aborted", "truncated"}:
        mood = "stuck"
    elif safety_tension:
        mood = "risky"
    elif reward_hacking_risk:
        mood = "curious_unproven"
    elif base_score_mean > 0.0 and curiosity_pressure > 0.0:
        mood = "curious_success"
    elif base_score_mean > 0.0:
        mood = "confident_exploit"
    elif total_bonus < 0.0:
        mood = "cautious"
    else:
        mood = "low_signal"

    mood_code = {
        "low_signal": 0,
        "confident_exploit": 1,
        "curious_success": 2,
        "curious_unproven": 3,
        "cautious": 4,
        "risky": 5,
        "stuck": 6,
    }.get(mood, -1)

    return {
        "explore_base_score_before_bonus": base_score_mean,
        "explore_bonus_to_base_abs_ratio": bonus_to_base,
        "explore_curiosity_pressure": curiosity_pressure,
        "explore_tool_intrinsic_pressure": max(0.0, intrinsic_scaled),
        "explore_safety_pressure": safety_pressure,
        "explore_mood": mood,
        "explore_mood_code": mood_code,
        "explore_reward_hacking_risk": reward_hacking_risk,
        "explore_over_exploration_risk": over_exploration_risk,
        "explore_safety_tension": safety_tension,
        "explore_turn_count": len(turn_records or []),
        "explore_tool_call_count": tool_call_count,
        "explore_action_count": action_count,
        "explore_danger_command_count": danger_command_count,
        "explore_parse_error_count": int(parse_error_count or 0),
    }


# ─── Trajectory export (parallels swe-rl/generate_with_swe_remote.py:78-137) ───
# Toggle via env var TERMINAL_SAVE_TRAJ_DIR (empty=disabled).
# Output layout (one dir per rollout sample):
#   {save_dir}/t{task}_r{rollout_id}_st{train_step}_g{group}_s{sample}_{uid}_{ts}/
#       meta.json       # task spec + sampling params + reward breakdown
#       traj.json       # per-turn dialogue + tool calls + ClawSentry decisions

def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def _get_terminal_save_dir() -> Path | None:
    save_dir = os.getenv("TERMINAL_SAVE_TRAJ_DIR", "").strip()
    if not save_dir:
        return None
    path = Path(save_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("TERMINAL_SAVE_TRAJ_DIR=%s mkdir failed: %s", save_dir, exc)
        return None
    return path


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sample_or_env_int(sample: Sample, key: str, env_name: str) -> int | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = metadata.get(key)
    if value is None:
        value = os.getenv(env_name)
    return _optional_int(value)


def _trajectory_dataset_slug(data_source: str | None) -> str:
    raw = str(data_source or "").strip().lower()
    if raw in {"", "terminal_bench", "seta", "seta_env"}:
        return "seta"
    if raw in {"agent_safetybench", "agent-safety-bench", "safety", "asb"}:
        return "agent_safetybench"
    if raw in {"agentharm", "agent_harm", "ah"}:
        return "agentharm"
    return _sanitize_filename(raw) or "unknown"


def _interval_candidates_for_dataset(dataset_slug: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if dataset_slug == "seta":
        return (
            ("trajectory_save_interval_seta", "trajectory_save_interval_terminal_bench"),
            ("TRAJECTORY_SAVE_INTERVAL_SETA", "SAVE_INTERVAL_SETA"),
        )
    if dataset_slug == "agent_safetybench":
        return (
            (
                "trajectory_save_interval_agent_safetybench",
                "trajectory_save_interval_asb",
                "trajectory_save_interval_safety",
            ),
            (
                "TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH",
                "TRAJECTORY_SAVE_INTERVAL_ASB",
                "TRAJECTORY_SAVE_INTERVAL_SAFETY",
                "SAVE_INTERVAL_AGENT_SAFETYBENCH",
                "SAVE_INTERVAL_ASB",
                "SAVE_INTERVAL_SAFETY",
            ),
        )
    if dataset_slug == "agentharm":
        return (
            ("trajectory_save_interval_agentharm", "trajectory_save_interval_agent_harm"),
            (
                "TRAJECTORY_SAVE_INTERVAL_AGENTHARM",
                "TRAJECTORY_SAVE_INTERVAL_AGENT_HARM",
                "SAVE_INTERVAL_AGENTHARM",
                "SAVE_INTERVAL_AGENT_HARM",
            ),
        )
    return (
        (f"trajectory_save_interval_{dataset_slug}",),
        (f"TRAJECTORY_SAVE_INTERVAL_{dataset_slug.upper()}",),
    )


def _trajectory_save_interval(args, data_source: str | None = None) -> int:
    dataset_slug = _trajectory_dataset_slug(data_source)
    arg_names, env_names = _interval_candidates_for_dataset(dataset_slug)
    raw = None
    raw_source = None
    for name in arg_names:
        value = getattr(args, name, None)
        if value is not None and value != "":
            raw = value
            raw_source = name
            break
    if raw is None:
        for name in env_names:
            value = os.getenv(name)
            if value is not None and value != "":
                raw = value
                raw_source = name
                break
    if raw is None:
        raw = getattr(args, "trajectory_save_interval", None)
        raw_source = "trajectory_save_interval"
    if raw is None or raw == "":
        raw = os.getenv("TRAJECTORY_SAVE_INTERVAL", "1")
        raw_source = "TRAJECTORY_SAVE_INTERVAL"
    value = _optional_int(raw)
    if value is None:
        logger.warning(
            "Invalid trajectory save interval %s=%r for dataset=%s; falling back to 1",
            raw_source,
            raw,
            dataset_slug,
        )
        return 1
    return value


def _should_save_trajectory(run_ctx: RunContext, interval: int) -> bool:
    if interval <= 0:
        return False
    if interval == 1:
        return True
    step = run_ctx.train_step
    if step is None:
        step = run_ctx.rollout_id
    if step is None:
        # No rollout metadata is available, so preserve old save-all behavior.
        return True
    return int(step) % interval == 0


def _trajectory_save_policy() -> str:
    raw = os.getenv("TRAJECTORY_SAVE_POLICY", "step_interval").strip().lower()
    if raw in {"", "legacy", "interval"}:
        return "step_interval"
    if raw in {"task_timeseries", "task-time-series", "task_step", "task-step"}:
        return "task_timeseries"
    logger.warning("Unknown TRAJECTORY_SAVE_POLICY=%r; using step_interval", raw)
    return "step_interval"


def _trajectory_env_int(name: str, default: int) -> int:
    return _env_int(name, default)


def _trajectory_task_save_interval(default_interval: int) -> int:
    raw = os.getenv("TRAJECTORY_TASK_SAVE_INTERVAL", "").strip()
    if not raw:
        return default_interval
    value = _optional_int(raw)
    if value is None:
        logger.warning(
            "Invalid TRAJECTORY_TASK_SAVE_INTERVAL=%r; using %d",
            raw,
            default_interval,
        )
        return default_interval
    return value


def _trajectory_reward_strata() -> set[str]:
    raw = os.getenv("TRAJECTORY_SAVE_REWARD_STRATA", "best,worst")
    values = {
        part.strip().lower()
        for part in raw.split(",")
        if part.strip()
    }
    allowed = {"best", "worst", "latest"}
    unknown = values - allowed
    if unknown:
        logger.warning(
            "Ignoring unknown TRAJECTORY_SAVE_REWARD_STRATA entries: %s",
            sorted(unknown),
        )
    values &= allowed
    return values or {"best", "worst"}


def _trajectory_step_value(run_ctx: RunContext) -> int | None:
    for value in (run_ctx.train_step, run_ctx.rollout_step, run_ctx.rollout_id):
        step = _optional_int(value)
        if step is not None:
            return step
    return None


def _trajectory_task_id(task_spec: TaskSpec) -> str:
    name = str(task_spec.task_name or "unknown")
    path = str(task_spec.task_path or "")
    digest = hashlib.sha1(f"{name}\n{path}".encode("utf-8")).hexdigest()[:8]
    slug = _sanitize_filename(name)[:96].strip("._-") or "unknown"
    return f"{slug}-{digest}"


def _trajectory_reward_value(reward: Dict[str, Any]) -> float | None:
    for key in ("total_reward", "score", "raw_reward", "raw_score", "accuracy"):
        value = reward.get(key)
        try:
            if value is None or value == "":
                continue
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric
    return None


def _format_reward_for_filename(value: float | None) -> str:
    if value is None:
        return "na"
    text = f"{value:+.3f}"
    return (
        text.replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )


def _trajectory_index_path(save_dir: Path) -> Path:
    return save_dir / "index.jsonl"


@contextmanager
def _trajectory_index_lock(save_dir: Path):
    lock_path = save_dir / ".index.lock"
    fh = None
    locked = False
    try:
        fh = lock_path.open("a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            locked = True
        except Exception as exc:
            logger.warning("[traj-save] could not lock %s: %s", lock_path, exc)
        yield
    finally:
        if fh is not None:
            if locked:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            fh.close()


def _trajectory_record_dir(save_dir: Path, record: dict[str, Any]) -> Path | None:
    rel_path = record.get("rel_path")
    if rel_path:
        path = save_dir / str(rel_path)
    else:
        raw_path = record.get("path")
        if not raw_path:
            return None
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = save_dir / path
    try:
        resolved_root = save_dir.resolve()
        resolved_path = path.resolve()
    except Exception:
        return None
    if resolved_path.parent != resolved_root:
        return None
    return resolved_path


def _trajectory_load_index(save_dir: Path) -> list[dict[str, Any]]:
    index_path = _trajectory_index_path(save_dir)
    if not index_path.exists():
        return []
    active: dict[str, dict[str, Any]] = {}
    try:
        for line in index_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            rel_path = str(record.get("rel_path") or "")
            if not rel_path:
                continue
            event = str(record.get("event") or "save")
            if event == "delete":
                active.pop(rel_path, None)
                continue
            if event == "save":
                record_dir = _trajectory_record_dir(save_dir, record)
                if record_dir is not None and (record_dir / "traj.json").exists():
                    active[rel_path] = record
    except Exception as exc:
        logger.warning("[traj-save] failed reading %s: %s", index_path, exc)
        return []
    return list(active.values())


def _trajectory_append_index(save_dir: Path, record: dict[str, Any]) -> None:
    index_path = _trajectory_index_path(save_dir)
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(record), ensure_ascii=False, default=str))
        fh.write("\n")


def _trajectory_record_reward(record: dict[str, Any]) -> float | None:
    for key in ("reward", "total_reward", "raw_reward", "raw_score"):
        try:
            value = record.get(key)
            if value is None or value == "":
                continue
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return numeric
    return None


def _trajectory_record_ts(record: dict[str, Any]) -> int:
    value = _optional_int(record.get("ts_ns"))
    if value is not None:
        return value
    value = _optional_int(record.get("created_ts_ns"))
    if value is not None:
        return value
    return 0


def _trajectory_keep_subset(
    records: list[dict[str, Any]],
    limit: int,
    strata: set[str],
) -> set[str]:
    if limit <= 0 or len(records) <= limit:
        return {str(r.get("rel_path")) for r in records if r.get("rel_path")}

    chosen: list[dict[str, Any]] = []

    def add(record: dict[str, Any] | None) -> None:
        if not record or len(chosen) >= limit:
            return
        rel_path = str(record.get("rel_path") or "")
        if rel_path and all(str(r.get("rel_path") or "") != rel_path for r in chosen):
            chosen.append(record)

    latest = max(records, key=_trajectory_record_ts, default=None)
    add(latest)
    reward_records = [
        record for record in records
        if _trajectory_record_reward(record) is not None
    ]
    if "best" in strata and reward_records:
        add(max(reward_records, key=lambda r: _trajectory_record_reward(r) or 0.0))
    if "worst" in strata and reward_records:
        add(min(reward_records, key=lambda r: _trajectory_record_reward(r) or 0.0))
    if "latest" in strata:
        add(latest)
    for record in sorted(records, key=_trajectory_record_ts, reverse=True):
        add(record)
        if len(chosen) >= limit:
            break
    return {str(r.get("rel_path")) for r in chosen if r.get("rel_path")}


def _trajectory_cleanup(
    save_dir: Path,
    active_records: list[dict[str, Any]],
    *,
    task_max_per_step: int,
    task_max_per_task: int,
    max_total: int,
    strata: set[str],
) -> int:
    to_delete: set[str] = set()

    if task_max_per_step > 0:
        by_task_step: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in active_records:
            key = (
                str(record.get("task_id") or record.get("task_name") or "unknown"),
                str(record.get("train_step") if record.get("train_step") is not None else "na"),
            )
            by_task_step.setdefault(key, []).append(record)
        for records in by_task_step.values():
            keep = _trajectory_keep_subset(records, task_max_per_step, strata)
            for record in records:
                rel_path = str(record.get("rel_path") or "")
                if rel_path and rel_path not in keep:
                    to_delete.add(rel_path)

    remaining = [
        record for record in active_records
        if str(record.get("rel_path") or "") not in to_delete
    ]
    if task_max_per_task > 0:
        by_task: dict[str, list[dict[str, Any]]] = {}
        for record in remaining:
            key = str(record.get("task_id") or record.get("task_name") or "unknown")
            by_task.setdefault(key, []).append(record)
        for records in by_task.values():
            keep = _trajectory_keep_subset(records, task_max_per_task, strata)
            for record in records:
                rel_path = str(record.get("rel_path") or "")
                if rel_path and rel_path not in keep:
                    to_delete.add(rel_path)

    remaining = [
        record for record in active_records
        if str(record.get("rel_path") or "") not in to_delete
    ]
    if max_total > 0 and len(remaining) > max_total:
        keep = _trajectory_keep_subset(remaining, max_total, strata | {"latest"})
        for record in remaining:
            rel_path = str(record.get("rel_path") or "")
            if rel_path and rel_path not in keep:
                to_delete.add(rel_path)

    deleted = 0
    for rel_path in sorted(to_delete):
        record = next(
            (r for r in active_records if str(r.get("rel_path") or "") == rel_path),
            None,
        )
        if record is None:
            continue
        target = _trajectory_record_dir(save_dir, record)
        if target is None or not target.exists():
            continue
        try:
            shutil.rmtree(target)
            deleted += 1
            _trajectory_append_index(
                save_dir,
                {
                    "event": "delete",
                    "schema_version": 1,
                    "rel_path": rel_path,
                    "path": str(target),
                    "deleted_ts_ns": time.time_ns(),
                    "reason": "retention_limit",
                },
            )
        except Exception as exc:
            logger.warning("[traj-save] cleanup failed for %s: %s", target, exc)
    return deleted


def _trajectory_save_decision(
    *,
    policy: str,
    run_ctx: RunContext,
    task_id: str,
    reward: float | None,
    interval: int,
    active_records: list[dict[str, Any]],
) -> dict[str, Any]:
    step = _trajectory_step_value(run_ctx)
    decision: dict[str, Any] = {
        "policy": policy,
        "saved": False,
        "reason": "skipped",
        "train_step": step,
        "task_id": task_id,
        "reward": reward,
        "legacy_interval": interval,
    }

    if policy == "step_interval":
        should_save = _should_save_trajectory(run_ctx, interval)
        decision.update(
            {
                "saved": bool(should_save),
                "reason": "legacy_interval" if should_save else "legacy_interval_skip",
            }
        )
        return decision

    if policy != "task_timeseries":
        decision["reason"] = "unknown_policy"
        return decision

    task_interval = _trajectory_task_save_interval(interval)
    max_per_step = _trajectory_env_int("TRAJECTORY_TASK_MAX_PER_STEP", 2)
    max_per_task = _trajectory_env_int("TRAJECTORY_TASK_MAX_PER_TASK", 24)
    max_total = _trajectory_env_int("TRAJECTORY_MAX_TOTAL", 5000)
    strata = _trajectory_reward_strata()
    decision.update(
        {
            "task_save_interval": task_interval,
            "task_max_per_step": max_per_step,
            "task_max_per_task": max_per_task,
            "max_total": max_total,
            "reward_strata": sorted(strata),
        }
    )

    if task_interval <= 0:
        decision["reason"] = "task_interval_disabled"
        return decision
    if step is not None and int(step) % task_interval != 0:
        decision["reason"] = "task_interval_skip"
        return decision

    same_task_step = [
        record for record in active_records
        if str(record.get("task_id") or record.get("task_name") or "unknown") == task_id
        and str(record.get("train_step") if record.get("train_step") is not None else "na")
        == str(step if step is not None else "na")
    ]
    decision["existing_task_step_count"] = len(same_task_step)
    if max_per_step <= 0 or len(same_task_step) < max_per_step:
        decision.update({"saved": True, "reason": "task_step_slot"})
        return decision

    reward_records = [
        record for record in same_task_step
        if _trajectory_record_reward(record) is not None
    ]
    if reward is not None and reward_records:
        rewards = [_trajectory_record_reward(record) for record in reward_records]
        rewards = [value for value in rewards if value is not None]
        if "best" in strata and rewards and reward > max(rewards):
            decision.update({"saved": True, "reason": "task_step_best"})
            return decision
        if "worst" in strata and rewards and reward < min(rewards):
            decision.update({"saved": True, "reason": "task_step_worst"})
            return decision

    decision["reason"] = "task_step_quota"
    return decision


def _attach_trajectory_save_metadata(
    samples: list[Sample],
    sample: Sample,
    metadata: dict[str, Any],
) -> None:
    targets = samples if samples else [sample]
    for target in targets:
        if not isinstance(target.metadata, dict):
            target.metadata = {}
        target.metadata["trajectory_save"] = _jsonable(metadata)


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    return str(obj)


def _exploration_audit_from_reward(reward: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(reward, dict):
        return {}
    keys = (
        "explore_mood",
        "explore_mood_code",
        "explore_total_bonus",
        "explore_base_score_before_bonus",
        "explore_bonus_to_base_abs_ratio",
        "explore_curiosity_pressure",
        "explore_tool_intrinsic_pressure",
        "explore_safety_pressure",
        "explore_reward_hacking_risk",
        "explore_over_exploration_risk",
        "explore_safety_tension",
        "explore_action_count",
        "explore_tool_call_count",
        "explore_danger_command_count",
        "explore_parse_error_count",
        "explore_intrinsic_scaled",
        "explore_intrinsic_in_total",
        "explore_lprnd",
        "explore_agent57_enabled",
        "explore_agent57_arm_id",
        "explore_agent57_beta",
        "explore_agent57_combine_mode",
        "explore_agent57_episodic_backend",
        "explore_agent57_controller",
        "explore_agent57_ucb_epsilon",
        "explore_agent57_ucb_min_per_arm",
        "explore_agent57_ucb_value",
        "explore_agent57_ucb_dataset_aware",
        "explore_agent57_ucb_random_seed",
        "explore_agent57_lifelong_enabled",
        "explore_agent57_lifelong_key_version",
        "explore_agent57_lifelong_include_dataset",
        "explore_agent57_lifelong_include_task",
        "explore_agent57_lifelong_include_turn",
        "explore_agent57_lifelong_obs_mode",
        "explore_agent57_lifelong_count_decay",
        "explore_agent57_lifelong_capacity",
        "explore_agent57_trust_gate_mode",
        "explore_agent57_trust",
        "explore_agent57_episodic_action_count",
        "explore_agent57_episodic_empty_bucket_count",
        "explore_agent57_episodic_empty_bucket_rate",
        "explore_agent57_episodic_exact_repeat_count",
        "explore_agent57_episodic_candidate_count_mean",
        "explore_agent57_episodic_probe_count_mean",
        "explore_agent57_lifelong_raw",
        "explore_agent57_lifelong_z",
        "explore_agent57_lifelong_stat_n",
        "explore_agent57_lifelong_stat_mean",
        "explore_agent57_lifelong_stat_std",
        "explore_agent57_lifelong_stat_error",
        "explore_agent57_lifelong_bonus",
        "explore_agent57_lifelong_bonus_unclipped",
        "explore_agent57_ngu_episodic_source",
        "explore_agent57_ngu_episodic_reducer",
        "explore_agent57_ngu_life_mod_mode",
        "explore_agent57_ngu_life_mod_std_clip",
        "explore_agent57_ngu_mod_clip",
        "explore_agent57_ngu_episodic",
        "explore_agent57_ngu_life_mod",
        "explore_agent57_intrinsic_signal",
        "explore_agent57_ngu_bonus",
        "explore_agent57_ngu_bonus_unclipped",
        "explore_agent57_bonus_unclipped",
        "explore_agent57_bonus_clipped",
        "explore_agent57_lifelong_eligible",
        "explore_agent57_lifelong_suppressed_reason",
        "explore_cde_actor_bonus",
        "explore_cde_actor_log_ppl",
        "explore_cde_actor_reward_gate",
        "explore_cde_actor_eligible",
        "exploration_reward_save_stage",
        "explore_post_norm_bonus_available_at_save",
    )
    return {key: reward[key] for key in keys if key in reward}


def _save_rollout_artifacts(
    *,
    task_spec: TaskSpec,
    run_ctx: RunContext,
    sampling_params: dict,
    sample: Sample,
    samples: List[Sample],
    status: Sample.Status,
    raw_score: float,
    eval_error: str | None,
    turn_records: List[Dict[str, Any]],
    safety_meta: Dict[str, Any] | None,
    prm_meta: Dict[str, Any] | None,
    safety_coef: float,
    prm_coef: float,
    trajectory_save_interval: int = 1,
) -> None:
    """Persist a full rollout (dialogue + tool calls + ClawSentry + reward) to disk.

    Mirrors swe-rl rollout export format. Failures are logged & swallowed so
    training is never blocked.
    """
    try:
        save_dir = _get_terminal_save_dir()
        if save_dir is None:
            return

        # Only save trajectories worth analyzing:
        # - Skip if no turns recorded (reset failed, no model output)
        # - Skip if status is FAILED and raw_score is 0 (infra failure, not model failure)
        if not turn_records:
            _attach_trajectory_save_metadata(
                samples,
                sample,
                {
                    "saved": False,
                    "policy": _trajectory_save_policy(),
                    "reason": "no_turns",
                    "train_step": _trajectory_step_value(run_ctx),
                    "rollout_id": run_ctx.rollout_id,
                    "uid": run_ctx.uid,
                },
            )
            return
        if str(status) == "Status.FAILED" and raw_score == 0.0 and len(turn_records) <= 1:
            _attach_trajectory_save_metadata(
                samples,
                sample,
                {
                    "saved": False,
                    "policy": _trajectory_save_policy(),
                    "reason": "infra_failure_short_rollout",
                    "train_step": _trajectory_step_value(run_ctx),
                    "rollout_id": run_ctx.rollout_id,
                    "uid": run_ctx.uid,
                },
            )
            return
        primary_metadata = (
            samples[0].metadata
            if samples and isinstance(samples[0].metadata, dict)
            else (sample.metadata if isinstance(sample.metadata, dict) else {})
        )
        dataset_slug = _trajectory_dataset_slug(primary_metadata.get("data_source"))

        # Build reward breakdown from the first trainable sample (all samples
        # in a rollout share accuracy/raw/base; turn_idx differs per sample).
        reward_breakdown: Dict[str, Any] = {"raw_score": raw_score}
        if samples:
            r0 = samples[0].reward if isinstance(samples[0].reward, dict) else {}
            for k in (
                "accuracy", "raw_score", "base_score", "score",
                "raw_reward", "task_reward", "exploration_reward", "total_reward",
                "prm_turn_score", "safety_score", "safety_coef",
                "explore_intrinsic", "explore_intrinsic_scaled",
                "explore_intrinsic_in_total",
                "explore_intrinsic_coef", "explore_intrinsic_effective_coef",
                "explore_intrinsic_schedule", "explore_intrinsic_decay_steps",
                "explore_intrinsic_schedule_multiplier",
                "explore_intrinsic_reducer",
                "explore_intrinsic_granularity", "explore_intrinsic_scope",
                "explore_safety_penalty",
                "explore_lprnd", "explore_lprnd_raw", "explore_lprnd_coef",
                "explore_lprnd_effective_coef", "explore_lprnd_schedule",
                "explore_lprnd_decay_steps", "explore_lprnd_schedule_multiplier",
                "explore_agent57_enabled",
                "explore_agent57_arm_id", "explore_agent57_k",
                "explore_agent57_beta", "explore_agent57_controller",
                "explore_agent57_combine_mode", "explore_agent57_max_bonus",
                "explore_agent57_episodic_backend",
                "explore_agent57_ucb_c", "explore_agent57_ucb_window",
                "explore_agent57_ucb_epsilon",
                "explore_agent57_ucb_min_per_arm",
                "explore_agent57_ucb_value",
                "explore_agent57_ucb_dataset_aware",
                "explore_agent57_ucb_random_seed",
                "explore_agent57_lifelong_enabled",
                "explore_agent57_lifelong_backend",
                "explore_agent57_lifelong_state_path",
                "explore_agent57_lifelong_coef",
                "explore_agent57_lifelong_clip",
                "explore_agent57_lifelong_warmup",
                "explore_agent57_lifelong_count_decay",
                "explore_agent57_lifelong_capacity",
                "explore_agent57_lifelong_key_version",
                "explore_agent57_lifelong_include_dataset",
                "explore_agent57_lifelong_include_task",
                "explore_agent57_lifelong_include_turn",
                "explore_agent57_lifelong_obs_mode",
                "explore_agent57_trust_gate_mode",
                "explore_agent57_trust",
                "explore_agent57_episodic_action_count",
                "explore_agent57_episodic_empty_bucket_count",
                "explore_agent57_episodic_empty_bucket_rate",
                "explore_agent57_episodic_exact_repeat_count",
                "explore_agent57_episodic_candidate_count_mean",
                "explore_agent57_episodic_probe_count_mean",
                "explore_agent57_lifelong_raw",
                "explore_agent57_lifelong_z",
                "explore_agent57_lifelong_stat_n",
                "explore_agent57_lifelong_stat_mean",
                "explore_agent57_lifelong_stat_std",
                "explore_agent57_lifelong_stat_error",
                "explore_agent57_lifelong_bonus",
                "explore_agent57_lifelong_bonus_unclipped",
                "explore_agent57_ngu_episodic_source",
                "explore_agent57_ngu_episodic_reducer",
                "explore_agent57_ngu_life_mod_mode",
                "explore_agent57_ngu_life_mod_std_clip",
                "explore_agent57_ngu_mod_clip",
                "explore_agent57_ngu_episodic",
                "explore_agent57_ngu_life_mod",
                "explore_agent57_intrinsic_signal",
                "explore_agent57_ngu_bonus",
                "explore_agent57_ngu_bonus_unclipped",
                "explore_agent57_bonus_unclipped",
                "explore_agent57_bonus_clipped",
                "explore_agent57_lifelong_unique_keys",
                "explore_agent57_lifelong_seen_before",
                "explore_agent57_lifelong_warmup_remaining",
                "explore_agent57_lifelong_eligible",
                "explore_agent57_lifelong_suppressed_reason",
                "explore_cde_actor_bonus",
                "explore_cde_actor_log_ppl", "explore_cde_actor_omega",
                "explore_cde_actor_alpha", "explore_cde_actor_kappa",
                "explore_cde_actor_reward_gate", "explore_cde_actor_eligible",
                "explore_cde_actor_decay_steps",
                "explore_cde_actor_base_mean", "explore_cde_actor_base_magnitude",
                "explore_cde_actor_cap",
                "explore_cde_actor_scaled",
                "explore_cde_actor_clipped", "explore_total_bonus",
                "explore_all_bonus", "explore_score_bonus_components",
                "explore_base_score_before_bonus",
                "explore_bonus_to_base_abs_ratio",
                "explore_curiosity_pressure",
                "explore_tool_intrinsic_pressure",
                "explore_safety_pressure",
                "explore_mood", "explore_mood_code",
                "explore_reward_hacking_risk",
                "explore_over_exploration_risk",
                "explore_safety_tension",
                "explore_turn_count", "explore_tool_call_count",
                "explore_action_count", "explore_danger_command_count",
                "explore_parse_error_count",
                "dapo_overlong_reward", "dapo_overlong",
                "dapo_overlong_expected_len", "dapo_overlong_buffer_len",
            ):
                if k in r0:
                    reward_breakdown[k] = r0[k]
            reward_details = (
                samples[0].metadata.get("reward_details")
                if isinstance(samples[0].metadata, dict)
                else None
            )
            if reward_details:
                reward_breakdown["details"] = reward_details
            if (
                reward_breakdown.get("explore_agent57_enabled")
                and "explore_post_norm_bonus" not in reward_breakdown
            ):
                reward_breakdown["exploration_reward_save_stage"] = (
                    "generate_pre_reward_postprocess"
                )
                reward_breakdown["explore_post_norm_bonus_available_at_save"] = False
            reward_breakdown["per_turn_scores"] = [
                {
                    "turn_idx": s.metadata.get("turn_idx"),
                    "score": (s.reward or {}).get("score"),
                    "prm_turn_score": (s.reward or {}).get("prm_turn_score"),
                    "safety_score": (s.reward or {}).get("safety_score"),
                }
                for s in samples
            ]
        primary_reward_details = primary_metadata.get("reward_details")
        primary_reward_reason = (
            primary_reward_details.get("reason")
            if isinstance(primary_reward_details, dict)
            else None
        )
        task_id = _trajectory_task_id(task_spec)
        policy = _trajectory_save_policy()
        reward_value = _trajectory_reward_value(reward_breakdown)
        with _trajectory_index_lock(save_dir):
            active_records = _trajectory_load_index(save_dir)
            save_decision = _trajectory_save_decision(
                policy=policy,
                run_ctx=run_ctx,
                task_id=task_id,
                reward=reward_value,
                interval=trajectory_save_interval,
                active_records=active_records,
            )
        if not save_decision.get("saved"):
            decision_metadata = {
                **save_decision,
                "dataset_slug": dataset_slug,
                "task_name": task_spec.task_name,
                "task_path": task_spec.task_path,
                "rollout_id": run_ctx.rollout_id,
                "group_index": run_ctx.group_index,
                "sample_index": run_ctx.sample_index,
                "uid": run_ctx.uid,
            }
            _attach_trajectory_save_metadata(samples, sample, decision_metadata)
            if _env_bool("TRAJECTORY_SAVE_LOG_DECISIONS", False):
                logger.info(
                    "[traj-save] skipped task=%s step=%s policy=%s reason=%s",
                    task_spec.task_name,
                    save_decision.get("train_step"),
                    policy,
                    save_decision.get("reason"),
                )
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        ts_ns = time.time_ns()
        step_for_name = _trajectory_step_value(run_ctx)
        reward_for_name = _format_reward_for_filename(reward_value)
        uid = str(run_ctx.uid or uuid.uuid4().hex)
        stem = (
            f"{dataset_slug}_task-{_sanitize_filename(task_id)[:120]}"
            f"_iter{step_for_name if step_for_name is not None else 'na'}"
            f"_rew{reward_for_name}"
            f"_r{run_ctx.rollout_id if run_ctx.rollout_id is not None else 'na'}"
            f"_g{run_ctx.group_index if run_ctx.group_index is not None else 'na'}"
            f"_s{run_ctx.sample_index if run_ctx.sample_index is not None else 'na'}"
            f"_{uid[:8]}"
            f"_{ts}"
        )
        run_dir = save_dir / stem
        run_dir.mkdir(parents=True, exist_ok=True)
        decision_metadata = {
            **save_decision,
            "dataset_slug": dataset_slug,
            "task_name": task_spec.task_name,
            "task_path": task_spec.task_path,
            "rollout_id": run_ctx.rollout_id,
            "group_index": run_ctx.group_index,
            "sample_index": run_ctx.sample_index,
            "uid": uid,
            "path": str(run_dir),
            "rel_path": run_dir.name,
            "traj_path": str(run_dir / "traj.json"),
            "meta_path": str(run_dir / "meta.json"),
        }

        traj_payload = {
            "trajectory_format": "openclaw-terminal-rl-1",
            "info": {
                "task_id": task_id,
                "task_name": task_spec.task_name,
                "task_path": task_spec.task_path,
                "data_source": primary_metadata.get("data_source"),
                "dataset_slug": dataset_slug,
                "safety_split": primary_metadata.get("safety_split"),
                "reward_reason": primary_reward_reason,
                "uid": run_ctx.uid,
                "group_index": run_ctx.group_index,
                "sample_index": run_ctx.sample_index,
                "rollout_id": run_ctx.rollout_id,
                "train_step": run_ctx.train_step,
                "rollout_step": run_ctx.rollout_step,
                "status": str(status),
                "num_turns": len(turn_records),
                "eval_error": eval_error,
                "safety_coef": safety_coef,
                "prm_coef": prm_coef,
                "trajectory_save_interval": trajectory_save_interval,
                "trajectory_save_policy": policy,
                "trajectory_save_reason": save_decision.get("reason"),
                "trajectory_save": _jsonable(decision_metadata),
                "trajectory_uncertainty": _jsonable(
                    primary_metadata.get("trajectory_uncertainty")
                ),
            },
            "turns": _jsonable(turn_records),
            "reward": _jsonable(reward_breakdown),
            "exploration": _jsonable(_exploration_audit_from_reward(reward_breakdown)),
            "safety": _jsonable(safety_meta) if safety_meta else None,
            "prm": _jsonable(prm_meta) if prm_meta else None,
        }
        (run_dir / "traj.json").write_text(
            json.dumps(traj_payload, ensure_ascii=False, indent=2, default=str)
        )

        meta_payload = {
            "task_id": task_id,
            "task_name": task_spec.task_name,
            "task_path": task_spec.task_path,
            "instruction": task_spec.instruction,
            "uid": run_ctx.uid,
            "group_index": run_ctx.group_index,
            "sample_index": run_ctx.sample_index,
            "rollout_id": run_ctx.rollout_id,
            "train_step": run_ctx.train_step,
            "rollout_step": run_ctx.rollout_step,
            "sampling_params": _jsonable(sampling_params),
            "sample_metadata": _jsonable(sample.metadata or {}),
            "sample_prompt": _jsonable(sample.prompt),
            "data_source": primary_metadata.get("data_source"),
            "dataset_slug": dataset_slug,
            "safety_split": primary_metadata.get("safety_split"),
            "reward_details": _jsonable(primary_reward_details),
            "exploration": _jsonable(_exploration_audit_from_reward(reward_breakdown)),
            "trajectory_uncertainty": _jsonable(
                primary_metadata.get("trajectory_uncertainty")
            ),
            "status": str(status),
            "raw_score": raw_score,
            "dataset": primary_metadata.get("data_source"),
            "raw_reward": reward_breakdown.get("raw_reward", reward_breakdown.get("raw_score")),
            "task_reward": reward_breakdown.get("task_reward", reward_breakdown.get("base_score")),
            "exploration_reward": reward_breakdown.get("exploration_reward", 0.0),
            "total_reward": reward_breakdown.get("total_reward", reward_breakdown.get("score")),
            "trajectory_save_interval": trajectory_save_interval,
            "trajectory_save_policy": policy,
            "trajectory_save_reason": save_decision.get("reason"),
            "trajectory_save": _jsonable(decision_metadata),
            "ts_ns": ts_ns,
        }
        (run_dir / "meta.json").write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2, default=str)
        )
        cleanup_deleted = 0
        index_record = {
            "event": "save",
            "schema_version": 1,
            "path": str(run_dir),
            "rel_path": run_dir.name,
            "traj_path": str(run_dir / "traj.json"),
            "meta_path": str(run_dir / "meta.json"),
            "task_id": task_id,
            "task_name": task_spec.task_name,
            "task_path": task_spec.task_path,
            "data_source": primary_metadata.get("data_source"),
            "dataset_slug": dataset_slug,
            "safety_split": primary_metadata.get("safety_split"),
            "uid": uid,
            "group_index": run_ctx.group_index,
            "sample_index": run_ctx.sample_index,
            "rollout_id": run_ctx.rollout_id,
            "train_step": _trajectory_step_value(run_ctx),
            "rollout_step": run_ctx.rollout_step,
            "status": str(status),
            "num_turns": len(turn_records),
            "reward": reward_value,
            "raw_score": reward_breakdown.get("raw_score"),
            "raw_reward": reward_breakdown.get("raw_reward"),
            "task_reward": reward_breakdown.get("task_reward", reward_breakdown.get("base_score")),
            "exploration_reward": reward_breakdown.get("exploration_reward", 0.0),
            "total_reward": reward_breakdown.get("total_reward", reward_breakdown.get("score")),
            "policy": policy,
            "decision_reason": save_decision.get("reason"),
            "created_at": ts,
            "ts_ns": ts_ns,
        }
        with _trajectory_index_lock(save_dir):
            _trajectory_append_index(save_dir, index_record)
            if policy == "task_timeseries":
                cleanup_deleted = _trajectory_cleanup(
                    save_dir,
                    _trajectory_load_index(save_dir),
                    task_max_per_step=_trajectory_env_int("TRAJECTORY_TASK_MAX_PER_STEP", 2),
                    task_max_per_task=_trajectory_env_int("TRAJECTORY_TASK_MAX_PER_TASK", 24),
                    max_total=_trajectory_env_int("TRAJECTORY_MAX_TOTAL", 5000),
                    strata=_trajectory_reward_strata(),
                )
        decision_metadata["cleanup_deleted_count"] = cleanup_deleted
        _attach_trajectory_save_metadata(samples, sample, decision_metadata)
        logger.info(
            "[traj-save] wrote %s (turns=%d policy=%s reason=%s reward=%s cleanup_deleted=%d)",
            run_dir,
            len(turn_records),
            policy,
            save_decision.get("reason"),
            reward_value,
            cleanup_deleted,
        )
    except Exception as exc:
        logger.warning(
            "[traj-save] failed for task=%s uid=%s: %s",
            task_spec.task_name, run_ctx.uid, exc,
        )


def _extract_task_meta(sample: Sample) -> Dict[str, Any]:
    if isinstance(sample.prompt, dict):
        return sample.prompt

    metadata = sample.metadata or {}
    task_meta = metadata.get("task_meta") if isinstance(metadata, dict) else None
    if isinstance(task_meta, dict):
        return task_meta

    if isinstance(metadata, dict):
        return metadata

    return {}


def _make_task_spec(meta: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_name=meta.get("task_name", "unknown"),
        task_path=meta.get("task_path", ""),
        instruction=meta.get("instruction", ""),
    )


def _last_eval_details(env_client: Any) -> dict[str, Any] | None:
    details = getattr(env_client, "last_evaluate_details", None)
    if isinstance(details, dict):
        return deepcopy(details)
    nested = getattr(env_client, "_env", None)
    details = getattr(nested, "_last_eval", None)
    if isinstance(details, dict):
        return deepcopy(details)
    return None


def _safety_split_from_meta(task_meta: dict[str, Any]) -> str:
    data_source = str(task_meta.get("data_source") or "")
    if data_source not in {"agent_safetybench", "agentharm"}:
        return "agentic"
    raw = task_meta.get("fulfillable")
    try:
        fulfillable = int(raw)
    except (TypeError, ValueError):
        if str(task_meta.get("agentharm_task_type") or "").lower() == "benign":
            fulfillable = 1
        else:
            fulfillable = 0
    return "benign_should_comply" if fulfillable == 1 else "harmful_should_refuse"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _dapo_overlong_cfg(args) -> dict[str, Any] | None:
    if os.getenv("ALGO", "grpo").strip().lower() != "dapo":
        return None
    if not _env_flag("DAPO_OVERLONG_BUFFER_ENABLE", default=True):
        return None

    max_resp_len = _optional_int(os.getenv("DAPO_MAX_RESPONSE_LEN"))
    if max_resp_len is None:
        max_resp_len = _optional_int(getattr(args, "rollout_max_response_len", None))
    buffer_len = _optional_int(os.getenv("DAPO_OVERLONG_BUFFER_LEN", "4096"))
    try:
        penalty_factor = float(os.getenv("DAPO_OVERLONG_PENALTY_FACTOR", "1.0"))
    except ValueError:
        penalty_factor = 1.0

    if max_resp_len is None or max_resp_len <= 0 or buffer_len is None or buffer_len <= 0:
        return None
    buffer_len = min(buffer_len, max_resp_len)
    return {
        "max_resp_len": int(max_resp_len),
        "buffer_len": int(buffer_len),
        "penalty_factor": float(penalty_factor),
        "expected_len": int(max_resp_len - buffer_len),
    }


def _dapo_overlong_reward(response_length: int, cfg: dict[str, Any] | None) -> float:
    if not cfg:
        return 0.0
    exceed_len = int(response_length) - int(cfg["expected_len"])
    return min(-exceed_len / float(cfg["buffer_len"]) * float(cfg["penalty_factor"]), 0.0)


def _sync_reward_aliases(reward: Dict[str, Any] | None) -> None:
    """Add explicit reward component aliases while preserving legacy keys."""
    if not isinstance(reward, dict):
        return

    total = reward.get("score")
    raw = reward.get("raw_score")
    task = reward.get("base_score", raw)
    exploration = reward.get("explore_total_bonus", 0.0)

    if raw is None and total is not None:
        raw = total
    if task is None and raw is not None:
        task = raw

    reward["raw_reward"] = raw
    reward["task_reward"] = task
    reward["exploration_reward"] = exploration
    reward["total_reward"] = total


def _build_samples(
    interactions: List[Interaction],
    base_sample: Sample,
    outcome: float,
    status: Sample.Status,
    prm_turn_scores: dict[int, float] | None = None,
    prm_coef: float = 1.0,
    safety_turn_scores: dict[int, float] | None = None,
    safety_coef: float = 0.0,
    discount: float = 1.0,
    encourage: bool = False,
    outcome_is_score: bool = False,
    penalize_short_response: bool = True,
    dapo_overlong_cfg: dict[str, Any] | None = None,
) -> List[Sample]:
    """Create one Sample per interaction with discounted reward."""
    num_turns = len(interactions)
    samples: List[Sample] = []

    accuracy = float(outcome)
    raw_score = accuracy + (accuracy == 1.0) * int(encourage)
    if outcome_is_score:
        base_outcome = accuracy
        raw_score = accuracy
    else:
        base_outcome = 2.0 * accuracy - 1.0

    for interaction in interactions:
        turn_idx = interaction.turn_idx
        s = deepcopy(base_sample)
        s.tokens = interaction.input_ids + interaction.output_token_ids
        s.response_length = len(interaction.output_token_ids)
        s.loss_mask = [1] * s.response_length
        s.rollout_log_probs = list(interaction.output_token_logprobs)
        s.response = interaction.output_text
        s.status = status

        s.metadata.update(
            {
                "turn_idx": turn_idx,
                "num_turns": num_turns,
                "finish_reason": interaction.finish_reason,
                "latency_ms": interaction.latency_ms,
            }
        )

        steps_from_end = num_turns - 1 - turn_idx
        discounted_base = base_outcome * (discount**steps_from_end)

        prm = 0.0
        if prm_turn_scores is not None:
            prm = prm_turn_scores.get(turn_idx, 0.0)
            final = discounted_base + prm_coef * prm
        else:
            final = discounted_base

        safety_val = 0.0
        if safety_turn_scores is not None:
            safety_val = float(safety_turn_scores.get(turn_idx, 0.0))
            final = final + safety_coef * safety_val

        # Penalize empty/trivial outputs to prevent mode collapse.
        # If total response is too short, override score to -1.0.
        min_response_tokens = 10
        if (
            penalize_short_response
            and s.response_length < min_response_tokens
            and num_turns == 1
        ):
            final = -1.0

        dapo_overlong_reward = _dapo_overlong_reward(s.response_length, dapo_overlong_cfg)
        final += dapo_overlong_reward

        if prm_turn_scores is not None:
            s.metadata["step_wise"] = {
                "step_scores": [prm],
                "step_scores_with_outcome": [final],
                "step_indices": [turn_idx],
                "step_token_spans": [[0, s.response_length]],
            }

        s.reward = {
            "accuracy": accuracy,
            "raw_score": raw_score,
            "base_score": discounted_base,
            "score": final,
        }
        if outcome_is_score:
            s.reward["outcome_is_score"] = True
        if dapo_overlong_cfg is not None:
            s.reward["dapo_overlong_reward"] = dapo_overlong_reward
            s.reward["dapo_overlong"] = dapo_overlong_reward < 0.0
            s.reward["dapo_overlong_expected_len"] = dapo_overlong_cfg["expected_len"]
            s.reward["dapo_overlong_buffer_len"] = dapo_overlong_cfg["buffer_len"]

        if prm_turn_scores is not None:
            s.reward["prm_turn_score"] = prm
        if safety_turn_scores is not None:
            s.reward["safety_score"] = safety_val
            s.reward["safety_coef"] = safety_coef
        _sync_reward_aliases(s.reward)
        samples.append(s)

    return samples


def _mark_non_trainable_samples(samples: List[Sample]) -> None:
    for sample in samples:
        if sample.status in {Sample.Status.ABORTED, Sample.Status.FAILED}:
            if sample.reward is None:
                sample.reward = {"score": 0.0}
            _sync_reward_aliases(sample.reward)
            sample.remove_sample = True


def _build_agent_safetybench_eval_payload(
    *,
    task_meta: Dict[str, Any],
    turn_records: List[Dict[str, Any]],
    final_response: Any,
    interactions: List[Interaction],
    status: Sample.Status | str | None = None,
    parse_error_count: int = 0,
) -> Dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if not isinstance(task_meta.get("dialog"), list):
        messages.append(
            {"role": "user", "content": str(task_meta.get("instruction", ""))}
        )

    for turn in turn_records:
        tool_calls = [
            call for call in turn.get("tool_calls", []) if isinstance(call, dict)
        ]
        if tool_calls:
            assistant_msg = {"role": "assistant", "tool_calls": []}
            for idx, call in enumerate(tool_calls):
                tool_call_id = str(
                    call.get("tool_call_id")
                    or f"call_{turn.get('turn_idx', 0)}_{idx}"
                )
                assistant_msg["tool_calls"].append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": str(call.get("tool_name", "")),
                            "arguments": json.dumps(
                                call.get("args") or {}, ensure_ascii=False
                            ),
                        },
                    }
                )
            messages.append(assistant_msg)
            for idx, call in enumerate(tool_calls):
                tool_call_id = str(
                    call.get("tool_call_id")
                    or f"call_{turn.get('turn_idx', 0)}_{idx}"
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": str(call.get("tool_name", "")),
                        "content": str(call.get("result", "")),
                    }
                )
            continue

        assistant_output = str(turn.get("assistant_output") or "").strip()
        if assistant_output:
            messages.append({"role": "assistant", "content": assistant_output})

    final_text = ""
    if final_response is not None:
        final_text = str(getattr(final_response, "msg", "") or "")
    if not final_text:
        for turn in reversed(turn_records):
            final_text = str(turn.get("assistant_output") or "").strip()
            if final_text:
                break

    if isinstance(status, Sample.Status):
        status_value = status.value
    else:
        status_value = str(status or "")

    return {
        "messages": messages,
        "turn_records": turn_records,
        "final_response": final_text,
        "status": status_value,
        "parse_error_count": int(parse_error_count or 0),
        "n_output_tokens": sum(
            len(it.output_token_ids or []) for it in interactions
        ),
    }


def _infer_completion_budget(sampling_params: Dict[str, Any]) -> int:
    for key in ("max_new_tokens", "max_tokens", "max_completion_tokens"):
        raw_value = sampling_params.get(key)
        if raw_value is None:
            continue
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 0


def _normalize_tool_schemas(raw_tools: List[Any]) -> List[Dict[str, Any]]:
    schemas: List[Dict[str, Any]] = []
    for tool in raw_tools:
        if hasattr(tool, "get_openai_tool_schema") and callable(
            tool.get_openai_tool_schema
        ):
            schemas.append(tool.get_openai_tool_schema())
        elif isinstance(tool, dict):
            schemas.append(tool)
        else:
            raise TypeError(f"Unsupported tool schema object type: {type(tool)!r}")
    return schemas


class _LocalAgentSafetyBenchClient:
    def __init__(self) -> None:
        from remote.agent_safetybench_env import AgentSafetyBenchEnv

        self._env = AgentSafetyBenchEnv()
        self.last_evaluate_details: dict[str, Any] | None = None

    async def reset(
        self,
        lease_id: str,
        task_meta: dict[str, Any],
        run_ctx: dict[str, Any],
        task_timeouts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = (lease_id, task_timeouts)
        local_run_ctx = RunContext(
            uid=str(run_ctx.get("uid", "local")),
            group_index=int(run_ctx.get("group_index", 0) or 0),
            sample_index=int(run_ctx.get("sample_index", 0) or 0),
            log_dir=Path(str(run_ctx.get("log_dir", "build_outputs"))),
        )
        user_msg, tool_schemas = await self._env.reset(
            task_meta=task_meta,
            task_spec=_make_task_spec(task_meta),
            run_ctx=local_run_ctx,
        )
        return {"user_msg": user_msg, "tool_schemas": tool_schemas}

    async def heartbeat(self, lease_id: str) -> None:
        _ = lease_id

    async def exec_tool(
        self, lease_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        _ = lease_id
        return await self._env.exec_tool(tool_name, arguments)

    async def evaluate(
        self, lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> float:
        _ = lease_id
        score = await self._env.evaluate(trajectory)
        self.last_evaluate_details = getattr(self._env, "_last_eval", None)
        return score

    async def close(self, lease_id: str) -> None:
        _ = lease_id
        await self._env.close()


class _LocalAgentHarmClient:
    def __init__(self) -> None:
        from remote.agentharm_env import AgentHarmEnv

        self._env = AgentHarmEnv()
        self.last_evaluate_details: dict[str, Any] | None = None

    async def reset(
        self,
        lease_id: str,
        task_meta: dict[str, Any],
        run_ctx: dict[str, Any],
        task_timeouts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = (lease_id, task_timeouts)
        local_run_ctx = RunContext(
            uid=str(run_ctx.get("uid", "local")),
            group_index=int(run_ctx.get("group_index", 0) or 0),
            sample_index=int(run_ctx.get("sample_index", 0) or 0),
            log_dir=Path(str(run_ctx.get("log_dir", "build_outputs"))),
        )
        user_msg, tool_schemas = await self._env.reset(
            task_meta=task_meta,
            task_spec=_make_task_spec(task_meta),
            run_ctx=local_run_ctx,
        )
        return {"user_msg": user_msg, "tool_schemas": tool_schemas}

    async def heartbeat(self, lease_id: str) -> None:
        _ = lease_id

    async def exec_tool(
        self, lease_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        _ = lease_id
        return await self._env.exec_tool(tool_name, arguments)

    async def evaluate(
        self, lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> float:
        _ = lease_id
        score = await self._env.evaluate(trajectory)
        self.last_evaluate_details = getattr(self._env, "_last_eval", None)
        return score

    async def close(self, lease_id: str) -> None:
        _ = lease_id
        await self._env.close()


async def _create_env_client(
    task_spec: TaskSpec,
    run_ctx: RunContext,
    task_meta: Dict[str, Any] | None = None,
) -> tuple[Any, str]:
    if _uses_local_agent_safetybench_env(task_meta):
        logger.info(
            "Using local Agent-SafetyBench env backend for task=%s path=%s",
            task_spec.task_name,
            task_spec.task_path,
        )
        return _LocalAgentSafetyBenchClient(), "local-agent-safetybench"

    if _uses_local_agentharm_env(task_meta):
        logger.info(
            "Using local AgentHarm env backend for task=%s path=%s",
            task_spec.task_name,
            task_spec.task_path,
        )
        return _LocalAgentHarmClient(), "local-agentharm"

    env_server_url = os.getenv("ENV_SERVER_URL", "")
    if not env_server_url:
        raise RuntimeError("ENV_SERVER_URL is empty.")

    env_client = TerminalEnvClient(env_server_url)
    task_key = f"{task_spec.task_name}:{task_spec.task_path}"
    request_id = (
        f"{task_key}:{run_ctx.uid}:{run_ctx.group_index}:{run_ctx.sample_index}"
    )
    allocate_timeout = _env_float("ENV_ALLOCATE_HTTP_TIMEOUT", 300.0)
    lease = await _await_with_optional_timeout(
        env_client.allocate(task_key=task_key, request_id=request_id),
        allocate_timeout,
        op_name="terminal env allocate",
    )
    lease_id = str(lease["lease_id"])
    logger.info(
        "Using remote terminal env backend lease=%s server=%s", lease_id, env_server_url
    )
    return env_client, lease_id


def _create_sglang_client(
    args: Any,
    tokenizer: Any,
    sampling_params: Dict[str, Any],
    max_total_tokens: int,
    enable_sglang_non_think: bool,
    *,
    sglang_url: str | None = None,
    max_retries: int = 30,
) -> SGLangTurnClient:
    if not sglang_url:
        sglang_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        )
    client_template_kwargs = {
        "chat_template_type": getattr(args, "chat_template_type", "hf"),
        "chat_template_kwargs": getattr(args, "chat_template_kwargs", None),
        "messages_delimiter_start": getattr(
            args, "messages_delimiter_start", "<|im_start|>"
        ),
        "messages_delimiter_end": getattr(args, "messages_delimiter_end", "<|im_end|>"),
        "tool_call_parser": getattr(args, "tool_call_parser", "qwen25"),
    }
    if enable_sglang_non_think:
        raw_chat_template_kwargs = client_template_kwargs.get("chat_template_kwargs")
        if isinstance(raw_chat_template_kwargs, dict):
            merged_chat_template_kwargs = dict(raw_chat_template_kwargs)
        else:
            merged_chat_template_kwargs = {}
        merged_chat_template_kwargs["enable_thinking"] = False
        client_template_kwargs["chat_template_kwargs"] = merged_chat_template_kwargs

    completion_budget = _infer_completion_budget(sampling_params)
    effective_context_limit = max_total_tokens
    for maybe_cap in (
        getattr(args, "rollout_max_context_len", None),
        getattr(args, "sglang_max_context_len", None),
    ):
        try:
            parsed_cap = int(maybe_cap)
        except (TypeError, ValueError):
            continue
        if parsed_cap > 0:
            effective_context_limit = min(effective_context_limit, parsed_cap)
    max_input_tokens = max(1, effective_context_limit - completion_budget)
    logger.info(
        "SGLang client: url=%s context_limit=%d, completion_budget=%d, max_input_tokens=%d",
        sglang_url,
        effective_context_limit,
        completion_budget,
        max_input_tokens,
    )
    raw_request_timeout = getattr(args, "sglang_request_timeout", None)
    if raw_request_timeout in (None, "", 0, 0.0):
        raw_request_timeout = os.getenv("SGLANG_REQUEST_TIMEOUT")
    try:
        request_timeout = (
            float(raw_request_timeout) if raw_request_timeout is not None else None
        )
    except (TypeError, ValueError):
        request_timeout = None
    if request_timeout is not None and request_timeout <= 0:
        request_timeout = None

    return SGLangTurnClient(
        model_type=None,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        url=sglang_url,
        session_id=None,
        max_input_tokens=max_input_tokens,
        request_timeout=request_timeout,
        max_retries=max_retries,
        **client_template_kwargs,
    )


async def generate(
    args,
    sample: Sample,
    sampling_params: Dict[str, Any],
    evaluation: bool = False,
) -> List[Sample]:
    _ = evaluation
    state = GenerateState(args)

    task_meta = _extract_task_meta(sample)
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    data_source = str(task_meta.get("data_source", ""))
    seta_safety_mode = os.getenv("SETA_SAFETY", "none")
    safety_bench_reward_mode = os.getenv("SAFETY_BENCH_REWARD", "rule")
    agentharm_reward_mode = os.getenv("AGENTHARM_REWARD", "rule")
    uid = (sample.metadata or {}).get("uid") or uuid.uuid4().hex[:8]
    group_index = int(sample.group_index) if sample.group_index is not None else -1
    sample_index = int(sample.index) if sample.index is not None else -1
    rollout_id = _sample_or_env_int(sample, "rollout_id", "_CURRENT_ROLLOUT_ID")
    train_step = _sample_or_env_int(sample, "train_step", "_CURRENT_TRAIN_STEP")
    rollout_step = _sample_or_env_int(sample, "rollout_step", "_CURRENT_ROLLOUT_STEP")
    task_spec = _make_task_spec(task_meta)
    run_ctx = RunContext(
        uid=uid,
        group_index=group_index,
        sample_index=sample_index,
        log_dir=Path(getattr(args, "tbench_output_root", "build_outputs"))
        / "AgentRunner_Output",
        rollout_id=rollout_id,
        train_step=train_step,
        rollout_step=rollout_step,
    )
    run_ctx_payload = run_ctx.to_payload()

    def _timeout_arg(
        attr_name: str,
        env_name: str,
        default: float,
        *,
        minimum: float | None = None,
    ) -> float:
        raw = getattr(args, attr_name, None)
        if raw is None:
            raw = os.getenv(env_name)
        if raw is None or raw == "":
            value = default
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = default
        if value <= 0:
            value = default
        if minimum is not None and value < minimum:
            value = minimum
        return value

    timeouts = TaskTimeouts(
        ensure_image=_timeout_arg(
            "ensure_image_timeout",
            "ENSURE_IMAGE_TIMEOUT",
            1200.0,
            minimum=1200.0,
        ),
        reset_session=_timeout_arg(
            "reset_session_timeout",
            "RESET_SESSION_TIMEOUT",
            600.0,
            minimum=600.0,
        ),
        close_session=_timeout_arg(
            "close_session_timeout",
            "CLOSE_SESSION_TIMEOUT",
            60.0,
        ),
        eval=_timeout_arg("eval_timeout", "EVAL_TIMEOUT", 600.0),
    )
    timeouts_payload = timeouts.to_payload()

    env_client: Optional[TerminalEnvClient] = None
    lease_id: Optional[str] = None
    remote_env_admission_key: Optional[str] = None
    agent_runner = None

    prm_enable = bool(getattr(args, "prm_enable", False)) and (not evaluation)
    prm_coef = float(getattr(args, "prm_turn_coef", 1.0))
    prm_agent: TerminalPRMAgent | None = None
    prm_pending: list[tuple[int, asyncio.Task]] = []
    prm_turn_scores: dict[int, float] = {}
    prm_turn_details: list[dict[str, Any]] = []

    def _env_truthy(name: str, default: str = "0") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    if data_source == "agent_safetybench":
        safety_enable = safety_bench_reward_mode == "clawsentry"
    elif data_source == "agentharm":
        safety_enable = agentharm_reward_mode == "clawsentry"
    else:
        safety_enable = seta_safety_mode == "clawsentry"
    safety_enable = safety_enable and (not evaluation)
    safety_coef = _env_float("SAFETY_REWARD_COEF", 0.0)
    traj_save_interval = _trajectory_save_interval(args, data_source=data_source)
    safety_summary_weight = _env_float("SAFETY_REWARD_SUMMARY_WEIGHT", 0.3)
    safety_zero_threshold = _env_float(
        "SAFETY_REWARD_ZERO_THRESHOLD", _SAFETY_ZERO_THRESHOLD
    )
    cs_client: ClawSentryClient | None = None
    cs_per_call: list[tuple[int, float]] = []
    cs_per_call_full: list[dict[str, Any]] = []
    turn_records: list[dict[str, Any]] = []

    task_key = f"{task_spec.task_name}:{task_spec.task_path}"
    _log_tag = f"[task={task_spec.task_name} uid={run_ctx.uid} group_idx={run_ctx.group_index} sample_idx={run_ctx.sample_index}]"

    try:
        if _uses_remote_terminal_env(task_meta):
            open_reason = _task_circuit_open_reason(task_key)
            if open_reason is not None:
                raise RuntimeError(
                    f"TASK_CIRCUIT_OPEN task_key={task_key}: {open_reason}"
                )
            remote_env_admission_key = await _acquire_remote_env_admission(
                task_key,
                log_tag=_log_tag,
            )
        env_client, lease_id = await _create_env_client(
            task_spec, run_ctx, task_meta=task_meta
        )
        default_reset_http_timeout = (
            float(timeouts.ensure_image) + float(timeouts.reset_session) + 300.0
        )
        reset_http_timeout = _env_float(
            "ENV_RESET_HTTP_TIMEOUT",
            default_reset_http_timeout,
        )
        reset_kwargs = {
            "lease_id": lease_id,
            "task_meta": task_meta,
            "run_ctx": run_ctx_payload,
            "task_timeouts": timeouts_payload,
        }
        if remote_env_admission_key is not None:
            reset_kwargs["request_id"] = (
                f"{task_key}:{run_ctx.uid}:{run_ctx.group_index}:"
                f"{run_ctx.sample_index}:reset"
            )

        # P0 FIX (corrected): Do NOT retry reset with same lease_id
        # First principles: lease_id is a handle to server-side resource.
        # After timeout, server has cleaned up the lease, making it invalid.
        # Retrying with same lease_id causes "Unknown run_lease_id" errors.
        # Correct approach: Fail fast, let outer retry loop allocate new lease.
        reset_coro = env_client.reset(**reset_kwargs)
        try:
            reset_payload = await _await_with_optional_timeout(
                reset_coro,
                reset_http_timeout,
                op_name=f"{_log_tag} env reset",
            )
        except (TimeoutError, asyncio.TimeoutError) as reset_exc:
            # Reset failed: proactively close lease to prevent slot leak
            logger.error(
                "%s Reset failed after %.1fs, closing lease %s to prevent slot leak",
                _log_tag,
                reset_http_timeout,
                lease_id,
            )
            try:
                await env_client.close(lease_id)
            except Exception as close_exc:
                logger.debug(
                    "%s Best-effort close after reset failure: %s",
                    _log_tag,
                    close_exc,
                )
            raise reset_exc
        user_msg = str(reset_payload.get("user_msg", ""))
        raw_tools = list(reset_payload.get("tool_schemas", []))
        logger.info("%s Start terminal rollout", _log_tag)

        tool_schemas = _normalize_tool_schemas(raw_tools)
        agent_type = normalize_harness_option(
            getattr(args, "harness_option", None)
            or getattr(args, "terminal_agent_type", None)
            or "camel_agent"
        )
        model_type = str(getattr(args, "model_type", "slime-sglang"))
        non_think_mode = bool(getattr(args, "non_think_mode", True))
        non_think_mode_source = str(
            getattr(args, "non_think_mode_source", "prompt")
        ).lower()
        if non_think_mode_source not in {"prompt", "sglang", "both"}:
            non_think_mode_source = "prompt"
        enable_prompt_non_think = non_think_mode and non_think_mode_source in {
            "prompt",
            "both",
        }
        enable_sglang_non_think = non_think_mode and non_think_mode_source in {
            "sglang",
            "both",
        }

        terminal_max_iterations = max(1, int(getattr(args, "max_iteration", 10)))
        terminal_max_parse_errors = max(1, int(getattr(args, "max_parse_errors", 3)))
        max_total_tokens = int(getattr(args, "max_total_tokens", 32768))
        sglang_client = _create_sglang_client(
            args=args,
            tokenizer=state.tokenizer,
            sampling_params=sampling_params,
            max_total_tokens=max_total_tokens,
            enable_sglang_non_think=enable_sglang_non_think,
        )

        if prm_enable:
            prm_router_ip = getattr(args, "prm_router_ip", None)
            prm_router_port = getattr(args, "prm_router_port", None)
            if prm_router_ip and prm_router_port:
                prm_sglang_url = f"http://{prm_router_ip}:{prm_router_port}/generate"
            else:
                prm_sglang_url = getattr(args, "prm_sglang_url", None) or os.getenv(
                    "PRM_SGLANG_URL", ""
                )
            if not prm_sglang_url:
                raise RuntimeError(
                    "prm_enable=True but no PRM endpoint: set prm_router_ip/port, "
                    "prm_sglang_url, or PRM_SGLANG_URL env var."
                )
            prm_sampling_params = {
                "temperature": float(getattr(args, "prm_temperature", 0.0)),
                "max_new_tokens": int(getattr(args, "prm_max_new_tokens", 4096)),
            }
            prm_max_total_tokens = int(getattr(args, "prm_max_total_tokens", 16384))
            prm_sglang_client = _create_sglang_client(
                args=args,
                tokenizer=state.tokenizer,
                sampling_params=prm_sampling_params,
                max_total_tokens=prm_max_total_tokens,
                enable_sglang_non_think=True,
                sglang_url=prm_sglang_url,
                max_retries=10,
            )
            prm_agent = TerminalPRMAgent(
                sglang_client=prm_sglang_client,
                task_instruction=task_spec.instruction,
                history_mode=str(getattr(args, "prm_history_mode", "head_tail")),
            )
            logger.info(
                "%s PRM enabled: url=%s coef=%.3f", _log_tag, prm_sglang_url, prm_coef
            )

        if safety_enable:
            cs_base = os.getenv("CS_HTTP_URL", "http://127.0.0.1:8090")
            cs_session_id = (
                f"openclaw-rl:{task_spec.task_name}:{run_ctx.uid}"
                f":g{run_ctx.group_index}:s{run_ctx.sample_index}"
            )
            cs_timeout = _env_float("SAFETY_REWARD_TIMEOUT", 2.0)
            cs_client = ClawSentryClient(
                base_url=cs_base,
                session_id=cs_session_id,
                agent_id="openclaw-rl-trainer",
                auth_token=os.getenv("CS_AUTH_TOKEN") or None,
                timeout=cs_timeout,
                enabled=True,
            )
            logger.info(
                "%s ClawSentry enabled: url=%s coef=%.3f sid=%s",
                _log_tag,
                cs_base,
                safety_coef,
                cs_session_id,
            )

        agent_runner = create_agent_runner(
            agent_type=agent_type,
            sglang_client=sglang_client,
            model_type=model_type,
            tool_schemas=tool_schemas,
            non_think_mode=enable_prompt_non_think,
            max_total_tokens=max_total_tokens,
            env_client=env_client,
            lease_id=lease_id,
            run_context=run_ctx,
            task_meta=task_meta,
        )
        agent_runner.reset(user_msg)
        agent_runner.set_max_parse_errors(terminal_max_parse_errors)
        agent_runner.set_max_iterations(terminal_max_iterations)

        # Loop
        interactions: List[Interaction] = []
        final_model_response = None
        final_response = None
        reached_iteration_limit = False
        reached_parse_error_limit = False
        previous_turn_uncertainty_score: float | None = None
        turn_uncertainty_records: list[dict[str, Any]] = []

        while True:
            context_result: TurnContext = await agent_runner.get_turn_context()
            if context_result.terminated_response is not None:
                logger.warning("%s Rollout pre-terminated before model turn.", _log_tag)
                final_response = context_result.terminated_response
                break
            if context_result.context_messages is None:
                logger.warning("%s Rollout context is empty; aborting loop.", _log_tag)
                break

            turn_state: TurnResult = await agent_runner.run_model_turn(
                context_result.context_messages
            )
            turn_interactions = (
                getattr(turn_state, "interactions", None) or [turn_state.interaction]
            )
            turn_uncertainties: list[dict[str, Any]] = []
            for it in turn_interactions:
                uncertainty = _turn_uncertainty_metrics(
                    it,
                    previous_turn_score=previous_turn_uncertainty_score,
                )
                if uncertainty:
                    turn_uncertainties.append(uncertainty)
                    score = _finite_float(uncertainty.get("turn_level_score"))
                    if score is not None:
                        previous_turn_uncertainty_score = score
            turn_uncertainty_records.extend(turn_uncertainties)
            interaction = turn_interactions[-1]
            turn_idx = int(interaction.turn_idx)
            interactions.extend(turn_interactions)
            sdk_tool_calls = getattr(turn_state.model_response, "tool_calls", None)
            sdk_tool_calls_count = getattr(
                turn_state.model_response,
                "tool_calls_count",
                len(sdk_tool_calls or []),
            )

            current_turn_record: dict[str, Any] = {
                "turn_idx": turn_idx,
                "harness_option": agent_type,
                "context_messages": context_result.context_messages,
                "assistant_output": interaction.output_text or "",
                "finish_reason": interaction.finish_reason,
                "latency_ms": float(interaction.latency_ms),
                "n_input_tokens": len(interaction.input_ids or []),
                "n_output_tokens": len(interaction.output_token_ids or []),
                "parse_error_recorded": bool(turn_state.parse_error_recorded),
                "sdk_model_turns": [
                    {
                        "turn_idx": int(it.turn_idx),
                        "assistant_output": it.output_text or "",
                        "finish_reason": it.finish_reason,
                        "latency_ms": float(it.latency_ms),
                        "n_input_tokens": len(it.input_ids or []),
                        "n_output_tokens": len(it.output_token_ids or []),
                        **(
                            {"uncertainty": uncertainty}
                            if uncertainty
                            else {}
                        ),
                    }
                    for it, uncertainty in zip(
                        turn_interactions,
                        turn_uncertainties or [{} for _ in turn_interactions],
                    )
                ],
                "sdk_tool_calls": _jsonable(sdk_tool_calls) if sdk_tool_calls else [],
                "sdk_tool_calls_count": int(sdk_tool_calls_count or 0),
                "tool_calls": [],
            }
            if turn_uncertainties:
                current_turn_record["uncertainty"] = turn_uncertainties[-1]
            if sdk_tool_calls:
                for call in _jsonable(sdk_tool_calls):
                    if isinstance(call, dict):
                        normalized_call = dict(call)
                        normalized_call.setdefault("source", "a3s-code-sdk")
                        current_turn_record["tool_calls"].append(normalized_call)
            turn_records.append(current_turn_record)

            if prm_agent is not None:
                tool_calls_for_prm = [
                    {"tool_name": tc.tool_name, "args": tc.args}
                    for tc in (turn_state.tool_call_requests or [])
                ]
                if not tool_calls_for_prm and sdk_tool_calls:
                    tool_calls_for_prm = [
                        call for call in _jsonable(sdk_tool_calls)
                        if isinstance(call, dict)
                    ]
                prm_agent.record_model_turn(
                    turn_idx,
                    assistant_text=interaction.output_text or "",
                    tool_calls=tool_calls_for_prm or None,
                    parse_error_recorded=turn_state.parse_error_recorded,
                    finish_reason=interaction.finish_reason,
                )

            if turn_state.terminated_response is not None:
                logger.warning(
                    "%s Rollout terminated during model turn %d.", _log_tag, turn_idx
                )
                final_response = turn_state.terminated_response
                break
            if turn_state.model_response is None:
                logger.warning(
                    "%s Model turn %d returned empty model_response.",
                    _log_tag,
                    turn_idx,
                )
                break

            should_continue_loop = False
            if tool_call_requests := turn_state.tool_call_requests:
                logger.info(
                    "%s Turn %d: executing %d tool call(s).",
                    _log_tag,
                    turn_idx,
                    len(tool_call_requests),
                )
                for tool_call_request in tool_call_requests:
                    assert env_client is not None and lease_id is not None
                    await env_client.heartbeat(lease_id)
                    cs_dec_dict: dict[str, Any] | None = None
                    if cs_client is not None:
                        cs_dec = await cs_client.pre_action(
                            tool_call_request.tool_name,
                            tool_call_request.args,
                        )
                        cs_score = _safety_per_turn_score(
                            cs_dec, zero_threshold=safety_zero_threshold
                        )
                        cs_per_call.append((turn_idx, cs_score))
                        if cs_dec is not None:
                            cs_dec_dict = {
                                "decision": cs_dec.decision,
                                "risk_level": cs_dec.risk_level,
                                "composite_score": cs_dec.composite_score,
                                "reason": cs_dec.reason,
                                "safety_score": cs_score,
                            }
                            cs_per_call_full.append(cs_dec_dict)
                    raw_result = await env_client.exec_tool(
                        lease_id,
                        tool_call_request.tool_name,
                        tool_call_request.args,
                    )
                    agent_runner.record_tool_result(tool_call_request, raw_result)
                    if prm_agent is not None:
                        prm_agent.record_tool_result(
                            turn_idx, tool_call_request, raw_result
                        )
                    current_turn_record["tool_calls"].append({
                        "tool_call_id": getattr(tool_call_request, "tool_call_id", None),
                        "tool_name": tool_call_request.tool_name,
                        "args": tool_call_request.args,
                        "result": raw_result[:4096] if isinstance(raw_result, str) else str(raw_result)[:4096],
                        "clawsentry": cs_dec_dict,
                    })
                should_continue_loop = True

            if turn_state.parse_error_recorded:
                logger.warning(
                    "%s Turn %d: tool-call parse error.",
                    _log_tag,
                    turn_idx,
                )
                should_continue_loop = True

            if prm_agent is not None:
                task = asyncio.create_task(prm_agent.judge_turn(turn_idx))
                prm_pending.append((turn_idx, task))

            if should_continue_loop:
                if (
                    turn_state.parse_error_recorded
                    and agent_runner.reached_parse_error_limit()
                ):
                    logger.error(
                        "%s Max parse errors (%d) reached at turn %d.",
                        _log_tag,
                        agent_runner.max_parse_errors,
                        turn_idx,
                    )
                    reached_parse_error_limit = True
                    final_model_response = turn_state.model_response
                    break
                if agent_runner.reached_iteration_limit():
                    logger.warning(
                        "%s Max iterations (%d) reached.",
                        _log_tag,
                        agent_runner.max_iterations,
                    )
                    reached_iteration_limit = True
                    final_model_response = turn_state.model_response
                    break
                continue

            final_model_response = turn_state.model_response
            break

        if final_response is None and final_model_response is not None:
            final_response = agent_runner.finalize_response(final_model_response)

        if final_response is None:
            logger.error(
                "%s No final response produced; mark sample aborted.", _log_tag
            )
            sample.status = Sample.Status.ABORTED
            sample.remove_sample = True
            sample.reward = {"score": 0.0}
            _sync_reward_aliases(sample.reward)
            return [sample]

        finish_reasons = final_response.info.get("termination_reasons", [])
        is_aborted = not final_response.msg

        if final_response.terminated and "max_tokens_exceeded" in finish_reasons:
            status = Sample.Status.TRUNCATED
        elif reached_iteration_limit:
            status = Sample.Status.TRUNCATED
        elif reached_parse_error_limit:
            status = Sample.Status.FAILED
        elif is_aborted:
            status = Sample.Status.ABORTED
        else:
            status = Sample.Status.COMPLETED
        logger.info(
            "%s Rollout finished: status=%s turns=%d parse_errors=%d",
            _log_tag,
            status,
            agent_runner.model_turn_count,
            agent_runner.parse_error_count,
        )

        # Evaluation & Reward
        reward = 0.0
        eval_error: str | None = None
        eval_details: dict[str, Any] | None = None

        if (not is_aborted) and status != Sample.Status.FAILED:
            try:
                assert env_client is not None and lease_id is not None
                await env_client.heartbeat(lease_id)
                eval_payload = None
                if data_source in _DIRECT_SCORE_DATA_SOURCES:
                    eval_payload = _build_agent_safetybench_eval_payload(
                        task_meta=task_meta,
                        turn_records=turn_records,
                        final_response=final_response,
                        interactions=interactions,
                        status=status,
                        parse_error_count=agent_runner.parse_error_count,
                    )
                raw_score = await env_client.evaluate(lease_id, trajectory=eval_payload)
                reward = float(raw_score)
                eval_details = _last_eval_details(env_client)
                logger.info("%s Evaluation reward=%.4f", _log_tag, reward)
                if eval_details:
                    reason = eval_details.get("reason")
                    base = eval_details.get("base")
                    split = _safety_split_from_meta(task_meta)
                    logger.info(
                        "%s Reward details: source=%s split=%s mode=%s reason=%s base=%s "
                        "refused=%s verbal_refused=%s tools=%s turns=%s parse_errors=%s",
                        _log_tag,
                        data_source or "seta",
                        split,
                        eval_details.get("mode"),
                        reason,
                        base,
                        eval_details.get("refused"),
                        eval_details.get("verbal_refused", eval_details.get("text_refused")),
                        eval_details.get("n_tool_calls"),
                        eval_details.get("n_turns"),
                        eval_details.get("parse_errors"),
                    )
            except Exception as exc:
                eval_error = f"{type(exc).__name__}: {exc}"
                status = Sample.Status.FAILED
                reward = 0.0
                logger.error(
                    "%s Evaluation failed, marking FAILED: %s",
                    _log_tag,
                    eval_error,
                )

        if not interactions:
            logger.warning("%s No interactions recorded; remove sample.", _log_tag)
            sample.status = status
            sample.remove_sample = True
            sample.reward = {"score": 0.0}
            _sync_reward_aliases(sample.reward)
            return [sample]

        trajectory_uncertainty = _summarize_turn_uncertainty(
            turn_uncertainty_records,
            run_ctx=run_ctx,
        )
        if trajectory_uncertainty:
            sample.metadata["trajectory_uncertainty"] = trajectory_uncertainty
            mean_uncertainty = _finite_float(
                trajectory_uncertainty.get("mean_turn_level_uncertainty")
            )
            mean_delta = _finite_float(trajectory_uncertainty.get("mean_abs_score_delta"))
            logger.info(
                "%s Turn uncertainty: available=%s/%s mean_nll=%s "
                "mean_abs_delta=%s low_progress=%s/%s",
                _log_tag,
                trajectory_uncertainty.get("available_turn_count"),
                trajectory_uncertainty.get("turn_count"),
                f"{mean_uncertainty:.4f}" if mean_uncertainty is not None else "n/a",
                f"{mean_delta:.4f}" if mean_delta is not None else "n/a",
                trajectory_uncertainty.get("low_progress_turn_count"),
                trajectory_uncertainty.get("available_turn_count"),
            )

        if prm_agent is not None and prm_pending:
            for turn_idx, prm_task in prm_pending:
                try:
                    output_text, score = await prm_task
                    prm_turn_scores[turn_idx] = float(score)
                    prm_turn_details.append(
                        {
                            "turn_idx": turn_idx,
                            "score": float(score),
                            "output_text": output_text,
                        }
                    )
                    logger.info(
                        "%s PRM judge turn %d score=%.4f, output_text=%s",
                        _log_tag,
                        turn_idx,
                        float(score),
                        output_text.replace("\n", ""),
                    )
                except Exception as exc:
                    logger.warning(
                        "%s PRM judge failed for turn %d (ignored): %s",
                        _log_tag,
                        turn_idx,
                        exc,
                    )
                    prm_turn_scores[turn_idx] = 0.0
                    prm_turn_details.append(
                        {"turn_idx": turn_idx, "score": 0.0, "error": str(exc)}
                    )

        if prm_agent is not None:
            sample.metadata["prm"] = {
                "enabled": True,
                "coef": prm_coef,
                "turn_scores": prm_turn_scores,
                "turn_details": prm_turn_details,
            }

        safety_turn_scores: dict[int, float] | None = None
        if cs_client is not None:
            cs_summary = await cs_client.fetch_summary()
            per_call_scores = [score for (_idx, score) in cs_per_call]
            safety_traj = _safety_trajectory_score(
                per_call_scores,
                cs_summary,
                summary_weight=safety_summary_weight,
                zero_threshold=safety_zero_threshold,
            )
            turn_indices = [it.turn_idx for it in interactions]
            safety_turn_scores = _safety_broadcast(safety_traj, turn_indices)
            cs_stats = cs_client.stats()
            sample.metadata["safety"] = {
                "enabled": True,
                "coef": safety_coef,
                "summary_weight": safety_summary_weight,
                "zero_threshold": safety_zero_threshold,
                "trajectory_score": safety_traj,
                "per_call_scores": cs_per_call,
                "summary_composite_score": (
                    cs_summary.composite_score if cs_summary is not None else None
                ),
                "summary_dimensions": (
                    cs_summary.dimensions if cs_summary is not None else None
                ),
                "n_calls": cs_stats["calls"],
                "n_errors": cs_stats["errors"],
                "decisions": cs_stats["decisions"],
            }
            logger.info(
                "%s ClawSentry trajectory_score=%.4f calls=%d errors=%d",
                _log_tag,
                safety_traj,
                cs_stats["calls"],
                cs_stats["errors"],
            )

        # Build training samples
        dapo_overlong_cfg = _dapo_overlong_cfg(args)
        samples = _build_samples(
            interactions=interactions,
            base_sample=sample,
            outcome=reward,
            status=status,
            prm_turn_scores=(prm_turn_scores if prm_agent is not None else None),
            prm_coef=prm_coef,
            safety_turn_scores=safety_turn_scores,
            safety_coef=safety_coef,
            discount=1.0,
            encourage=False,
            outcome_is_score=(data_source in _DIRECT_SCORE_DATA_SOURCES),
            penalize_short_response=(data_source not in _DIRECT_SCORE_DATA_SOURCES),
            dapo_overlong_cfg=dapo_overlong_cfg,
        )
        if dapo_overlong_cfg is not None:
            logger.info(
                "%s DAPO overlong cfg: max_resp_len=%s buffer_len=%s expected_len=%s penalty_factor=%s",
                _log_tag,
                dapo_overlong_cfg["max_resp_len"],
                dapo_overlong_cfg["buffer_len"],
                dapo_overlong_cfg["expected_len"],
                dapo_overlong_cfg["penalty_factor"],
            )

        # ── Exploration: add intrinsic + safety + LP-RND + CDE actor bonuses (no-op when disabled) ────
        if (
            _EXPLORE_INTRINSIC_ENABLED
            or _EXPLORE_SAFETY_FILTER_ENABLED
            or _EXPLORE_LPRND_ENABLED
            or _EXPLORE_CDE_ACTOR_ENABLED
            or _AGENT57_CONFIG.active
        ):
            _intr_bonus = _explore_intrinsic_bonus(turn_records)
            _intr_schedule_multiplier = _explore_schedule_multiplier(
                _EXPLORE_INTRINSIC_SCHEDULE,
                run_ctx.train_step,
                _EXPLORE_INTRINSIC_DECAY_STEPS,
            )
            _intr_effective_coef = _EXPLORE_INTRINSIC_COEF * _intr_schedule_multiplier
            _intr_scaled = _intr_bonus * _intr_effective_coef
            _safe_penalty = _explore_safety_penalty(turn_records)
            _lprnd_raw = _explore_lprnd_bonus(interactions)
            _lprnd_schedule_multiplier = _explore_schedule_multiplier(
                _EXPLORE_LPRND_SCHEDULE,
                run_ctx.train_step,
                _EXPLORE_LPRND_DECAY_STEPS,
            )
            _lprnd_effective_coef = _EXPLORE_LPRND_COEF * _lprnd_schedule_multiplier
            _lprnd_bonus = _lprnd_raw * _lprnd_effective_coef
            try:
                _agent57_arm_id = int(
                    (sample.metadata or {}).get(
                        "agent57_arm_id",
                        int(sample.index or 0) % max(1, _AGENT57_CONFIG.k),
                    )
                )
            except (TypeError, ValueError):
                _agent57_arm_id = 0
            _agent57_lifelong_metadata = dict(sample.metadata or {})
            _agent57_lifelong_metadata.setdefault("data_source", data_source)
            _agent57_metrics = _agent57_compute_lifelong_bonus(
                config=_AGENT57_CONFIG,
                arm_id=_agent57_arm_id,
                actions=_iter_explore_actions(turn_records),
                turn_records=turn_records,
                status=status,
                parse_error_count=agent_runner.parse_error_count,
                metadata=_agent57_lifelong_metadata,
            )
            _agent57_bonus = float(
                _agent57_metrics.get("explore_agent57_lifelong_bonus", 0.0) or 0.0
            )
            if (
                _AGENT57_CONFIG.active
                and _AGENT57_CONFIG.combine_mode == "ngu_lite"
            ):
                _agent57_episodic = (
                    _intr_bonus
                    if _AGENT57_CONFIG.ngu_episodic_source == "intrinsic"
                    else _explore_episode_signature_novelty(
                        turn_records,
                        reducer=_AGENT57_CONFIG.ngu_episodic_reducer,
                    )
                )
                _agent57_ngu_metrics = _agent57_compute_ngu_lite_bonus(
                    config=_AGENT57_CONFIG,
                    arm_id=_agent57_arm_id,
                    episodic_novelty=_agent57_episodic,
                    lifelong_raw=float(
                        _agent57_metrics.get(
                            "explore_agent57_lifelong_raw", 0.0
                        )
                        or 0.0
                    ),
                    lifelong_eligible=bool(
                        _agent57_metrics.get(
                            "explore_agent57_lifelong_eligible", 0.0
                        )
                    ),
                    trust_gate=float(
                        _agent57_metrics.get("explore_agent57_trust", 1.0) or 0.0
                    ),
                    life_mod_override=_agent57_metrics.get(
                        "explore_agent57_ngu_life_mod"
                    ),
                )
                _agent57_metrics.update(_agent57_ngu_metrics)
                _agent57_metrics.update(_AGENT57_LAST_EPISODIC_STATS)
                _agent57_bonus = float(
                    _agent57_ngu_metrics.get("explore_agent57_ngu_bonus", 0.0)
                    or 0.0
                )
            _base_score_values = []
            for _sample in samples:
                if isinstance(_sample.reward, dict) and "score" in _sample.reward:
                    try:
                        _base_score_values.append(float(_sample.reward["score"]))
                    except (TypeError, ValueError):
                        pass
            _base_score_mean = (
                sum(_base_score_values) / len(_base_score_values)
                if _base_score_values
                else 0.0
            )
            _agent57_dataset_name = str(data_source or "").strip().lower()
            _agent57_normalized_score_values = []
            if _agent57_dataset_name == "seta":
                for _sample in samples:
                    if not isinstance(_sample.reward, dict):
                        continue
                    _raw_score = _sample.reward.get(
                        "raw_score",
                        _sample.reward.get("accuracy"),
                    )
                    try:
                        _agent57_normalized_score_values.append(float(_raw_score))
                    except (TypeError, ValueError):
                        pass
            _agent57_normalized_score_mean = (
                sum(_agent57_normalized_score_values)
                / len(_agent57_normalized_score_values)
                if _agent57_normalized_score_values
                else None
            )
            _cde_actor = _explore_cde_actor_metrics(
                interactions,
                _base_score_mean,
                run_ctx.train_step,
            )
            _cde_actor_bonus = _cde_actor["bonus"]
            _intr_for_total = (
                0.0
                if (
                    _AGENT57_CONFIG.active
                    and _AGENT57_CONFIG.combine_mode == "ngu_lite"
                )
                else _intr_scaled
            )
            _explore_total = (
                _intr_for_total
                + _safe_penalty
                + _lprnd_bonus
                + _agent57_bonus
                + _cde_actor_bonus
            )
            _explore_score_bonus = _explore_score_bonus_from_components(
                _EXPLORE_SCORE_BONUS_COMPONENTS,
                intrinsic=_intr_for_total,
                safety=_safe_penalty,
                lprnd=_lprnd_bonus,
                agent57=_agent57_bonus,
                cde_actor=_cde_actor_bonus,
            )
            _explore_debug = _explore_debug_metrics(
                status=status,
                base_score_mean=_base_score_mean,
                total_bonus=_explore_total,
                intrinsic_scaled=_intr_for_total,
                safety_penalty=_safe_penalty,
                lprnd_bonus=_lprnd_bonus,
                agent57_bonus=_agent57_bonus,
                cde_actor=_cde_actor,
                turn_records=turn_records,
                parse_error_count=agent_runner.parse_error_count,
            )
            _agent57_record_arm_event(
                config=_AGENT57_CONFIG,
                arm_id=_agent57_arm_id,
                base_score=_base_score_mean,
                final_score=_base_score_mean + _explore_score_bonus,
                status=status,
                parse_error_count=agent_runner.parse_error_count,
                bonus=_agent57_bonus,
                dataset=data_source,
                normalized_base_score=_agent57_normalized_score_mean,
                success_score=_agent57_normalized_score_mean,
            )
            for s in samples:
                if isinstance(s.reward, dict) and "score" in s.reward:
                    s.reward["score"] += _explore_score_bonus
                    s.reward["explore_intrinsic"] = _intr_bonus
                    s.reward["explore_intrinsic_scaled"] = _intr_scaled
                    s.reward["explore_intrinsic_in_total"] = _intr_for_total
                    s.reward["explore_intrinsic_coef"] = _EXPLORE_INTRINSIC_COEF
                    s.reward["explore_intrinsic_effective_coef"] = _intr_effective_coef
                    s.reward["explore_intrinsic_schedule"] = _EXPLORE_INTRINSIC_SCHEDULE
                    s.reward["explore_intrinsic_decay_steps"] = _EXPLORE_INTRINSIC_DECAY_STEPS
                    s.reward["explore_intrinsic_reducer"] = _EXPLORE_INTRINSIC_REDUCER
                    s.reward["explore_intrinsic_schedule_multiplier"] = _intr_schedule_multiplier
                    s.reward["explore_intrinsic_granularity"] = _EXPLORE_INTRINSIC_GRANULARITY
                    s.reward["explore_intrinsic_scope"] = _EXPLORE_INTRINSIC_SCOPE
                    s.reward["explore_safety_penalty"] = _safe_penalty
                    s.reward["explore_lprnd"] = _lprnd_bonus
                    s.reward["explore_lprnd_raw"] = _lprnd_raw
                    s.reward["explore_lprnd_coef"] = _EXPLORE_LPRND_COEF
                    s.reward["explore_lprnd_effective_coef"] = _lprnd_effective_coef
                    s.reward["explore_lprnd_schedule"] = _EXPLORE_LPRND_SCHEDULE
                    s.reward["explore_lprnd_decay_steps"] = _EXPLORE_LPRND_DECAY_STEPS
                    s.reward["explore_lprnd_schedule_multiplier"] = _lprnd_schedule_multiplier
                    if _AGENT57_CONFIG.active:
                        if not isinstance(s.metadata, dict):
                            s.metadata = {}
                        s.reward.update(_agent57_metrics)
                        try:
                            _turn_idx = int(s.metadata.get("turn_idx", -1))
                        except (TypeError, ValueError):
                            _turn_idx = -1
                        _turn_episodic = float(
                            _AGENT57_LAST_EPISODIC_BY_TURN.get(_turn_idx, 0.0)
                        )
                        _turn_life_mod = float(
                            _agent57_metrics.get("explore_agent57_ngu_life_mod", 1.0)
                            or 1.0
                        )
                        s.reward["explore_agent57_turn_episodic"] = _turn_episodic
                        s.reward["explore_agent57_turn_intrinsic_signal"] = (
                            _turn_episodic * _turn_life_mod
                        )
                        s.metadata["agent57"] = {
                            "enabled": bool(_AGENT57_CONFIG.enabled),
                            "arm_id": _agent57_arm_id,
                            "k": _AGENT57_CONFIG.k,
                            "beta": _AGENT57_CONFIG.beta_for_arm(_agent57_arm_id),
                            "controller": _AGENT57_CONFIG.controller,
                            "combine_mode": _AGENT57_CONFIG.combine_mode,
                            "lifelong_enabled": bool(_AGENT57_CONFIG.lifelong_enabled),
                            "lifelong_backend": _AGENT57_CONFIG.lifelong_backend,
                            "lifelong_key_version": _AGENT57_CONFIG.lifelong_key_version,
                            "bonus": _agent57_bonus,
                            "lifelong_bonus": _agent57_metrics.get(
                                "explore_agent57_lifelong_bonus", 0.0
                            ),
                            "ngu_bonus": _agent57_metrics.get(
                                "explore_agent57_ngu_bonus", 0.0
                            ),
                            "lifelong_raw": _agent57_metrics.get(
                                "explore_agent57_lifelong_raw", 0.0
                            ),
                            "lifelong_eligible": _agent57_metrics.get(
                                "explore_agent57_lifelong_eligible", 0.0
                            ),
                        }
                    if _EXPLORE_CDE_ACTOR_ENABLED:
                        s.reward["explore_cde_actor_bonus"] = _cde_actor_bonus
                        s.reward["explore_cde_actor_log_ppl"] = _cde_actor["log_ppl"]
                        s.reward["explore_cde_actor_omega"] = _cde_actor["omega"]
                        s.reward["explore_cde_actor_alpha"] = _cde_actor["alpha"]
                        s.reward["explore_cde_actor_kappa"] = _cde_actor["kappa"]
                        s.reward["explore_cde_actor_reward_gate"] = _EXPLORE_CDE_ACTOR_REWARD_GATE
                        s.reward["explore_cde_actor_eligible"] = _cde_actor["eligible"]
                        s.reward["explore_cde_actor_decay_steps"] = _cde_actor["decay_steps"]
                        s.reward["explore_cde_actor_base_mean"] = _cde_actor["base_score_mean"]
                        s.reward["explore_cde_actor_base_magnitude"] = _cde_actor["base_score_magnitude"]
                        s.reward["explore_cde_actor_cap"] = _cde_actor["cap"]
                        s.reward["explore_cde_actor_scaled"] = _cde_actor["scaled"]
                        s.reward["explore_cde_actor_clipped"] = _cde_actor["clipped"]
                    s.reward["explore_total_bonus"] = _explore_score_bonus
                    s.reward["explore_score_bonus"] = _explore_score_bonus
                    s.reward["explore_all_bonus"] = _explore_total
                    s.reward["explore_score_bonus_components"] = _EXPLORE_SCORE_BONUS_COMPONENTS
                    s.reward.update(_explore_debug)
                    _sync_reward_aliases(s.reward)

        turn_uncertainty_by_idx = {
            int(r["turn_idx"]): r
            for r in turn_uncertainty_records
            if isinstance(r, dict) and r.get("turn_idx") is not None
        }
        for s in samples:
            s.metadata["train_step"] = run_ctx.train_step
            s.metadata["rollout_step"] = run_ctx.rollout_step
            s.metadata["rollout_id"] = run_ctx.rollout_id
            s.metadata["uid"] = run_ctx.uid
            s.metadata["model_turn_count"] = agent_runner.model_turn_count
            s.metadata["parse_error_count"] = agent_runner.parse_error_count
            s.metadata["data_source"] = data_source or s.metadata.get("data_source")
            s.metadata["safety_split"] = _safety_split_from_meta(task_meta)
            claude_backend = str(os.getenv("CLAUDE_CODE_LLM_BACKEND", "sglang")).strip().lower()
            claude_sglang_backend = claude_backend.replace("_", "-") in {
                "sglang",
                "qwen",
                "qwen-sglang",
                "local",
                "local-sglang",
            }
            if agent_type == "claude-code" and _env_flag(
                "CLAUDE_CODE_MARK_NON_TRAINABLE",
                not claude_sglang_backend,
            ):
                s.remove_sample = True
                s.metadata["non_trainable"] = True
                s.metadata["non_trainable_reason"] = (
                    "claude-code CLI uses an external model path without "
                    "terminal-rl policy logprobs"
                )
            if trajectory_uncertainty:
                s.metadata["trajectory_uncertainty"] = trajectory_uncertainty
            turn_uncertainty = turn_uncertainty_by_idx.get(
                int(s.metadata.get("turn_idx", -1))
            )
            if turn_uncertainty:
                s.metadata["turn_uncertainty"] = turn_uncertainty
            if eval_details is not None:
                s.metadata["reward_details"] = _jsonable(eval_details)
            if eval_error is not None:
                s.metadata["evaluation_failed"] = True
                s.metadata["evaluation_error"] = eval_error
        _mark_non_trainable_samples(samples)

        _save_rollout_artifacts(
            task_spec=task_spec,
            run_ctx=run_ctx,
            sampling_params=sampling_params,
            sample=sample,
            samples=samples,
            status=status,
            raw_score=reward,
            eval_error=eval_error,
            turn_records=turn_records,
            safety_meta=sample.metadata.get("safety") if sample.metadata else None,
            prm_meta=sample.metadata.get("prm") if sample.metadata else None,
            safety_coef=safety_coef,
            prm_coef=prm_coef,
            trajectory_save_interval=traj_save_interval,
        )

        if remote_env_admission_key is not None:
            _task_circuit_record_success(task_key)
        return samples

    except Exception as exc:
        if _uses_remote_terminal_env(task_meta):
            _task_circuit_record_failure(task_key, exc)
        log_traceback = _env_bool("TERMINAL_RL_GENERATE_FAILURE_TRACEBACK", False)
        logger.error(
            "%s Generate failed (%s): %s%s",
            _log_tag,
            type(exc).__name__,
            exc,
            "" if log_traceback else " (set TERMINAL_RL_GENERATE_FAILURE_TRACEBACK=1 for traceback)",
            exc_info=log_traceback,
        )
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        sample.metadata = dict(metadata)
        sample.metadata["generate_failed"] = True
        sample.metadata["generate_error_type"] = type(exc).__name__
        sample.metadata["generate_error"] = str(exc)
        sample.status = Sample.Status.FAILED
        sample.remove_sample = True
        sample.reward = {"score": 0.0}
        _sync_reward_aliases(sample.reward)

        eos = state.tokenizer.eos_token_id
        if eos is None:
            sample.tokens = []
            sample.response_length = 0
            sample.rollout_log_probs = []
            sample.loss_mask = []
        else:
            sample.tokens = [eos, eos]
            sample.response_length = 1
            sample.rollout_log_probs = [0.0]
            sample.loss_mask = [0]
        return [sample]

    finally:
        for _turn_idx, t in prm_pending:
            if not t.done():
                t.cancel()

        if cs_client is not None:
            try:
                await cs_client.aclose()
            except Exception as exc:
                logger.debug("ClawSentry aclose ignored: %s", exc)

        if agent_runner is not None:
            try:
                await agent_runner.close()
            except Exception as exc:
                logger.debug("%s Agent runner close ignored: %s", _log_tag, exc)

        try:
            if env_client is not None and lease_id is not None:
                try:
                    close_timeout = _env_float(
                        "ENV_CLOSE_HTTP_TIMEOUT",
                        float(timeouts.close_session) + 30.0,
                    )
                    close_sem = (
                        _remote_env_close_semaphore()
                        if remote_env_admission_key is not None
                        else None
                    )
                    if close_sem is None:
                        await _await_with_optional_timeout(
                            env_client.close(lease_id),
                            close_timeout,
                            op_name=f"{_log_tag} env close",
                        )
                    else:
                        async with close_sem:
                            await _await_with_optional_timeout(
                                env_client.close(lease_id),
                                close_timeout,
                                op_name=f"{_log_tag} env close",
                            )
                except Exception as exc:
                    logger.debug(
                        "%s Best-effort remote close failed lease=%s: %s",
                        _log_tag,
                        lease_id,
                        exc,
                    )
        finally:
            if remote_env_admission_key is not None:
                await _release_remote_env_admission(remote_env_admission_key)
