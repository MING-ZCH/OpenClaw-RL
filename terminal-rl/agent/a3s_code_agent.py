from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import shlex
import socket
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

from custom_types import Interaction, TurnResult
from inference_client import SGLangTurnClient

logger = logging.getLogger(__name__)

DEFAULT_A3S_CODE_TOOL_TIMEOUT_MS = 300_000

_INTERACTIVE_SHELL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(^|[;&|()\n]\s*)tmux\s+(?:a|attach|attach-session)\b"),
        "tmux attach is interactive and can block the rollout",
    ),
    (
        re.compile(r"(^|[;&|()\n]\s*)tmux\s+new(?:-session)?\b(?![^\n;&|]*\s-d(?:\s|$))"),
        "tmux new-session without -d is interactive and can block the rollout",
    ),
    (
        re.compile(r"(^|[;&|()\n]\s*)screen\s+-(?:r|R|x)\b"),
        "screen attach is interactive and can block the rollout",
    ),
    (
        re.compile(r"(^|[;&|()\n]\s*)(?:bash|sh|zsh|fish)\s+-i\b"),
        "interactive shells are not allowed in rollout tool execution",
    ),
    (
        re.compile(r"(^|[;&|()\n]\s*)(?:vim|vi|nano|emacs|less|more)\b"),
        "interactive editors/pagers are not allowed in rollout tool execution",
    ),
)


@dataclass
class A3SCodeResponse:
    msg: str
    terminated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    tool_calls: List[dict[str, Any]] = field(default_factory=list)
    tool_calls_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw_result: Any = None

    @property
    def text(self) -> str:
        return self.msg


A3SCodeFinalResponse = A3SCodeResponse
A3SCodeModelResponse = A3SCodeResponse


def _env_flag(name: str, default: bool) -> bool:
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
        logger.warning("Invalid %s=%r; using %.2f", name, raw, default)
        return default


def _resolve_tool_timeout_ms(raw: Any, turn_timeout_sec: float) -> int:
    if raw is None or raw == "":
        value = DEFAULT_A3S_CODE_TOOL_TIMEOUT_MS
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid A3S_CODE_TOOL_TIMEOUT_MS=%r; using %d",
                raw,
                DEFAULT_A3S_CODE_TOOL_TIMEOUT_MS,
            )
            value = DEFAULT_A3S_CODE_TOOL_TIMEOUT_MS

    if value <= 0:
        logger.warning(
            "Non-positive A3S_CODE_TOOL_TIMEOUT_MS=%r; using %d",
            raw,
            DEFAULT_A3S_CODE_TOOL_TIMEOUT_MS,
        )
        value = DEFAULT_A3S_CODE_TOOL_TIMEOUT_MS

    if turn_timeout_sec > 0:
        turn_timeout_ms = max(1_000, int(turn_timeout_sec * 1000))
        if value >= turn_timeout_ms:
            capped = max(1_000, turn_timeout_ms - 1_000)
            logger.warning(
                "A3S_CODE_TOOL_TIMEOUT_MS=%d is not below turn timeout %.0fs; using %d",
                value,
                turn_timeout_sec,
                capped,
            )
            value = capped

    return value


def _terminal_rl_prompt_extra(max_tool_rounds: int) -> str:
    return f"""## Terminal-RL Harness Instructions

- You are running under the OpenClaw terminal-rl harness. Terminal commands and
  task file operations are executed in the task Docker container; treat that
  Docker environment as the source of truth for task state.
- A3S Code tools are bridged to terminal-rl tools. `bash` and `execute` map to
  `shell_exec`; `read`, `ls`, `grep`, and `glob` are executed through shell
  commands; `write` maps to `shell_write_content_to_file`; `edit` is emulated
  as a single string replacement.
- Use absolute paths from the task instruction whenever possible. Start with
  compact inspection commands such as `pwd`, `ls -la`, and targeted `grep` only
  when they help solve the task.
- Keep command output bounded. Redirect verbose logs to files, use `head`,
  `tail`, or targeted filters, and avoid repeatedly printing large files.
- Long-running commands should be managed with non-blocking terminal execution
  if the exposed schema supports it; otherwise use explicit timeouts and write
  progress or logs to files that can be inspected later.
- Do not attach to interactive terminal sessions or editors (`tmux attach`,
  `screen -r`, interactive shells, vim/nano/less/more). Use non-interactive
  commands and file outputs instead.
- The session is limited to about {max_tool_rounds} A3S agent turns. Batch
  related inspection, implementation, and verification work so there is enough
  budget left to produce a final answer.
- Always verify the result with the narrowest meaningful command available. Once
  the task is complete, stop calling tools and return a concise final answer.
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bootstrap_a3s_code() -> tuple[Any, Any, Any, Any]:
    def _import() -> tuple[Any, Any, Any, Any]:
        from a3s_code import Agent, PermissionPolicy, SessionOptions, SessionQueueConfig

        return Agent, PermissionPolicy, SessionOptions, SessionQueueConfig

    try:
        return _import()
    except ImportError:
        pass

    repo_root = Path(
        os.getenv(
            "A3S_CODE_REPO_ROOT",
            str(_repo_root().parent / "Code"),
        )
    )
    sdk_python = repo_root / "sdk" / "python"
    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(sys.prefix) / "lib" / version_dir / "site-packages",
        Path(os.getenv("CONDA_PREFIX", "")) / "lib" / version_dir / "site-packages",
        sdk_python / ".venv" / "lib" / "python3.13" / "site-packages",
        sdk_python / ".venv" / "lib" / "python3.12" / "site-packages",
    ]
    candidates.extend(
        Path(item).expanduser()
        for item in os.getenv("A3S_CODE_EXTRA_SITE_PACKAGES", "").split(":")
        if item.strip()
    )
    for site in candidates:
        if (site / "a3s_code").exists():
            sys.path.insert(0, str(site))
            return _import()

    raise RuntimeError(
        "a3s_code is not importable. Set A3S_CODE_REPO_ROOT or "
        "A3S_CODE_EXTRA_SITE_PACKAGES, or install the a3s-code SDK before "
        "running HARNESS_OPTION=a3s-code."
    )


def _text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _last_user_text(messages: List[dict[str, Any]] | None) -> str:
    for message in reversed(messages or []):
        if str(message.get("role", "")).lower() == "user":
            text = _text_from_message_content(message.get("content"))
            if text:
                return text
    return ""


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    if tokenizer is None or not text:
        return []
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, dict):
            return list(encoded.get("input_ids") or [])
    except Exception:
        pass
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return []


def _append_no_proxy(hosts: list[str]) -> None:
    existing = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
    parts = [item.strip() for item in existing.split(",") if item.strip()]
    for host in hosts:
        if host not in parts:
            parts.append(host)
    value = ",".join(parts)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _clear_proxy_env_for_local_bridge() -> None:
    _append_no_proxy(["127.0.0.1", "localhost", "::1"])
    if not _env_flag("A3S_CODE_CLEAR_PROXY_FOR_BRIDGE", True):
        return
    for key in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        os.environ.pop(key, None)


class A3SOpenAIModelBridge:
    """OpenAI-compatible local bridge from a3s-code SDK back to SGLangTurnClient."""

    def __init__(self, *, sglang_client: SGLangTurnClient, model_name: str) -> None:
        self._sglang_client = sglang_client
        self._model_name = model_name
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._interactions: list[Interaction] = []
        self._turn_idx_base = 0

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("A3S OpenAI bridge is not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def set_turn_idx_base(self, turn_idx: int) -> None:
        self._turn_idx_base = max(0, int(turn_idx))

    def start(self) -> None:
        if self._server is not None:
            return

        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("a3s-code bridge: " + fmt, *args)

            def do_GET(self) -> None:
                if self.path == "/v1/models":
                    self._write_json(
                        {"object": "list", "data": [{"id": parent._model_name}]}
                    )
                    return
                self.send_error(404)

            def do_POST(self) -> None:
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    if payload.get("stream"):
                        self._write_sse(parent._stream_chunks(payload))
                        return
                    self._write_json(parent._complete(payload))
                except Exception as exc:
                    logger.exception("a3s-code bridge request failed")
                    self._write_json({"error": str(exc)}, status=500)

            def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_sse(self, chunks: list[dict[str, Any]]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for chunk in chunks:
                    body = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode(
                        "utf-8"
                    )
                    self.wfile.write(body)
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        sock.close()
        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"a3s-openai-bridge-{port}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def interactions(self) -> list[Interaction]:
        with self._lock:
            return list(self._interactions)

    def _complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = self._normalize_messages(payload.get("messages") or [])
        tools = payload.get("tools") or None
        with self._lock:
            turn_idx = self._turn_idx_base + len(self._interactions)

        chat_completion, interaction = asyncio.run(
            self._sglang_client.generate_turn(
                messages=messages,
                tools=tools,
                turn_idx=turn_idx,
            )
        )
        with self._lock:
            self._interactions.append(interaction)

        response = chat_completion.model_dump(mode="json", exclude_none=True)
        response["model"] = str(payload.get("model") or self._model_name)
        return response

    def _stream_chunks(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        full_payload = dict(payload)
        full_payload["stream"] = False
        response = self._complete(full_payload)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "stop"
        base = {
            "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": "chat.completion.chunk",
            "created": response.get("created", int(time.time())),
            "model": response.get("model", self._model_name),
        }

        chunks = [
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        ]
        delta: dict[str, Any] = {}
        content = message.get("content")
        if content:
            delta["content"] = str(content)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            delta["tool_calls"] = [
                self._stream_tool_call_delta(index, tool_call)
                for index, tool_call in enumerate(tool_calls)
            ]
            finish_reason = "tool_calls"
        if delta:
            chunks.append(
                {
                    **base,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": None}
                    ],
                }
            )
        chunks.append(
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": finish_reason}
                ],
            }
        )
        return chunks

    @staticmethod
    def _stream_tool_call_delta(index: int, tool_call: dict[str, Any]) -> dict[str, Any]:
        function = dict(tool_call.get("function") or {})
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return {
            "index": index,
            "id": str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
            "type": str(tool_call.get("type") or "function"),
            "function": {
                "name": str(function.get("name") or ""),
                "arguments": arguments,
            },
        }

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            item = dict(message)
            content = item.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if text is None:
                            text = block.get("content")
                        if text is not None:
                            parts.append(str(text))
                    else:
                        parts.append(str(block))
                item["content"] = "\n".join(parts)
            elif content is None:
                item["content"] = ""
            normalized.append(item)
        return normalized


class A3SCodeAgent:
    """a3s-code SDK harness for terminal-rl rollouts.

    The SDK owns the agent loop. Model calls are routed back to terminal-rl's
    SGLangTurnClient through a local OpenAI-compatible bridge, and SDK external
    tool tasks are forwarded to the terminal env lease.
    """

    def __init__(
        self,
        *,
        model_type: str,
        sglang_client: SGLangTurnClient,
        max_total_tokens: int,
        env_client: Any | None = None,
        lease_id: str | None = None,
        run_context: Any | None = None,
        task_meta: Dict[str, Any] | None = None,
        non_think_mode: bool | None = None,
        max_parse_errors: int | None = None,
        tool_timeout_ms: int | None = None,
    ) -> None:
        _ = non_think_mode
        self.model_type = model_type or "slime-sglang"
        self._sglang_client = sglang_client
        self._max_total_tokens = int(max_total_tokens)
        self._env_client = env_client
        self._lease_id = lease_id
        self._run_context = run_context
        self._task_meta = task_meta or {}
        self._max_tool_rounds = max(1, _env_int("A3S_CODE_MAX_TOOL_ROUNDS", 10))
        self._turn_timeout_sec = _env_float("A3S_CODE_TURN_TIMEOUT_SEC", 900.0)
        self._tool_timeout_ms = _resolve_tool_timeout_ms(
            tool_timeout_ms
            if tool_timeout_ms is not None
            else os.getenv("A3S_CODE_TOOL_TIMEOUT_MS"),
            self._turn_timeout_sec,
        )
        self.max_parse_errors = max(1, int(max_parse_errors or 3))
        self.parse_error_count = 0
        self._prompt = ""
        self._session_id = ""
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._bridge: A3SOpenAIModelBridge | None = None
        self._agent: Any | None = None
        self._session: Any | None = None
        self._workspace = self._resolve_workspace()
        self._last_response: A3SCodeResponse | None = None
        self._tool_call_records: list[dict[str, Any]] = []
        self._external_tool_errors_as_results = _env_flag(
            "A3S_CODE_EXTERNAL_TOOL_ERRORS_AS_RESULTS", True
        )
        self._local_workspace_guard = _env_flag("A3S_CODE_LOCAL_WORKSPACE_GUARD", True)
        self._workspace_baseline: dict[str, tuple[int, int]] = {}

    def set_max_parse_errors(self, max_parse_errors: int) -> None:
        self.max_parse_errors = max(1, int(max_parse_errors))

    def set_max_iterations(self, max_iterations: int) -> None:
        # The a3s-code SDK controls inner tool rounds independently through
        # A3S_CODE_MAX_TOOL_ROUNDS. terminal-rl max_iteration limits outer loops.
        _ = max_iterations

    def start_turn_loop(self, input_message: Any) -> None:
        self.parse_error_count = 0
        self._tool_call_records = []
        self._last_response = None
        self._prompt = _text_from_message_content(input_message)
        uid = getattr(self._run_context, "uid", None) or uuid.uuid4().hex[:8]
        self._session_id = os.getenv("A3S_CODE_SESSION_ID") or (
            f"terminal-rl-a3s-{uid}-{uuid.uuid4().hex[:8]}"
        )
        self._close_session()

    async def get_turn_context(
        self,
    ) -> tuple[list[dict[str, Any]] | None, A3SCodeResponse | None]:
        if self._last_response is not None:
            return None, self._last_response
        return [{"role": "user", "content": self._prompt}], None

    async def consume_completion(
        self, chat_completion: Any
    ) -> tuple[Any | None, list[Any], bool, A3SCodeResponse | None]:
        _ = chat_completion
        raise RuntimeError("A3SCodeAgent uses the a3s-code SDK run path")

    def record_tool_result(self, tool_call_request: Any, raw_result: Any) -> None:
        _ = (tool_call_request, raw_result)

    async def run_model_turn(
        self,
        context_messages: list[dict[str, Any]] | None = None,
        *,
        sglang_client: SGLangTurnClient | None = None,
        tool_schemas: List[Dict[str, Any]] | None = None,
        turn_idx: int = 0,
    ) -> TurnResult:
        _ = (sglang_client, tool_schemas)
        prompt = _last_user_text(context_messages) or self._prompt
        if prompt:
            self._prompt = prompt

        try:
            result = await self._run_send_thread(turn_idx)
        except asyncio.TimeoutError as exc:
            self._try_cancel_session()
            self._close_session()
            raise TimeoutError(
                f"a3s-code session.send timed out after {self._turn_timeout_sec:.0f}s"
            ) from exc
        except Exception as exc:
            try:
                self._check_local_workspace_mutation()
            except Exception as guard_exc:
                self._close_session()
                raise guard_exc from exc
            raise

        interactions = self._bridge.interactions() if self._bridge is not None else []
        self._last_response = self._response_from_result(result)
        try:
            self._check_local_workspace_mutation()
        except Exception:
            self._close_session()
            raise
        if not interactions:
            interactions = [self._fallback_interaction(turn_idx, self._last_response.msg)]

        return TurnResult(
            interaction=interactions[-1],
            model_response=self._last_response,
            tool_call_requests=[],
            parse_error_recorded=False,
            terminated_response=None,
            interactions=interactions,
        )

    def finalize_response(self, model_response: Any) -> A3SCodeResponse:
        if isinstance(model_response, A3SCodeResponse):
            return model_response
        return self._last_response or A3SCodeResponse(
            msg="",
            terminated=True,
            info={
                "termination_reasons": ["missing_a3s_code_response"],
                "harness_option": "a3s-code",
            },
        )

    async def close(self) -> None:
        self._close_session()

    def _resolve_workspace(self) -> Path:
        raw = os.getenv("A3S_CODE_WORKSPACE")
        if raw:
            path = Path(raw).expanduser()
        else:
            uid = getattr(self._run_context, "uid", None) or uuid.uuid4().hex[:8]
            task_name = str(self._task_meta.get("task_name") or "task")
            safe_task = "".join(c if c.isalnum() or c in "._-" else "-" for c in task_name)
            root = Path(
                os.getenv(
                    "A3S_CODE_WORKSPACE_ROOT",
                    str(_repo_root() / "runs" / "a3s_code_workspaces"),
                )
            )
            path = root / f"a3s-code-{safe_task[:48]}-{uid}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _snapshot_workspace(self) -> dict[str, tuple[int, int]]:
        snapshot: dict[str, tuple[int, int]] = {}
        if not self._local_workspace_guard:
            return snapshot
        try:
            for item in self._workspace.rglob("*"):
                if not item.is_file():
                    continue
                rel = str(item.relative_to(self._workspace))
                stat = item.stat()
                snapshot[rel] = (int(stat.st_size), int(stat.st_mtime_ns))
        except OSError as exc:
            logger.warning("Failed to snapshot a3s-code workspace %s: %s", self._workspace, exc)
        return snapshot

    def _check_local_workspace_mutation(self) -> None:
        if not self._local_workspace_guard:
            return
        before = self._workspace_baseline
        after = self._snapshot_workspace()
        changed = [
            path
            for path, stat in after.items()
            if before.get(path) != stat
        ]
        deleted = [path for path in before if path not in after]
        if not changed and not deleted:
            return
        preview = ", ".join((changed + deleted)[:8])
        suffix = "" if len(changed) + len(deleted) <= 8 else ", ..."
        raise RuntimeError(
            "a3s-code local workspace mutation detected; tool execution likely "
            "bypassed the terminal-rl Docker bridge. "
            f"workspace={self._workspace} changed={len(changed)} "
            f"deleted={len(deleted)} files=[{preview}{suffix}]"
        )

    def _ensure_session(self, turn_idx: int = 0) -> None:
        if self._session is not None:
            if self._bridge is not None:
                self._bridge.set_turn_idx_base(turn_idx)
            return

        _clear_proxy_env_for_local_bridge()
        Agent, PermissionPolicy, SessionOptions, SessionQueueConfig = _bootstrap_a3s_code()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="terminal-rl-a3s-")
        tmpdir = Path(self._tmpdir.name)

        self._bridge = A3SOpenAIModelBridge(
            sglang_client=self._sglang_client,
            model_name=self.model_type,
        )
        self._bridge.set_turn_idx_base(turn_idx)
        self._bridge.start()
        config_path = tmpdir / "agent.acl"
        config_path.write_text(
            self._render_agent_config(self._bridge.base_url),
            encoding="utf-8",
        )

        opts = SessionOptions()
        opts.session_id = self._session_id
        opts.builtin_skills = _env_flag("A3S_CODE_BUILTIN_SKILLS", False)
        opts.max_parse_retries = self.max_parse_errors
        opts.max_tool_rounds = self._max_tool_rounds
        opts.tool_timeout_ms = self._tool_timeout_ms
        opts.circuit_breaker_threshold = _env_int("A3S_CODE_CIRCUIT_BREAKER", 3)
        opts.planning_mode = os.getenv("A3S_CODE_PLANNING_MODE", "disabled")
        extra_prompts: list[str] = []
        if _env_flag("A3S_CODE_TERMINAL_RL_EXTRA_PROMPT", True):
            extra_prompts.append(_terminal_rl_prompt_extra(self._max_tool_rounds))
        custom_extra = os.getenv("A3S_CODE_EXTRA_PROMPT", "").strip()
        if custom_extra:
            extra_prompts.append(custom_extra)
        if extra_prompts:
            opts.extra = "\n\n".join(extra_prompts)
        thinking_budget = os.getenv("A3S_CODE_THINKING_BUDGET", "").strip()
        if thinking_budget:
            opts.thinking_budget = int(thinking_budget)
        opts.permission_policy = PermissionPolicy(default_decision="allow")

        queue = SessionQueueConfig()
        queue.set_lane_handler("query", "external", self._tool_timeout_ms)
        queue.set_lane_handler("execute", "external", self._tool_timeout_ms)
        opts.queue_config = queue

        self._agent = Agent.create(str(config_path))
        self._session = self._agent.session(str(self._workspace), opts)
        self._workspace_baseline = self._snapshot_workspace()

    async def _run_send_thread(self, turn_idx: int) -> Any:
        loop = asyncio.get_running_loop()
        result_box: dict[str, Any] = {}

        def target() -> None:
            try:
                result_box["result"] = self._send_with_external_tools(turn_idx, loop)
            except BaseException as exc:
                result_box["error"] = exc

        thread = threading.Thread(
            target=target,
            name=f"a3s-code-send-{self._session_id or uuid.uuid4().hex[:8]}",
            daemon=True,
        )
        thread.start()

        deadline = (
            time.monotonic() + self._turn_timeout_sec
            if self._turn_timeout_sec > 0
            else None
        )
        while thread.is_alive():
            if deadline is not None and time.monotonic() >= deadline:
                raise asyncio.TimeoutError()
            await asyncio.sleep(0.05)

        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("result")

    def _send_with_external_tools(
        self,
        turn_idx: int,
        loop: asyncio.AbstractEventLoop,
    ) -> Any:
        self._ensure_session(turn_idx)
        assert self._session is not None

        result_box: dict[str, Any] = {}

        def send_target() -> None:
            try:
                result_box["result"] = self._session.send(self._prompt)
            except Exception as exc:
                result_box["error"] = exc

        sender = threading.Thread(target=send_target, daemon=True)
        sender.start()

        handled: set[str] = set()
        while sender.is_alive():
            self._drain_external_tasks(handled, loop)
            sender.join(timeout=0.1)

        self._drain_external_tasks(handled, loop)
        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("result")

    def _drain_external_tasks(
        self,
        handled: set[str],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        if self._session is None:
            return
        pending_fn = getattr(self._session, "pending_external_tasks", None)
        if not callable(pending_fn):
            return
        for task in list(pending_fn() or []):
            task_id = str(self._task_get(task, "task_id") or "")
            if not task_id or task_id in handled:
                continue
            handled.add(task_id)
            tool_name = str(
                self._task_get(task, "command_type")
                or self._task_get(task, "tool_name")
                or self._task_get(task, "name")
                or ""
            )
            payload = self._task_get(task, "payload")
            if payload is None:
                payload = self._task_get(task, "arguments")
            if payload is None:
                payload = self._task_get(task, "args")
            args = payload if isinstance(payload, dict) else {}
            mapped_name = str(tool_name or "unknown")
            mapped_args: dict[str, Any] = dict(args)

            try:
                mapped_name, mapped_args = self._map_tool_call(tool_name, args)
                output = self._exec_terminal_tool_on_loop(loop, mapped_name, mapped_args)
                self._complete_external_task(
                    task_id,
                    success=True,
                    payload={"output": output, "exit_code": 0},
                    error=None,
                )
                self._tool_call_records.append(
                    {
                        "tool_call_id": task_id,
                        "tool_name": mapped_name,
                        "sdk_tool_name": tool_name,
                        "args": mapped_args,
                        "sdk_args": args,
                        "result": (
                            output[:4096]
                            if isinstance(output, str)
                            else str(output)[:4096]
                        ),
                        "source": "a3s-code-sdk",
                    }
                )
            except Exception as exc:
                error_output = (
                    "[terminal-rl] Docker-bridged a3s-code tool execution failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                # The a3s-code core falls back to direct local execution when an
                # external queue task is completed as failed. In terminal-rl that
                # is unsafe: a failed Docker/env call must be surfaced to the
                # model as a normal tool result, never retried on the GPU host.
                complete_as_success = self._external_tool_errors_as_results
                error_payload = {
                    "output": error_output,
                    "exit_code": 1,
                    "metadata": {
                        "terminal_rl_external_error": True,
                        "error_type": type(exc).__name__,
                    },
                }
                self._complete_external_task(
                    task_id,
                    success=complete_as_success,
                    payload=error_payload if complete_as_success else None,
                    error=None if complete_as_success else error_output,
                )
                self._tool_call_records.append(
                    {
                        "tool_call_id": task_id,
                        "tool_name": mapped_name,
                        "sdk_tool_name": tool_name,
                        "args": mapped_args,
                        "sdk_args": args,
                        "error": f"{type(exc).__name__}: {exc}",
                        "result": error_output[:4096],
                        "exit_code": 1,
                        "completed_as_result": complete_as_success,
                        "source": "a3s-code-sdk",
                    }
                )

    def _exec_terminal_tool_on_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        tool_name: str,
        args: dict[str, Any],
    ) -> str:
        future = asyncio.run_coroutine_threadsafe(
            self._exec_terminal_tool(tool_name, args),
            loop,
        )
        try:
            return future.result(timeout=max(1.0, self._tool_timeout_ms / 1000.0))
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"a3s-code terminal tool {tool_name!r} timed out after "
                f"{self._tool_timeout_ms}ms"
            )

    @staticmethod
    def _task_get(task: Any, key: str) -> Any:
        if isinstance(task, dict):
            return task.get(key)
        return getattr(task, key, None)

    @staticmethod
    def _guard_shell_command(command: str) -> str:
        command = str(command or "")
        for pattern, reason in _INTERACTIVE_SHELL_PATTERNS:
            if pattern.search(command):
                raise RuntimeError(
                    f"Refusing interactive shell command in a3s-code bridge: {reason}"
                )
        return command

    def _complete_external_task(
        self,
        task_id: str,
        *,
        success: bool,
        payload: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        if self._session is None:
            return
        complete = getattr(self._session, "complete_external_task", None)
        if not callable(complete):
            return
        try:
            if success:
                complete(task_id, True, payload)
            else:
                complete(task_id, False, payload, error)
        except TypeError:
            complete(task_id=task_id, success=success, result=payload, error=error)

    async def _exec_terminal_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        if self._env_client is None or self._lease_id is None:
            raise RuntimeError("terminal env client is required for a3s-code tool execution")
        heartbeat = getattr(self._env_client, "heartbeat", None)
        if callable(heartbeat):
            await heartbeat(self._lease_id)
        return await self._env_client.exec_tool(self._lease_id, tool_name, args)

    @staticmethod
    def _map_tool_call(tool_name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if tool_name in {
            "shell_exec",
            "shell_view",
            "shell_write_to_process",
            "shell_write_content_to_file",
        }:
            if tool_name == "shell_exec":
                mapped_args = dict(args)
                command = str(mapped_args.get("command") or mapped_args.get("cmd") or "")
                mapped_args["command"] = A3SCodeAgent._guard_shell_command(command)
                return tool_name, mapped_args
            return tool_name, args

        if tool_name in {"bash", "execute"}:
            command = str(args.get("command") or args.get("cmd") or "")
            command = A3SCodeAgent._guard_shell_command(command)
            return "shell_exec", {"command": command}

        if tool_name == "read":
            path = str(args.get("path") or args.get("file_path") or "")
            offset = int(args.get("offset") or args.get("line_offset") or 1)
            limit = int(args.get("limit") or args.get("line_limit") or 200)
            command = (
                f"sed -n '{max(1, offset)},{max(1, offset) + max(1, limit) - 1}p' "
                f"{shlex.quote(path)}"
            )
            return "shell_exec", {"command": command}

        if tool_name == "ls":
            path = str(args.get("path") or ".")
            return "shell_exec", {"command": f"ls -la {shlex.quote(path)}"}

        if tool_name == "grep":
            pattern = str(args.get("pattern") or args.get("query") or "")
            path = str(args.get("path") or ".")
            command = (
                f"grep -RIn -- {shlex.quote(pattern)} {shlex.quote(path)} | head -200"
            )
            return "shell_exec", {"command": command}

        if tool_name == "glob":
            pattern = str(args.get("pattern") or args.get("glob") or "*")
            command = (
                "python3 - <<'PY'\n"
                "import glob\n"
                f"for p in glob.glob({json.dumps(pattern)}, recursive=True)[:200]: print(p)\n"
                "PY"
            )
            return "shell_exec", {"command": command}

        if tool_name == "write":
            path = str(args.get("path") or args.get("file_path") or "")
            content = str(args.get("content") or "")
            return "shell_write_content_to_file", {"file_path": path, "content": content}

        if tool_name == "edit":
            path = str(args.get("path") or args.get("file_path") or "")
            old = str(args.get("old_string") or args.get("old") or "")
            new = str(args.get("new_string") or args.get("new") or "")
            command = (
                "python3 - <<'PY'\n"
                "from pathlib import Path\n"
                f"path = Path({json.dumps(path)})\n"
                f"old = {json.dumps(old)}\n"
                f"new = {json.dumps(new)}\n"
                "text = path.read_text()\n"
                "if old not in text:\n"
                "    raise SystemExit('old_string not found')\n"
                "path.write_text(text.replace(old, new, 1))\n"
                "PY"
            )
            return "shell_exec", {"command": command}

        return "shell_exec", {
            "command": f"echo unsupported a3s-code tool: {shlex.quote(tool_name)}"
        }

    def _response_from_result(self, result: Any) -> A3SCodeResponse:
        text = str(getattr(result, "text", "") or "")
        sdk_tool_calls = list(getattr(result, "tool_calls", []) or [])
        tool_calls = self._tool_call_records or sdk_tool_calls
        prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(result, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(result, "total_tokens", prompt_tokens + completion_tokens) or 0
        )
        tool_calls_count = int(
            getattr(result, "tool_calls_count", len(tool_calls)) or len(tool_calls)
        )
        info = {
            "termination_reasons": [],
            "harness_option": "a3s-code",
            "harness": "a3s-code",
            "session_id": self._session_id,
            "workspace": str(self._workspace),
            "task_path": self._task_meta.get("task_path"),
            "tool_calls_count": tool_calls_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "parse_error_count": self.parse_error_count,
            "tool_calls": list(tool_calls),
        }
        return A3SCodeResponse(
            msg=text,
            terminated=False,
            info=info,
            tool_calls=list(tool_calls),
            tool_calls_count=tool_calls_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_result=result,
        )

    def _fallback_interaction(self, turn_idx: int, text: str) -> Interaction:
        tokenizer = getattr(self._sglang_client, "tokenizer", None)
        input_ids = _tokenize(tokenizer, self._prompt)
        output_ids = _tokenize(tokenizer, text)
        return Interaction(
            turn_idx=turn_idx,
            input_ids=input_ids,
            output_token_ids=output_ids,
            output_token_logprobs=[0.0] * len(output_ids),
            output_text=text,
            finish_reason="stop",
            messages=[{"role": "user", "content": self._prompt}],
            latency_ms=0.0,
        )

    def _render_agent_config(self, base_url: str) -> str:
        output_tokens = _env_int("A3S_CODE_OUTPUT_TOKENS", 8192)
        return (
            f'default_model = "openai/{self.model_type}"\n\n'
            'providers "openai" {\n'
            '  api_key = "terminal-rl"\n'
            f'  base_url = "{base_url}"\n\n'
            f'  models "{self.model_type}" {{\n'
            f'    name = "{self.model_type}"\n'
            "    tool_call = true\n\n"
            "    limit = {\n"
            f"      context = {self._max_total_tokens}\n"
            f"      output = {output_tokens}\n"
            "    }\n"
            "  }\n"
            "}\n"
        )

    def _try_cancel_session(self) -> None:
        cancel = getattr(self._session, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass

    def _close_session(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            close_fn = getattr(session, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    logger.debug("a3s-code session close ignored", exc_info=True)
        if self._bridge is not None:
            self._bridge.close()
        self._bridge = None
        self._agent = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
        self._tmpdir = None
