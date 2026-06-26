from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from agent.claude_code_qwen_gateway import ClaudeCodeQwenGateway
from custom_types import Interaction, TurnResult
from inference_client import SGLangTurnClient

logger = logging.getLogger(__name__)


DEFAULT_ALLOWED_TOOLS = (
    "mcp__terminal_rl__shell_exec,"
    "mcp__terminal_rl__shell_view,"
    "mcp__terminal_rl__shell_write_to_process,"
    "mcp__terminal_rl__shell_write_content_to_file"
)


@dataclass
class ClaudeCodeResponse:
    msg: str
    terminated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    tool_calls: List[dict[str, Any]] = field(default_factory=list)
    tool_calls_count: int = 0
    raw_result: Any = None

    @property
    def text(self) -> str:
        return self.msg


ClaudeCodeFinalResponse = ClaudeCodeResponse
ClaudeCodeModelResponse = ClaudeCodeResponse


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


def _normalize_llm_backend(value: str | None) -> str:
    text = str(value or "sglang").strip().lower().replace("_", "-")
    if text in {"sglang", "qwen", "qwen-sglang", "local", "local-sglang"}:
        return "sglang"
    if text in {"anthropic", "claude", "claude-api", "external"}:
        return "anthropic"
    logger.warning("Invalid CLAUDE_CODE_LLM_BACKEND=%r; using sglang", value)
    return "sglang"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _sanitize_filename(value: str) -> str:
    text = "".join(c if c.isalnum() or c in "._-" else "-" for c in str(value))
    text = "-".join(part for part in text.split("-") if part)
    return text[:96].strip("._-") or "task"


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is not None:
                    chunks.append(str(text))
            elif item is not None:
                chunks.append(str(item))
        return "\n".join(chunks)
    if isinstance(value, dict):
        for key in ("text", "content", "result"):
            if key in value:
                return _content_text(value[key])
    return "" if value is None else str(value)


def _extract_result_text(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("result", "text", "response", "summary"):
            value = payload.get(key)
            if value:
                return _content_text(value)
        message = payload.get("message")
        if isinstance(message, dict):
            text = _content_text(message.get("content"))
            if text:
                return text
        content = payload.get("content")
        if content:
            return _content_text(content)
    return ""


def _parse_claude_output(stdout: str, output_format: str) -> tuple[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        return "", None
    if output_format == "text":
        return stripped, None

    parsed_events: list[Any] = []
    try:
        payload = json.loads(stripped)
        text = _extract_result_text(payload)
        return text or stripped, payload
    except Exception:
        pass

    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed_events.append(json.loads(line))
        except Exception:
            continue

    for event in reversed(parsed_events):
        text = _extract_result_text(event)
        if text:
            return text, parsed_events
    return stripped, parsed_events or None


class ClaudeCodeAgent:
    """Claude Code CLI harness for terminal-rl rollouts.

    Claude Code owns the agent loop. Tool calls are restricted to an MCP server
    that forwards terminal tools to the current terminal-rl env lease.
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
    ) -> None:
        _ = (model_type, max_total_tokens, non_think_mode)
        self._sglang_client = sglang_client
        self._env_client = env_client
        self._lease_id = lease_id
        self._run_context = run_context
        self._task_meta = task_meta or {}
        self.max_parse_errors = max(1, int(max_parse_errors or 3))
        self.parse_error_count = 0
        self._prompt = ""
        self._session_id = ""
        self._last_response: ClaudeCodeResponse | None = None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._workspace = self._resolve_workspace()
        self._tool_log_path = self._workspace / "terminal_rl_tool_calls.jsonl"
        self._stdout_path = self._workspace / "claude_stdout.log"
        self._stderr_path = self._workspace / "claude_stderr.log"
        self._mcp_config_path = self._workspace / "claude_mcp_config.json"
        self._qwen_records_path = self._workspace / "qwen_gateway_records.jsonl"
        self._qwen_gateway: ClaudeCodeQwenGateway | None = None

        self._cli = os.getenv("CLAUDE_CODE_CLI", "claude").strip() or "claude"
        self._model = os.getenv("CLAUDE_CODE_MODEL", "").strip()
        self._llm_backend = _normalize_llm_backend(os.getenv("CLAUDE_CODE_LLM_BACKEND"))
        self._qwen_gateway_model = (
            os.getenv("CLAUDE_CODE_QWEN_GATEWAY_MODEL", "qwen-8b-sglang").strip()
            or "qwen-8b-sglang"
        )
        self._output_format = os.getenv("CLAUDE_CODE_OUTPUT_FORMAT", "json").strip() or "json"
        self._turn_timeout_sec = _env_float("CLAUDE_CODE_TURN_TIMEOUT_SEC", 900.0)
        self._tool_timeout_ms = _env_int("CLAUDE_CODE_TOOL_TIMEOUT_MS", 300_000)
        self._max_turns = max(1, _env_int("CLAUDE_CODE_MAX_TOOL_ROUNDS", 10))
        self._mcp_python = os.getenv("CLAUDE_CODE_MCP_PYTHON", sys.executable).strip() or sys.executable
        self._permission_mode = os.getenv(
            "CLAUDE_CODE_PERMISSION_MODE",
            "bypassPermissions",
        ).strip()
        self._allowed_tools = os.getenv(
            "CLAUDE_CODE_ALLOWED_TOOLS",
            DEFAULT_ALLOWED_TOOLS,
        ).strip()
        self._disallowed_tools = os.getenv("CLAUDE_CODE_DISALLOWED_TOOLS", "").strip()
        self._extra_args = os.getenv("CLAUDE_CODE_EXTRA_ARGS", "").strip()
        self._system_prompt = os.getenv("CLAUDE_CODE_SYSTEM_PROMPT", "").strip()

    def set_max_parse_errors(self, max_parse_errors: int) -> None:
        self.max_parse_errors = max(1, int(max_parse_errors))

    def set_max_iterations(self, max_iterations: int) -> None:
        self._max_turns = max(1, int(max_iterations))

    def start_turn_loop(self, input_message: Any) -> None:
        self.parse_error_count = 0
        self._last_response = None
        self._prompt = _text_from_message_content(input_message)
        uid = getattr(self._run_context, "uid", None) or uuid.uuid4().hex[:8]
        self._session_id = os.getenv("CLAUDE_CODE_SESSION_ID") or (
            f"terminal-rl-claude-{uid}-{uuid.uuid4().hex[:8]}"
        )
        self._tool_log_path.unlink(missing_ok=True)
        self._stdout_path.unlink(missing_ok=True)
        self._stderr_path.unlink(missing_ok=True)
        self._qwen_records_path.unlink(missing_ok=True)

    async def get_turn_context(
        self,
    ) -> tuple[list[dict[str, Any]] | None, ClaudeCodeResponse | None]:
        if self._last_response is not None:
            return None, self._last_response
        return [{"role": "user", "content": self._prompt}], None

    async def consume_completion(
        self, chat_completion: Any
    ) -> tuple[Any | None, list[Any], bool, ClaudeCodeResponse | None]:
        _ = chat_completion
        raise RuntimeError("ClaudeCodeAgent uses the Claude Code CLI run path")

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

        started = time.monotonic()
        completed = await self._run_claude_code_async(self._prompt)
        latency_ms = (time.monotonic() - started) * 1000.0
        text, raw_result = _parse_claude_output(completed.stdout, self._output_format)
        tool_calls = self._load_tool_calls()
        qwen_records = self._load_qwen_gateway_records()
        if self._uses_sglang_gateway():
            if not qwen_records:
                raise RuntimeError(
                    "claude-code sglang backend produced no Qwen gateway records; "
                    "cannot build trainable GRPO samples"
                )
            interactions = self._interactions_from_qwen_records(qwen_records, turn_idx)
            interaction = interactions[-1]
        else:
            interaction = self._interaction(turn_idx, self._prompt, text, latency_ms)
            interactions = [interaction]
        self._last_response = self._response_from_completed(
            completed,
            text,
            raw_result,
            tool_calls,
            qwen_records=qwen_records,
        )
        return TurnResult(
            interaction=interaction,
            model_response=self._last_response,
            tool_call_requests=[],
            parse_error_recorded=False,
            terminated_response=None,
            interactions=interactions,
        )

    def finalize_response(self, model_response: Any) -> ClaudeCodeResponse:
        if isinstance(model_response, ClaudeCodeResponse):
            return model_response
        return self._last_response or ClaudeCodeResponse(
            msg="",
            terminated=True,
            info={
                "termination_reasons": ["missing_claude_code_response"],
                "harness_option": "claude-code",
            },
        )

    async def close(self) -> None:
        if self._qwen_gateway is not None:
            self._qwen_gateway.close()
            self._qwen_gateway = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def _resolve_workspace(self) -> Path:
        raw = os.getenv("CLAUDE_CODE_WORKSPACE")
        if raw:
            path = Path(raw).expanduser()
        else:
            uid = getattr(self._run_context, "uid", None) or uuid.uuid4().hex[:8]
            task_name = str(self._task_meta.get("task_name") or "task")
            root = Path(
                os.getenv(
                    "CLAUDE_CODE_WORKSPACE_ROOT",
                    str(_repo_root() / "runs" / "claude_code_workspaces"),
                )
            )
            path = root / f"claude-code-{_sanitize_filename(task_name)}-{uid}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_claude_code(self, prompt: str) -> subprocess.CompletedProcess[str]:
        args, env = self._prepare_claude_command()

        try:
            completed = subprocess.run(
                args,
                input=prompt,
                text=True,
                cwd=str(self._workspace),
                env=env,
                capture_output=True,
                timeout=self._turn_timeout_sec if self._turn_timeout_sec > 0 else None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"claude-code CLI timed out after {self._turn_timeout_sec:.0f}s"
            ) from exc

        self._check_completed(completed)
        return completed

    async def _run_claude_code_async(self, prompt: str) -> subprocess.CompletedProcess[str]:
        result: list[subprocess.CompletedProcess[str]] = []
        errors: list[BaseException] = []

        def target() -> None:
            try:
                result.append(self._run_claude_code(prompt))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(
            target=target,
            name="terminal-rl-claude-code-cli",
            daemon=True,
        )
        thread.start()
        started = time.monotonic()
        guard_timeout = (
            self._turn_timeout_sec + 5.0 if self._turn_timeout_sec > 0 else None
        )
        while thread.is_alive():
            if guard_timeout is not None and time.monotonic() - started > guard_timeout:
                raise TimeoutError(
                    f"claude-code CLI thread did not finish after {guard_timeout:.0f}s"
                )
            await asyncio.sleep(0.05)
        if errors:
            raise errors[0]
        if not result:
            raise RuntimeError("claude-code CLI finished without a result")
        return result[0]

    def _prepare_claude_command(self) -> tuple[list[str], dict[str, str]]:
        if self._env_client is None or self._lease_id is None:
            raise RuntimeError("terminal env client is required for claude-code tool execution")
        cli_path = self._resolve_cli()
        self._write_mcp_config()
        if self._uses_sglang_gateway():
            self._ensure_qwen_gateway()
        return self._build_cli_args(cli_path), self._build_subprocess_env()

    def _check_completed(self, completed: subprocess.CompletedProcess[str]) -> None:
        self._stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        self._stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()[:2000]
            raise RuntimeError(
                f"claude-code CLI exited with {completed.returncode}: {stderr}"
            )

    def _resolve_cli(self) -> str:
        if os.path.sep in self._cli:
            path = Path(self._cli).expanduser()
            if not path.exists():
                raise RuntimeError(f"CLAUDE_CODE_CLI does not exist: {path}")
            return str(path)
        resolved = shutil.which(self._cli)
        if not resolved:
            raise RuntimeError(
                f"Claude Code CLI {self._cli!r} not found. Set CLAUDE_CODE_CLI."
            )
        return resolved

    def _build_cli_args(self, cli_path: str) -> list[str]:
        args = [cli_path, "-p", "--output-format", self._output_format]
        if self._model:
            args.extend(["--model", self._model])
        args.extend(["--max-turns", str(self._max_turns)])
        args.extend(["--mcp-config", str(self._mcp_config_path)])
        if self._allowed_tools:
            args.extend(["--allowedTools", self._allowed_tools])
        if self._disallowed_tools:
            args.extend(["--disallowedTools", self._disallowed_tools])
        if self._permission_mode:
            args.extend(["--permission-mode", self._permission_mode])
        system_prompt = self._system_prompt or self._default_system_prompt()
        if system_prompt:
            args.extend(["--append-system-prompt", system_prompt])
        if self._extra_args:
            args.extend(shlex.split(self._extra_args))
        return args

    def _default_system_prompt(self) -> str:
        return (
            "You are running inside the OpenClaw terminal-rl harness. Use only the "
            "terminal_rl MCP tools for task inspection and modification; they execute "
            "inside the benchmark task environment. Do not rely on local Read, Write, "
            "Edit, or Bash tools for task state. Keep command output bounded and stop "
            "once the task is complete."
        )

    def _build_subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("CLAUDE_CODE_ENTRYPOINT", "terminal-rl")
        env.setdefault("CLAUDE_CODE_SESSION_ID", self._session_id)
        if self._uses_sglang_gateway():
            gateway = self._ensure_qwen_gateway()
            env["ANTHROPIC_BASE_URL"] = gateway.base_url
            env["ANTHROPIC_AUTH_TOKEN"] = "terminal-rl-qwen"
            env["ANTHROPIC_API_KEY"] = "terminal-rl-qwen"
            env.pop("ANTHROPIC_API_URL", None)
            env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
            env.setdefault("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "0")
        no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or ""
        parts = [p.strip() for p in no_proxy.split(",") if p.strip()]
        for host in ("127.0.0.1", "localhost", "::1"):
            if host not in parts:
                parts.append(host)
        env["NO_PROXY"] = ",".join(parts)
        env["no_proxy"] = env["NO_PROXY"]
        return env

    def _uses_sglang_gateway(self) -> bool:
        return self._llm_backend == "sglang"

    def _default_non_trainable(self) -> bool:
        return not self._uses_sglang_gateway()

    def _ensure_qwen_gateway(self) -> ClaudeCodeQwenGateway:
        if self._qwen_gateway is None:
            self._qwen_gateway = ClaudeCodeQwenGateway(
                sglang_client=self._sglang_client,
                records_path=self._qwen_records_path,
                model_name=self._qwen_gateway_model,
            )
            self._qwen_gateway.start()
        return self._qwen_gateway

    def _write_mcp_config(self) -> None:
        base_url = str(getattr(self._env_client, "base_url", "")).rstrip("/")
        if not base_url:
            raise RuntimeError("env_client.base_url is required for claude-code MCP tools")
        server_path = Path(__file__).with_name("claude_code_mcp_server.py")
        env = {
            "CLAUDE_CODE_TERMINAL_ENV_SERVER_URL": base_url,
            "CLAUDE_CODE_TERMINAL_LEASE_ID": str(self._lease_id),
            "CLAUDE_CODE_TOOL_TIMEOUT_SEC": str(max(1.0, self._tool_timeout_ms / 1000.0)),
            "CLAUDE_CODE_TOOL_LOG_PATH": str(self._tool_log_path),
            "CLAUDE_CODE_HTTP_MAX_RETRIES": os.getenv("CLAUDE_CODE_HTTP_MAX_RETRIES", "3"),
            "CLAUDE_CODE_HTTP_RETRY_DELAY": os.getenv("CLAUDE_CODE_HTTP_RETRY_DELAY", "1.0"),
        }
        pythonpath = os.getenv("PYTHONPATH", "")
        if pythonpath:
            env["PYTHONPATH"] = pythonpath
        config = {
            "mcpServers": {
                "terminal_rl": {
                    "command": self._mcp_python,
                    "args": [str(server_path)],
                    "env": env,
                }
            }
        }
        self._mcp_config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_tool_calls(self) -> list[dict[str, Any]]:
        if not self._tool_log_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self._tool_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                records.append(item)
        return records

    def _load_qwen_gateway_records(self) -> list[dict[str, Any]]:
        if self._qwen_gateway is not None:
            records = self._qwen_gateway.records()
            if records:
                return records
        if not self._qwen_records_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self._qwen_records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                records.append(item)
        return records

    def _interactions_from_qwen_records(
        self,
        records: list[dict[str, Any]],
        first_turn_idx: int,
    ) -> list[Interaction]:
        interactions: list[Interaction] = []
        for offset, record in enumerate(records):
            output_token_ids = [int(x) for x in (record.get("output_token_ids") or [])]
            output_token_logprobs = [
                float(x) for x in (record.get("output_token_logprobs") or [])
            ]
            if len(output_token_logprobs) != len(output_token_ids):
                raise RuntimeError(
                    "Qwen gateway token/logprob length mismatch: "
                    f"{len(output_token_ids)} tokens vs {len(output_token_logprobs)} logprobs"
                )
            interactions.append(
                Interaction(
                    turn_idx=first_turn_idx + offset,
                    input_ids=[int(x) for x in (record.get("input_ids") or [])],
                    output_token_ids=output_token_ids,
                    output_token_logprobs=output_token_logprobs,
                    output_text=str(record.get("output_text") or ""),
                    finish_reason=str(record.get("finish_reason") or "stop"),
                    messages=list(record.get("messages") or []),
                    latency_ms=float(record.get("latency_ms") or 0.0),
                )
            )
        return interactions

    def _response_from_completed(
        self,
        completed: subprocess.CompletedProcess[str],
        text: str,
        raw_result: Any,
        tool_calls: list[dict[str, Any]],
        *,
        qwen_records: list[dict[str, Any]] | None = None,
    ) -> ClaudeCodeResponse:
        non_trainable = _env_flag(
            "CLAUDE_CODE_MARK_NON_TRAINABLE",
            self._default_non_trainable(),
        )
        info = {
            "termination_reasons": [],
            "harness_option": "claude-code",
            "harness": "claude-code",
            "source": (
                "claude-code-cli+sglang-gateway"
                if self._uses_sglang_gateway()
                else "claude-code-cli"
            ),
            "llm_backend": self._llm_backend,
            "session_id": self._session_id,
            "workspace": str(self._workspace),
            "task_path": self._task_meta.get("task_path"),
            "returncode": completed.returncode,
            "stdout_path": str(self._stdout_path),
            "stderr_path": str(self._stderr_path),
            "mcp_config_path": str(self._mcp_config_path),
            "output_format": self._output_format,
            "tool_calls_count": len(tool_calls),
            "tool_calls": list(tool_calls),
            "qwen_gateway_records_path": str(self._qwen_records_path),
            "qwen_gateway_turns": len(qwen_records or []),
            "non_trainable": non_trainable,
            "non_trainable_reason": None if not non_trainable else (
                "claude-code CLI uses an external model path without terminal-rl "
                "policy logprobs"
            ),
        }
        return ClaudeCodeResponse(
            msg=text,
            terminated=False,
            info=info,
            tool_calls=list(tool_calls),
            tool_calls_count=len(tool_calls),
            raw_result=raw_result,
        )

    def _interaction(
        self,
        turn_idx: int,
        prompt: str,
        text: str,
        latency_ms: float,
    ) -> Interaction:
        tokenizer = getattr(self._sglang_client, "tokenizer", None)
        input_ids = _tokenize(tokenizer, prompt)
        output_ids = _tokenize(tokenizer, text)
        return Interaction(
            turn_idx=turn_idx,
            input_ids=input_ids,
            output_token_ids=output_ids,
            output_token_logprobs=[0.0] * len(output_ids),
            output_text=text,
            finish_reason="stop",
            messages=[{"role": "user", "content": prompt}],
            latency_ms=latency_ms,
        )
