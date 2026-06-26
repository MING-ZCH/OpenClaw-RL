from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

TERMINAL_RL_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = TERMINAL_RL_DIR.parent
if str(TERMINAL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL_DIR))
if str(ROOT_DIR / "slime") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "slime"))

import agent.claude_code_agent as claude_agent_module
from agent.claude_code_qwen_gateway import ClaudeCodeQwenGateway


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        _ = add_special_tokens
        return {"input_ids": list(range(len(str(text).split())))}

    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        return list(range(len(str(text).split())))

    def apply_chat_template(
        self,
        messages,
        tools=None,
        add_generation_prompt=True,
        tokenize=True,
        **kwargs,
    ):
        _ = (tools, add_generation_prompt, tokenize, kwargs)
        text = "\n".join(str(m.get("content", "")) for m in messages)
        return self.encode(text)


class DummySGLangClient:
    tokenizer = DummyTokenizer()
    sampling_params = {"max_new_tokens": 32, "temperature": 1.0}
    url = "http://127.0.0.1:9/generate"
    session_id = None
    tool_call_parser = "qwen25"
    request_timeout = 5
    max_retries = 1

    def _apply_chat_template(self, messages, tools):
        return self.tokenizer.apply_chat_template(messages, tools=tools)

    def _truncate_input_ids(self, input_ids):
        return list(input_ids)


class FakeEnvClient:
    base_url = "http://127.0.0.1:18081"


def test_claude_code_agent_runs_cli_and_writes_mcp_config(tmp_path, monkeypatch):
    fake_cli = tmp_path / "claude"
    fake_cli.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "pathlib.Path('argv.json').write_text(json.dumps(sys.argv[1:]))\n"
        "print(json.dumps({'result': 'done from claude'}))\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    monkeypatch.setenv("CLAUDE_CODE_CLI", str(fake_cli))
    monkeypatch.setenv("CLAUDE_CODE_LLM_BACKEND", "anthropic")
    monkeypatch.setenv("CLAUDE_CODE_MARK_NON_TRAINABLE", "1")
    monkeypatch.setenv("CLAUDE_CODE_WORKSPACE_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv("CLAUDE_CODE_MCP_PYTHON", sys.executable)
    monkeypatch.setenv("CLAUDE_CODE_TURN_TIMEOUT_SEC", "5")
    monkeypatch.setenv("CLAUDE_CODE_TOOL_TIMEOUT_MS", "9000")
    monkeypatch.delenv("CLAUDE_CODE_SYSTEM_PROMPT", raising=False)

    agent = claude_agent_module.ClaudeCodeAgent(
        model_type="Qwen3",
        sglang_client=DummySGLangClient(),
        env_client=FakeEnvClient(),
        lease_id="lease-1",
        run_context=types.SimpleNamespace(uid="abc123"),
        task_meta={"task_name": "seta-task", "task_path": "seta_env/1"},
        max_total_tokens=8192,
    )
    agent.start_turn_loop("fix the bug")
    context, terminated = asyncio.run(agent.get_turn_context())
    assert terminated is None
    assert context == [{"role": "user", "content": "fix the bug"}]

    result = asyncio.run(
        agent.run_model_turn(
            context_messages=context,
            sglang_client=DummySGLangClient(),
            tool_schemas=[],
            turn_idx=0,
        )
    )

    assert result.interaction.output_text == "done from claude"
    assert result.model_response.info["harness_option"] == "claude-code"
    assert result.model_response.info["non_trainable"] is True
    assert result.model_response.tool_calls_count == 0

    workspace = Path(result.model_response.info["workspace"])
    argv = json.loads((workspace / "argv.json").read_text())
    assert "--mcp-config" in argv
    assert "--allowedTools" in argv
    assert "--permission-mode" in argv
    cfg = json.loads((workspace / "claude_mcp_config.json").read_text())
    server = cfg["mcpServers"]["terminal_rl"]
    assert server["env"]["CLAUDE_CODE_TERMINAL_ENV_SERVER_URL"] == FakeEnvClient.base_url
    assert server["env"]["CLAUDE_CODE_TERMINAL_LEASE_ID"] == "lease-1"


def test_claude_code_sglang_backend_records_qwen_logprobs(tmp_path, monkeypatch):
    qwen_record = {
        "messages": [{"role": "user", "content": "fix the bug"}],
        "input_ids": [1, 2, 3],
        "output_token_ids": [101, 102],
        "output_token_logprobs": [-0.1, -0.2],
        "output_text": "qwen final answer",
        "finish_reason": "stop",
        "latency_ms": 12.0,
    }

    class FakeGateway:
        def __init__(self, *, sglang_client, records_path, model_name):
            _ = (sglang_client, model_name)
            self.base_url = "http://127.0.0.1:12345"
            self.records_path = records_path

        def start(self):
            self.records_path.write_text(json.dumps(qwen_record) + "\n", encoding="utf-8")
            return self.base_url

        def close(self):
            return None

        def records(self):
            return [dict(qwen_record)]

    monkeypatch.setattr(claude_agent_module, "ClaudeCodeQwenGateway", FakeGateway)

    fake_cli = tmp_path / "claude"
    fake_cli.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "base = os.environ['ANTHROPIC_BASE_URL']\n"
        "assert base == 'http://127.0.0.1:12345'\n"
        "assert os.environ['ANTHROPIC_API_KEY'] == 'terminal-rl-qwen'\n"
        "_ = sys.stdin.read()\n"
        "print(json.dumps({'result': 'qwen final answer'}))\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)

    monkeypatch.setenv("CLAUDE_CODE_CLI", str(fake_cli))
    monkeypatch.setenv("CLAUDE_CODE_LLM_BACKEND", "sglang")
    monkeypatch.delenv("CLAUDE_CODE_MARK_NON_TRAINABLE", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_WORKSPACE_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv("CLAUDE_CODE_MCP_PYTHON", sys.executable)
    monkeypatch.setenv("CLAUDE_CODE_TURN_TIMEOUT_SEC", "5")
    monkeypatch.setenv("CLAUDE_CODE_TOOL_TIMEOUT_MS", "9000")

    agent = claude_agent_module.ClaudeCodeAgent(
        model_type="Qwen3",
        sglang_client=DummySGLangClient(),
        env_client=FakeEnvClient(),
        lease_id="lease-1",
        run_context=types.SimpleNamespace(uid="abc123"),
        task_meta={"task_name": "seta-task", "task_path": "seta_env/1"},
        max_total_tokens=8192,
    )
    agent.start_turn_loop("fix the bug")
    context, _ = asyncio.run(agent.get_turn_context())
    result = asyncio.run(
        agent.run_model_turn(
            context_messages=context,
            sglang_client=DummySGLangClient(),
            tool_schemas=[],
            turn_idx=0,
        )
    )

    assert result.interaction.output_text == "qwen final answer"
    assert result.interaction.output_token_ids == [101, 102]
    assert result.interaction.output_token_logprobs == [-0.1, -0.2]
    assert result.model_response.info["llm_backend"] == "sglang"
    assert result.model_response.info["non_trainable"] is False
    assert result.model_response.info["qwen_gateway_turns"] == 1


def test_qwen_gateway_converts_anthropic_messages_to_sglang_logprob_record(tmp_path):
    class FakeSGLangClient(DummySGLangClient):
        seen_payload = None

    client = FakeSGLangClient()
    gateway = ClaudeCodeQwenGateway(
        sglang_client=client,
        records_path=tmp_path / "records.jsonl",
        model_name="qwen-8b-test",
    )

    def fake_post(payload):
        client.seen_payload = payload
        return {
            "text": "qwen final answer",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 101], [-0.2, 102]],
            },
        }

    gateway._post_sglang = fake_post
    response = gateway._build_message_response(
        {
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "fix the bug"}],
            "tools": [
                {
                    "name": "mcp__terminal_rl__shell_exec",
                    "description": "",
                    "input_schema": {"type": "object"},
                }
            ],
        }
    )

    assert response["content"][0]["text"] == "qwen final answer"
    assert client.seen_payload["return_logprob"] is True
    assert client.seen_payload["sampling_params"]["max_new_tokens"] == 16
    record = gateway.records()[0]
    assert record["output_token_ids"] == [101, 102]
    assert record["output_token_logprobs"] == [-0.1, -0.2]


def test_parse_claude_stream_json_prefers_result_event():
    text, raw = claude_agent_module._parse_claude_output(
        '{"type":"assistant","message":{"content":[{"text":"thinking"}]}}\n'
        '{"type":"result","result":"final answer"}\n',
        "stream-json",
    )
    assert text == "final answer"
    assert isinstance(raw, list)
