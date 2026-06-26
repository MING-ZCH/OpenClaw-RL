from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path

TERMINAL_RL_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = TERMINAL_RL_DIR.parent
if str(TERMINAL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL_DIR))
if str(ROOT_DIR / "slime") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "slime"))

import agent.a3s_code_agent as a3s_agent_module


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        _ = add_special_tokens
        return {"input_ids": list(range(len(str(text).split())))}


class DummySGLangClient:
    tokenizer = DummyTokenizer()


class FakeResult:
    text = "done"
    tool_calls = []
    tool_calls_count = 0
    prompt_tokens = 7
    completion_tokens = 3
    total_tokens = 10


class FakeSessionOptions:
    pass


class FakeSessionQueueConfig:
    def __init__(self):
        self.handlers = []

    def set_lane_handler(self, lane, mode, timeout_ms):
        self.handlers.append((lane, mode, timeout_ms))


class FakePermissionPolicy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeBridge:
    closed = False

    def __init__(self, *, sglang_client, model_name):
        _ = (sglang_client, model_name)
        self.base_url = "http://127.0.0.1:9"
        self.turn_idx_base = 0

    def set_turn_idx_base(self, turn_idx):
        self.turn_idx_base = int(turn_idx)

    def start(self):
        pass

    def interactions(self):
        return [
            a3s_agent_module.Interaction(
                turn_idx=self.turn_idx_base,
                input_ids=[1, 2],
                output_token_ids=[3, 4],
                output_token_logprobs=[-0.1, -0.2],
                output_text="done",
                finish_reason="stop",
                messages=[{"role": "user", "content": "fix the bug"}],
            )
        ]

    def close(self):
        FakeBridge.closed = True


class FakeSession:
    closed = False

    def __init__(self, *, external_task=None):
        self.sent = []
        self.external_task = external_task
        self.completed = external_task is None
        self.completed_payloads = []

    def send(self, prompt):
        self.sent.append(prompt)
        deadline = time.time() + 2
        while not self.completed and time.time() < deadline:
            time.sleep(0.01)
        if not self.completed:
            raise RuntimeError("external task was not completed")
        return FakeResult()

    def pending_external_tasks(self):
        if self.completed or self.external_task is None:
            return []
        return [self.external_task]

    def complete_external_task(self, task_id, success, payload=None, error=None):
        self.completed = True
        self.completed_payloads.append((task_id, success, payload, error))

    def close(self):
        self.closed = True
        FakeSession.closed = True


class FakeAgent:
    created_config = None
    last_session = None
    next_session = None
    last_opts = None

    @classmethod
    def create(cls, path):
        config_path = Path(path)
        assert config_path.exists()
        assert "base_url" in config_path.read_text()
        cls.created_config = str(config_path)
        return cls()

    def session(self, workspace, opts):
        assert Path(workspace).exists()
        FakeAgent.last_opts = opts
        FakeAgent.last_session = FakeAgent.next_session or FakeSession()
        return FakeAgent.last_session


class FakeEnvClient:
    def __init__(self):
        self.calls = []
        self.heartbeats = []

    async def heartbeat(self, lease_id):
        self.heartbeats.append(lease_id)

    async def exec_tool(self, lease_id, tool_name, args):
        self.calls.append((lease_id, tool_name, args))
        return "tool output"


def _patch_sdk(monkeypatch):
    FakeBridge.closed = False
    FakeSession.closed = False
    FakeAgent.created_config = None
    FakeAgent.last_session = None
    FakeAgent.next_session = None
    FakeAgent.last_opts = None
    monkeypatch.setattr(
        a3s_agent_module,
        "_bootstrap_a3s_code",
        lambda: (
            FakeAgent,
            FakePermissionPolicy,
            FakeSessionOptions,
            FakeSessionQueueConfig,
        ),
    )
    monkeypatch.setattr(a3s_agent_module, "A3SOpenAIModelBridge", FakeBridge)
    monkeypatch.setenv("A3S_CODE_TURN_TIMEOUT_SEC", "5")
    monkeypatch.delenv("A3S_CODE_TERMINAL_RL_EXTRA_PROMPT", raising=False)
    monkeypatch.delenv("A3S_CODE_EXTRA_PROMPT", raising=False)


def test_a3s_code_agent_run_model_turn_with_mock_sdk(monkeypatch):
    _patch_sdk(monkeypatch)
    monkeypatch.setenv("A3S_CODE_WORKSPACE_ROOT", "/tmp/openclaw_a3s_test_workspaces")

    agent = a3s_agent_module.A3SCodeAgent(
        model_type="Qwen3",
        sglang_client=DummySGLangClient(),
        env_client=None,
        lease_id=None,
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

    assert FakeAgent.created_config is not None
    assert FakeAgent.last_session.sent == ["fix the bug"]
    assert FakeAgent.last_opts.max_tool_rounds == 10
    assert "Terminal-RL Harness Instructions" in FakeAgent.last_opts.extra
    assert "limited to about 10 A3S agent turns" in FakeAgent.last_opts.extra
    assert result.interaction.output_text == "done"
    assert result.interactions == [result.interaction]
    assert result.tool_call_requests == []
    assert result.model_response.tool_calls_count == 0

    final = agent.finalize_response(result.model_response)
    assert isinstance(final, a3s_agent_module.A3SCodeFinalResponse)
    assert final.msg == "done"
    assert final.info["harness_option"] == "a3s-code"
    assert final.info["workspace"].endswith("a3s-code-seta-task-abc123")

    asyncio.run(agent.close())
    assert FakeSession.closed is True
    assert FakeBridge.closed is True


def test_a3s_code_agent_external_tasks_route_to_terminal_env(monkeypatch):
    _patch_sdk(monkeypatch)
    monkeypatch.setenv("A3S_CODE_WORKSPACE_ROOT", "/tmp/openclaw_a3s_test_workspaces")
    FakeAgent.next_session = FakeSession(
        external_task={
            "task_id": "task-1",
            "command_type": "bash",
            "payload": {"command": "pwd"},
        }
    )
    env_client = FakeEnvClient()

    agent = a3s_agent_module.A3SCodeAgent(
        model_type="Qwen3",
        sglang_client=DummySGLangClient(),
        env_client=env_client,
        lease_id="lease-1",
        run_context=types.SimpleNamespace(uid="abc123"),
        task_meta={"task_name": "seta-task"},
        max_total_tokens=8192,
    )
    agent.start_turn_loop("fix the bug")

    result = asyncio.run(
        agent.run_model_turn(
            context_messages=[{"role": "user", "content": "fix the bug"}],
            sglang_client=DummySGLangClient(),
            tool_schemas=[],
            turn_idx=0,
        )
    )

    assert env_client.heartbeats == ["lease-1"]
    assert env_client.calls == [("lease-1", "shell_exec", {"command": "pwd"})]
    assert FakeAgent.last_session.completed_payloads == [
        ("task-1", True, {"output": "tool output", "exit_code": 0}, None)
    ]
    assert result.model_response.tool_calls == [
        {
            "tool_call_id": "task-1",
            "tool_name": "shell_exec",
            "sdk_tool_name": "bash",
            "args": {"command": "pwd"},
            "sdk_args": {"command": "pwd"},
            "result": "tool output",
            "source": "a3s-code-sdk",
        }
    ]


def test_a3s_code_agent_prompt_extra_can_be_disabled_or_extended(monkeypatch):
    _patch_sdk(monkeypatch)
    monkeypatch.setenv("A3S_CODE_WORKSPACE_ROOT", "/tmp/openclaw_a3s_test_workspaces")
    monkeypatch.setenv("A3S_CODE_TERMINAL_RL_EXTRA_PROMPT", "0")
    monkeypatch.setenv("A3S_CODE_EXTRA_PROMPT", "Custom a3s extra instruction.")

    agent = a3s_agent_module.A3SCodeAgent(
        model_type="Qwen3",
        sglang_client=DummySGLangClient(),
        env_client=None,
        lease_id=None,
        run_context=types.SimpleNamespace(uid="abc123"),
        task_meta={"task_name": "seta-task"},
        max_total_tokens=8192,
    )
    agent.start_turn_loop("fix the bug")
    asyncio.run(
        agent.run_model_turn(
            context_messages=[{"role": "user", "content": "fix the bug"}],
            sglang_client=DummySGLangClient(),
            tool_schemas=[],
            turn_idx=0,
        )
    )

    assert FakeAgent.last_opts.extra == "Custom a3s extra instruction."


def test_a3s_code_tool_timeout_is_capped_below_turn_timeout(monkeypatch):
    _patch_sdk(monkeypatch)
    monkeypatch.setenv("A3S_CODE_WORKSPACE_ROOT", "/tmp/openclaw_a3s_test_workspaces")
    monkeypatch.setenv("A3S_CODE_TURN_TIMEOUT_SEC", "10")
    monkeypatch.setenv("A3S_CODE_TOOL_TIMEOUT_MS", "7200000")

    agent = a3s_agent_module.A3SCodeAgent(
        model_type="Qwen3",
        sglang_client=DummySGLangClient(),
        env_client=None,
        lease_id=None,
        run_context=types.SimpleNamespace(uid="abc123"),
        task_meta={"task_name": "seta-task"},
        max_total_tokens=8192,
    )
    agent.start_turn_loop("fix the bug")
    asyncio.run(
        agent.run_model_turn(
            context_messages=[{"role": "user", "content": "fix the bug"}],
            sglang_client=DummySGLangClient(),
            tool_schemas=[],
            turn_idx=0,
        )
    )

    assert agent._tool_timeout_ms == 9000
    assert FakeAgent.last_opts.tool_timeout_ms == 9000
    assert FakeAgent.last_opts.queue_config.handlers == [
        ("query", "external", 9000),
        ("execute", "external", 9000),
    ]


def test_a3s_code_tool_mapping_helpers():
    assert a3s_agent_module.A3SCodeAgent._map_tool_call(
        "bash", {"command": "pwd"}
    ) == ("shell_exec", {"command": "pwd"})
    assert a3s_agent_module.A3SCodeAgent._map_tool_call(
        "write", {"path": "x.txt", "content": "hi"}
    ) == ("shell_write_content_to_file", {"file_path": "x.txt", "content": "hi"})
