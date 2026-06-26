from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TERMINAL_RL = ROOT / "terminal-rl"


def test_rollout_configs_default_to_camel_agent():
    for name in ("rollout_qwen3.yaml", "rollout_qwen3_think.yaml"):
        cfg = yaml.safe_load((TERMINAL_RL / "configs" / name).read_text())
        assert cfg["harness_option"] == "camel-agent"


def test_training_script_routes_harness_without_polluting_camel_runtime():
    script = (TERMINAL_RL / "terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh").read_text()
    assert 'HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"' in script
    assert 'cfg["harness_option"] = harness_option' in script
    assert 'if [[ "${HARNESS_OPTION}" == "a3s-code" && "${DRY_RUN}" != "1" ]]' in script
    assert 'if [[ "${HARNESS_OPTION}" == "a3s-code" ]]; then' in script
    assert 'claude-code|claude_code)' in script
    assert 'if [[ "${HARNESS_OPTION}" == "claude-code" && "${DRY_RUN}" != "1" ]]' in script
    assert 'CLAUDE_CODE_MARK_NON_TRAINABLE' in script
    assert '\\"HARNESS_OPTION\\": \\"${HARNESS_OPTION}\\"' in script
    assert '-- "${TRAIN_PYTHON}" -u "${SLIME_DIR}/train_async.py"' in script


def test_claude_code_wrapper_delegates_to_nodynamic_base():
    script = (
        TERMINAL_RL / "terminal-rl_qwen3-8b_seta_dapo_nodynamic_claude_code_pu.sh"
    ).read_text()
    assert 'HARNESS_OPTION="${HARNESS_OPTION:-claude_code}"' in script
    assert 'CLAUDE_CODE_LLM_BACKEND="${CLAUDE_CODE_LLM_BACKEND:-sglang}"' in script
    assert 'CLAUDE_CODE_MARK_NON_TRAINABLE="0"' in script
    assert 'CLAUDE_CODE_MARK_NON_TRAINABLE="1"' in script
    assert 'DAPO_DYNAMIC_SAMPLING="${DAPO_DYNAMIC_SAMPLING:-0}"' in script
    assert 'terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh' in script


def test_exploration_script_accepts_claude_code_harness_alias():
    script = (
        TERMINAL_RL / "terminal-rl_qwen3-8b_mixed_dapo_nodynamic_exploration_pu.sh"
    ).read_text()
    assert "claude-code|claude_code)" in script
    assert "Use: camel-agent|a3s-code|claude_code" in script
