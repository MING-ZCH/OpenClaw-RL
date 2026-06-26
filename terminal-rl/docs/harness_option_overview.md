## 结论

`dev-agenticrl-safety-exploration-harness` 已加入 `camel-agent` / `a3s-code` 双 harness 路由，默认仍是 `camel-agent`。本地已能引用 `origin-lxt/a3s-harness-option-puyuan` 的 `fa1f857`；独立 review 后采用其核心的 a3s-code OpenAI bridge + external queue 架构，同时保留本分支更严格的别名兼容、run metadata、async close、a3s-only runtime env 和结构化轨迹字段。

## 合入摘要

| 项 | 结果 |
|---|---|
| base | `dev-agenticrl-safety-exploration` at `fc872aa8` |
| work branch | `dev-agenticrl-safety-exploration-harness` |
| initial compat commit | `3f204a2fa6e2` |
| parity optimization commit | `c89353a7` |
| reference PR commit | `origin-lxt/a3s-harness-option-puyuan` = `fa1f857f7cbd` |
| default behavior | `rollout_qwen3*.yaml` 和 `pu.sh` 默认 `camel-agent` |

## Diff Stat

```text
 terminal-rl/agent/a3s_code_agent.py                | 950 +++++++++++++++++++++
 terminal-rl/agent_runner.py                        |  91 +-
 terminal-rl/configs/rollout_qwen3.yaml             |   1 +
 terminal-rl/configs/rollout_qwen3_think.yaml       |   1 +
 terminal-rl/custom_types.py                        |   1 +
 terminal-rl/docs/README.md                         |   9 +
 terminal-rl/docs/a3s_code_baseline_guide.md        |  65 ++
 terminal-rl/docs/harness_option_compat_report.md   |  34 +
 terminal-rl/docs/harness_option_overview.md        |  53 ++
 terminal-rl/generate.py                            |  57 +-
 terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh         |  30 +
 terminal-rl/terminal-rl_qwen3-8b_pu.sh             |  96 ++-
 terminal-rl/tests/test_a3s_code_agent.py           | 275 ++++++
 terminal-rl/tests/test_agent_runner_harness_option.py | 148 ++++
 terminal-rl/tests/test_harness_option_routing.py   |  24 +
 15 files changed, 1821 insertions(+), 14 deletions(-)
```

## Harness 路由总览

| 阶段 | camel-agent | a3s-code |
|---|---|---|
| 配置入口 | `terminal-rl/configs/rollout_qwen3.yaml:3` | `HARNESS_OPTION=a3s-code` 或 `terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh:16` |
| shell 默认 | `terminal-rl/terminal-rl_qwen3-8b_pu.sh:122` | wrapper 覆盖后复用 `pu.sh` |
| 参数归一化 | `terminal-rl/agent_runner.py:31` 支持 `camel_agent` / `camel-agent` | 同处支持 `a3s_code` / `a3s-code` |
| 训练入口 | `terminal-rl/generate.py:1453` 读取 `harness_option` | 同入口，优先级 `harness_option > terminal_agent_type > camel_agent` |
| Agent 创建 | `terminal-rl/agent_runner.py:177` 创建 `CamelAgent` | `terminal-rl/agent_runner.py:177` 创建 `A3SCodeAgent` |
| 模型 turn | 无 `run_model_turn` 时走旧 SGLang fallback：`terminal-rl/agent_runner.py:107` | `terminal-rl/agent/a3s_code_agent.py:484` 通过 SDK loop 驱动 |
| 模型 bridge | 旧路径直接调用 SGLang | `terminal-rl/agent/a3s_code_agent.py:186` 本地 OpenAI-compatible bridge 转回 SGLang |
| 工具调用 | terminal-rl 执行 `env_client.exec_tool` | a3s SDK external tasks 调度回主 event loop，再走 terminal env lease：`terminal-rl/agent/a3s_code_agent.py:728` |
| 收尾 | `AgentRunner.close()` 对无 `close` 的 camel-agent no-op | `A3SCodeAgent.close()` 释放 SDK session、bridge、tmpdir |

## 关键路径

| 文件 | 要点 |
|---|---|
| `terminal-rl/custom_types.py:79` | `TurnResult.interactions` 为 Optional，旧调用不需要改 |
| `terminal-rl/generate.py:1581` | `interactions` 新旧兜底：`turn_state.interactions or [turn_state.interaction]` |
| `terminal-rl/generate.py:1596` | `turn_records` 新增 `harness_option` / `sdk_model_turns` / `sdk_tool_calls` |
| `terminal-rl/generate.py:1619` | SDK tool calls 也写回旧 `tool_calls` 字段，便于 trajectory/PRM 审计 |
| `terminal-rl/terminal-rl_qwen3-8b_pu.sh:332` | 仅非 dry-run 且 `a3s-code` 时尝试安装 SDK |
| `terminal-rl/terminal-rl_qwen3-8b_pu.sh:1193` | runtime env 只在 `a3s-code` 下追加 A3S_CODE_* |
