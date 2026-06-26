## 结论

两条 harness 的配置路由、构造、turn 执行、trajectory/PRM 记录和 close 降级已做兼容修复。本地 `fa1f857` parity review 后，a3s-code 已从 mock-only adapter 升级为 SDK bridge + external queue adapter；真实 GPU/Ray/a3s SDK 端到端仍未跑，是当前主要风险。

## A-E 兼容性检查

| 维度 | 结论 | 修复内容 | 修复 commit |
|---|---|---|---|
| A. `agent_runner.py` | 兼容 | `normalize_harness_option()` 覆盖下划线/连字符；camel 路径不依赖 `env_client/lease_id`；自定义 `run_model_turn` 支持 keyword/positional；`model_turn_count` 按 `interactions` 计数；`close()` 安全 no-op | `3f204a2fa6e2`, `c89353a7` |
| B. `custom_types.py` | 兼容 | `TurnResult.interactions` 为 Optional；fallback camel 路径补 `[interaction]`，旧读取仍可用 | `3f204a2fa6e2`, `c89353a7` |
| C. `generate.py` | 兼容 | agent 类型优先级为 `harness_option > terminal_agent_type > camel_agent`；新增 `sdk_model_turns/sdk_tool_calls` 为 additive 字段；SDK tool calls 回填旧 `tool_calls` 并可供 PRM 使用；`agent_runner.close()` 在 env lease close 前 best-effort 执行 | `3f204a2fa6e2`, `c89353a7` |
| D. rollout yaml | 兼容 | `rollout_qwen3.yaml:3`、`rollout_qwen3_think.yaml:3` 默认 `harness_option: camel-agent` | `3f204a2fa6e2` |
| E. `terminal-rl_qwen3-8b_pu.sh` | 兼容 | 默认 `HARNESS_OPTION=camel-agent`；a3s SDK 安装只在 `a3s-code && DRY_RUN!=1`；优先本地 SDK，缺失时安装 `a3s-code==3.3.0`；A3S runtime env 仅 a3s 注入；Ray job 用 `${TRAIN_PYTHON}` | `3f204a2fa6e2`, `c89353a7` |

## 独立取舍

| 来自 `origin-lxt/fa1f857` 的设计 | 本分支处理 |
|---|---|
| 采用 local OpenAI bridge 把 a3s SDK model call 转回 SGLang | 已采用：`terminal-rl/agent/a3s_code_agent.py:186` |
| 采用 `SessionQueueConfig` external queue 执行 SDK 工具 | 已采用，并把工具调用调度回 generate 主 event loop：`terminal-rl/agent/a3s_code_agent.py:728` |
| sync-only `close()`、较窄 alias、缺 run metadata | 未采用；保留 async close、别名归一化、`run_context/task_meta` 和结构化记录 |
| SDK tool calls 只放 a3s 私有字段 | 扩展：同时写 `sdk_tool_calls` 与旧 `tool_calls`，降低下游解析风险 |

## 自测结果

| 命令 | 结果 | 关键输出 |
|---|---|---|
| `python3 -m py_compile terminal-rl/agent/a3s_code_agent.py terminal-rl/agent_runner.py terminal-rl/generate.py` | 通过 | 无错误 |
| `bash -n terminal-rl/terminal-rl_qwen3-8b_pu.sh terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh` | 通过 | 无错误 |
| `/mnt/.../lightrft_py312/bin/python -m pytest terminal-rl/tests/test_a3s_code_agent.py terminal-rl/tests/test_agent_runner_harness_option.py terminal-rl/tests/test_harness_option_routing.py -v` | 通过 | `9 passed in 2.86s` |
| `HARNESS_OPTION=camel-agent DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_pu.sh` | 通过 | `Harness:  camel-agent`，打印 `[dry-run] python3 -u ... train_async.py` |
| `HARNESS_OPTION=a3s-code DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_pu.sh` | 通过 | `Harness:  a3s-code`，打印最终 `train_async.py` 命令 |
| `DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh` | 通过 | `ALGO=dapo`，`Harness:  a3s-code`，DAPO 参数进入最终命令 |

## 已知风险 / TODO

| # | 风险 / TODO | 状态 |
|---|---|---|
| 1 | 当前 a3s SDK 仍只用 mock 测试；未跑真实 SDK + Ray + GPU 端到端 | 未完成 |
| 2 | SDK external task schema 可能随 a3s-code 版本变化，当前兼容 `task_id/command_type/payload` 等常见字段 | 已防御 |
| 3 | ClawSentry pre-action 对 SDK 内部工具仍只能通过回填后的 trajectory 审计，不能像 camel 路径一样在 action 前拦截 | 已标注 |
| 4 | 真实运行前需确认 `A3S_CODE_REPO_ROOT` 或 `a3s-code==3.3.0` 可用 | 已标注 |
