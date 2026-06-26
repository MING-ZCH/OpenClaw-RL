# claude_code harness support

生成时间：2026-06-17 16:28:21 HKT

## 结论

本次在 `harness_option` 中新增 `claude_code`/`claude-code` 支持，并保持现有 `camel-agent`、`a3s-code` 行为不变。实现采用与 `a3s-code` 相同的接入模式：入口脚本写入 rollout 配置，`slime` 读取 custom config，`generate.py` 统一创建 agent runner，agent 自己执行外部 harness 并把 turn、tool call、trajectory 交回既有 reward/eval 流程。

重要限制：Claude Code CLI 是外部模型执行路径，不能提供当前 Qwen/SGLang policy 的真实 logprob。为避免污染 on-policy RL，`claude-code` 样本默认标记为 non-trainable，但仍保留 trajectory、tool calls、reward/score 和评测结果。除非另行实现 Anthropic/Claude Code 到训练 policy 的 logprob 对齐，不建议设置 `CLAUDE_CODE_MARK_NON_TRAINABLE=0`。

## 基准脚本流程分析

基准脚本：`terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh`

- 行 16-19 设置 `DATASET=seta`、`ALGO=dapo`、`HARNESS_OPTION=${HARNESS_OPTION:-camel-agent}`、`CUSTOM_CONFIG_PATH=rollout_qwen3_think.yaml`。
- 行 21-24 对齐 rollout 规模：`ROLLOUT_BATCH_SIZE=8`、`N_SAMPLES=8`、`MAX_TURN=10`、`MAX_CKPT_KEEP=2`。
- 行 26-31 固定 `DAPO_DYNAMIC_SAMPLING=0`，不启用 dynamic sampling。
- 行 33-41 设置 run id/name 并委托 `terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh`。

主 base 脚本处理链路：

- `terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:120-132` 读取并规范化 `HARNESS_OPTION`。
- `terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:517-541` 将规范化后的 `HARNESS_OPTION` 写入 per-run `rollout_config.yaml` 的 `harness_option`。
- `slime/slime/utils/arguments.py:1946-1952` 读取 `--custom-config-path`，把 YAML 字段设置到 `args`。
- `terminal-rl/generate.py:3024-3028` 用 `harness_option > terminal_agent_type > camel_agent` 的优先级选择 agent type。
- `terminal-rl/generate.py:3116-3127` 调 `create_agent_runner(...)`，把 `env_client`、`lease_id`、`run_context`、`task_meta` 传给 harness。
- `terminal-rl/agent_runner.py:31-53` 规范化 harness option；`terminal-rl/agent_runner.py:181-237` 分支创建具体 agent。
- `terminal-rl/generate.py:3151-3220` 进入统一 rollout loop，记录 `harness_option`、`sdk_model_turns`、`sdk_tool_calls`、`tool_calls`。
- `terminal-rl/generate.py:3396` 仍调用现有 `env_client.evaluate(lease_id, trajectory=eval_payload)` 计算 reward/score。

## 当前 harness_option 支持取值

变更前主链路支持：

- `camel-agent`，别名 `camel_agent`。Python 层还接受 `camel`、`camelagent`。
- `a3s-code`，别名 `a3s_code`。Python 层还接受 `a3s`、`a3s-code-agent`、`a3s-code-harness`。

变更后新增：

- `claude-code`，别名 `claude_code`。Python 层还接受 `claude`、`claude-code-cli`、`claude-code-harness`。

默认值仍是 `camel-agent`：`terminal-rl/configs/rollout_qwen3.yaml:3`、`terminal-rl/configs/rollout_qwen3_think.yaml:3` 和基准 wrapper 均未改默认行为。

## 全局搜索位置

以下为与读取、分支、传参、运行相关的源代码位置；搜索也命中了 `terminal-rl/logs/latest` 和既有 docs，但那些是历史运行日志或说明文档，不属于运行时调用链。

- `terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh:18,37,41`：基准 wrapper 设置并导出 `HARNESS_OPTION`，委托主 base 脚本。
- `terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh:18`：A3S wrapper 默认 `HARNESS_OPTION=a3s-code`。
- `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:120-132`：主 base 脚本规范化 `camel-agent`、`a3s-code`、`claude-code`。
- `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:323`：`HARNESS_OPTION` 进入 run tag。
- `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:385-410`：Claude Code env 初始化时临时关闭 xtrace，避免密钥打入日志。
- `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:453-468`：Claude Code CLI/MCP/auth preflight。
- `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:517-541`：写入 rollout YAML 的 `harness_option`。
- `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:563,587`：仅在对应 harness 分支执行 A3S/Claude preflight。
- `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:1582,1801-1810`：run config 记录 harness 和非敏感 Claude 配置。
- `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:1840-1891,1966,2085,2119`：Ray runtime env 按 harness 追加 A3S/Claude env，并对 Claude 分支关闭 xtrace。
- `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_exploration_pu.sh:117,132-143,147,320,471`：exploration 入口同样接受并规范化 `claude_code`。
- `terminal-rl/configs/rollout_qwen3.yaml:3`、`terminal-rl/configs/rollout_qwen3_think.yaml:3`：默认 `harness_option: camel-agent`。
- `slime/slime/utils/arguments.py:1946-1952`：读取 custom config 并设置 `args.harness_option`。
- `terminal-rl/generate.py:34,3024-3028,3116-3127,3181,3208-3219,3396,3817-3826`：导入 routing helper、选择 harness、创建 runner、记录 turn/tool calls、评测、Claude non-trainable 标记。
- `terminal-rl/agent_runner.py:31-53,181-237`：规范化和分发到 `CamelAgent`、`A3SCodeAgent`、`ClaudeCodeAgent`。
- `terminal-rl/agent/a3s_code_agent.py:640,1065`：A3S response info 中写入 `harness_option=a3s-code`。
- `terminal-rl/agent/claude_code_agent.py:326,530`：Claude Code response info 中写入 `harness_option=claude-code`。
- `terminal-rl/tests/test_agent_runner_harness_option.py:19-26,71-101`、`terminal-rl/tests/test_harness_option_routing.py:17-44`、`terminal-rl/tests/test_claude_code_agent.py:33-74`：新增/更新测试覆盖别名、路由、脚本和 CLI/MCP config。

## 参照实现：a3s-code 数据流

参照实现选择 `a3s-code`，因为它也是外部 harness，和 `camel-agent` 的内置 SGLang tool loop 不同。

1. `terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh:18` 或用户环境设置 `HARNESS_OPTION=a3s-code`。
2. 主 base 脚本在 `terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:123-126` 规范化，并在 `:563` 非 dry-run 时执行 A3S import/preflight。
3. 同脚本在 `:517-541` 写 `rollout_config.yaml`，在 `:1840-1858` 仅给 A3S 分支追加 A3S runtime env。
4. `slime/slime/utils/arguments.py:1946-1952` 把 `harness_option` 注入 `args`。
5. `terminal-rl/generate.py:3024-3028` 读取并规范化；`terminal-rl/agent_runner.py:204-216` 创建 `A3SCodeAgent`，传入 env lease、run context、task meta。
6. A3S agent 自己执行 harness turn，并在 `terminal-rl/agent/a3s_code_agent.py:640,1065` 写入 `harness_option=a3s-code`。
7. `terminal-rl/generate.py:3151-3220` 使用统一 turn record 格式收集 SDK model turns/tool calls；`terminal-rl/generate.py:3396` 使用同一 env eval 口径计算 reward。

`claude-code` 按这个模式接入：脚本只在 Claude 分支追加 Claude env/preflight；Python routing 创建新 agent；agent 自行运行外部 CLI，但返回既有 `TurnResult`/response 结构，后续 trajectory/eval/reward 走统一流程。

## 改动清单

- `terminal-rl/agent_runner.py:31-53,217-229`：新增 `claude-code` 规范化别名和 `ClaudeCodeAgent` 路由。
- `terminal-rl/agent/claude_code_agent.py:33-577`：新增 Claude Code CLI harness。负责 workspace、CLI 参数、MCP config、输出解析、tool call 读取、`TurnResult` 生成和 non-trainable metadata。
- `terminal-rl/agent/claude_code_mcp_server.py:37-178`：新增 stdio MCP server，把 Claude Code 的 MCP tools 转发到 terminal env `/exec_tool`。
- `terminal-rl/generate.py:3817-3826`：仅在 `agent_type == "claude-code"` 且 `CLAUDE_CODE_MARK_NON_TRAINABLE` 开启时移除训练样本并记录原因。
- `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:120-132,385-410,453-468,587-588,1801-1810,1845-1891,2119`：新增 Claude 分支、env 默认、preflight、run config、runtime env、xtrace 密钥保护。
- `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_exploration_pu.sh:132-143`：同步接受 `claude_code`，避免同名入口不一致。
- `terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_claude_code_pu.sh:1-37`：新增 SetA/DAPO/nodynamic Claude Code wrapper，保持基准脚本参数并仅默认 `HARNESS_OPTION=claude_code`。
- `terminal-rl/tests/test_agent_runner_harness_option.py:19-26,71-101`：覆盖别名和 routing。
- `terminal-rl/tests/test_harness_option_routing.py:17-44`：覆盖脚本分支、wrapper、exploration alias。
- `terminal-rl/tests/test_claude_code_agent.py:33-96`：覆盖 fake CLI、MCP config、输出解析。

## claude_code 接入设计

入口调用约定：

- 用户可传 `HARNESS_OPTION=claude_code` 或 `HARNESS_OPTION=claude-code`；内部统一为 `claude-code`。
- 新 wrapper 默认值为 `claude_code`，其他 rollout 参数与基准脚本一致。
- per-run `rollout_config.yaml` 仍写 `harness_option`，由 `slime` 注入 `args`。

执行方式：

- `ClaudeCodeAgent` 每个 terminal-rl task 建一个 workspace：默认 `${RUN_DIR}/claude_code_workspaces/claude-code-<task>-<uid>`。
- agent 写出 `claude_mcp_config.json`，把 `terminal_rl` MCP server 作为 stdio server 暴露给 Claude Code。
- CLI 默认命令形态为 `claude -p --output-format json --max-turns <MAX_TURN> --mcp-config <config> --allowedTools <terminal_rl tools> --permission-mode bypassPermissions`。
- agent 解析 JSON 或 stream JSON 输出，生成一个 `Interaction` 和 `ClaudeCodeResponse`。MCP tool calls 从 `terminal_rl_tool_calls.jsonl` 读取并放入 `tool_calls`/`sdk_tool_calls`。

工具与环境：

- MCP server 暴露 `shell_exec`、`shell_view`、`shell_write_to_process`、`shell_write_content_to_file`。
- MCP server 使用现有 env lease 调 `/exec_tool`，不直接访问本地任务文件系统。
- tool timeout、HTTP retry、日志格式与现有 terminal env 调用保持一致的 JSON 记录方式。

日志和密钥：

- 运行配置只记录非敏感 Claude 配置，不记录 API key/token。
- Claude 分支构造 runtime env 和 `ray job submit` 时临时关闭 xtrace；env 默认初始化也关闭 xtrace，避免 `ANTHROPIC_API_KEY` 等值进入日志。

## 兼容性说明

- `camel-agent` 和 `a3s-code` 的 normalization、routing、preflight、runtime env 均保持原分支行为。
- 默认 config 和基准 wrapper 默认值仍为 `camel-agent`。
- 新增逻辑都以 `HARNESS_OPTION == "claude-code"` 为条件，或仅增加别名映射。
- `generate.py` 的样本移除逻辑只影响 `agent_type == "claude-code"`。

## 性能对齐说明

- wrapper 参数与基准一致：`ROLLOUT_BATCH_SIZE=8`、`N_SAMPLES=8`、`MAX_TURN=10`、`DAPO_DYNAMIC_SAMPLING=0`。
- Claude 默认超时与 A3S 对齐：`CLAUDE_CODE_TURN_TIMEOUT_SEC=900`、`CLAUDE_CODE_TOOL_TIMEOUT_MS=300000`、`CLAUDE_CODE_MAX_TOOL_ROUNDS=10`。
- reward/eval 口径不变：仍由 `env_client.evaluate()` 对同一 trajectory 计算。
- 并发由现有 rollout/Ray 并发控制。单个 Claude Code agent 的 CLI 调用以一次外部进程为单位；实现用普通线程承载同步 CLI，并在 async loop 中短轮询，避免当前环境中 `asyncio.to_thread/subprocess` 的挂起问题。
- 差异：Claude Code 依赖外部 API/CLI，吞吐会受 API 限流、CLI 启动和网络延迟影响；不参与 SGLang 批量推理。样本默认 non-trainable，因此适合评测/采样/对比，不适合作为现有 Qwen policy 的 on-policy 训练样本。

## 环境变量与配置

必需或常用：

- `CLAUDE_CODE_CLI`：Claude Code CLI 路径，默认 `claude`。
- `ANTHROPIC_API_KEY` 或 `ANTHROPIC_AUTH_TOKEN`：鉴权；也可依赖 CLI 已登录状态。
- `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_URL`：可选 endpoint 覆盖。
- `CLAUDE_CODE_MODEL`：可选模型名，空值则使用 CLI 默认。
- `CLAUDE_CODE_MCP_PYTHON`：运行 MCP server 的 Python，默认 `${TRAIN_PYTHON}`，必须能 `import mcp.server.fastmcp`。

调优项：

- `CLAUDE_CODE_WORKSPACE_ROOT`：workspace 根目录，默认 `${RUN_DIR}/claude_code_workspaces`。
- `CLAUDE_CODE_TURN_TIMEOUT_SEC`：单 task CLI 超时，默认 `900`。
- `CLAUDE_CODE_TOOL_TIMEOUT_MS`：单 tool timeout，默认 `300000`。
- `CLAUDE_CODE_MAX_TOOL_ROUNDS`：传给 CLI 的 `--max-turns`，默认 `10`。
- `CLAUDE_CODE_OUTPUT_FORMAT`：默认 `json`。
- `CLAUDE_CODE_PERMISSION_MODE`：默认 `bypassPermissions`。
- `CLAUDE_CODE_ALLOWED_TOOLS`：默认只允许 `mcp__terminal_rl__shell_exec,mcp__terminal_rl__shell_view,mcp__terminal_rl__shell_write_to_process,mcp__terminal_rl__shell_write_content_to_file`。
- `CLAUDE_CODE_DISALLOWED_TOOLS`、`CLAUDE_CODE_EXTRA_ARGS`、`CLAUDE_CODE_SYSTEM_PROMPT`：可选 CLI 扩展。
- `CLAUDE_CODE_HTTP_MAX_RETRIES`、`CLAUDE_CODE_HTTP_RETRY_DELAY`：MCP 到 env server 的 retry。
- `CLAUDE_CODE_MARK_NON_TRAINABLE`：默认 `1`。

不要在脚本中硬编码任何密钥；使用环境变量或 CLI 自带鉴权状态。

## 运行/复现步骤

Dry run：

```bash
MAX_CKPT_KEEP=0 DRY_RUN=1 \
  bash terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_claude_code_pu.sh
```

真实运行示例：

```bash
export CLAUDE_CODE_CLI=claude
export ANTHROPIC_API_KEY=<redacted>
export CLAUDE_CODE_MODEL=<optional-model-name>
export CLAUDE_CODE_MCP_PYTHON=/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python

bash terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_claude_code_pu.sh
```

前置条件：

- Ray worker 所在环境可执行 `CLAUDE_CODE_CLI`。
- `CLAUDE_CODE_MCP_PYTHON` 环境可 import `mcp.server.fastmcp`。
- Anthropic/Claude Code 鉴权在 worker 环境可用。
- 若使用远端 env pool，`WORKER_URLS` 或 `terminal-rl/worker_urls.txt` 与现有训练流程一致。

## 验证结果

已执行：

```bash
python3 -m py_compile terminal-rl/agent/claude_code_agent.py terminal-rl/agent/claude_code_mcp_server.py terminal-rl/agent_runner.py terminal-rl/generate.py
bash -n terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh
bash -n terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_exploration_pu.sh
bash -n terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_claude_code_pu.sh
timeout 60s /mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python -m pytest terminal-rl/tests/test_claude_code_agent.py terminal-rl/tests/test_agent_runner_harness_option.py terminal-rl/tests/test_harness_option_routing.py -v
MAX_CKPT_KEEP=0 DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_claude_code_pu.sh
```

结果：

- Python 编译通过。
- 3 个 shell 脚本语法检查通过。
- 相关 pytest：`11 passed in 5.80s`。
- dry-run 到最终 `train_async.py` 命令，显示 `Harness: claude-code`，并生成 per-run config。
- 使用假 `ANTHROPIC_API_KEY=supersecret` 做 dry-run，日志中未检出 `supersecret`。

## 假设与待确认

- 假设目标环境中的 Claude Code CLI 支持 `-p`、`--output-format`、`--max-turns`、`--mcp-config`、`--allowedTools`、`--permission-mode`、`--append-system-prompt`。
- 假设 Claude Code 对 MCP tool 的名字采用 `mcp__terminal_rl__<tool>` 格式。
- 假设 CLI 鉴权可通过环境变量或已有 CLI 登录状态完成。
- 待确认：如果需要把 Claude Code 结果作为 trainable on-policy 样本，需要另行设计与训练 policy 对齐的 logprob 路径。

## 已知限制

- 默认 non-trainable；这会使 Claude Code 分支更适合评测和轨迹采样，而不是直接训练 Qwen policy。
- Claude Code CLI/API 的速率限制、网络错误、CLI 启动开销会影响吞吐。
- `CLAUDE_CODE_EXTRA_ARGS`、`CLAUDE_CODE_SYSTEM_PROMPT` 通过环境变量传入 Ray runtime env；复杂多行内容建议谨慎转义。
- 如果 worker 上没有 `mcp` 包，preflight 会失败并给出 `claude_code_mcp_import_check.log`。

## 回滚方式

最小回滚可以直接运行旧 harness：

```bash
HARNESS_OPTION=camel-agent bash terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh
```

代码级回滚：

- 删除 `terminal-rl/agent/claude_code_agent.py`。
- 删除 `terminal-rl/agent/claude_code_mcp_server.py`。
- 删除 `terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_claude_code_pu.sh`。
- 删除 `terminal-rl/tests/test_claude_code_agent.py`。
- 回退 `terminal-rl/agent_runner.py`、`terminal-rl/generate.py`、两个 base/exploration shell 脚本和相关测试中的 `claude-code` 分支。

