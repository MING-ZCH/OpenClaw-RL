# Claude Code harness with Qwen/SGLang trainable backend

生成时间：2026-06-17 17:23:47 HKT

## 目标

本次更新把 `harness_option=claude_code` 的默认语义调整为：复用 Claude Code CLI 的 agent/harness 外壳，但底层 LLM 不调用 Anthropic Claude，而是通过本地 Anthropic Messages 兼容 gateway 调当前 rollout 的 Qwen/SGLang `/generate`。这样生成动作来自正在训练的 Qwen policy，并带有 SGLang 返回的 token ids/logprobs，可进入 GRPO/DAPO 训练。

## 关键改动

- `terminal-rl/agent/claude_code_qwen_gateway.py`：新增本地 gateway，支持 `/v1/messages`、`/v1/messages/count_tokens`、`/v1/models`。它把 Claude Code 的 Anthropic Messages 请求转换为 Qwen chat template + SGLang `/generate`，并记录 `output_token_ids`、`output_token_logprobs`。
- `terminal-rl/agent/claude_code_agent.py`：新增 `CLAUDE_CODE_LLM_BACKEND=sglang|anthropic`。默认 `sglang`；sglang 模式启动本地 gateway，把 Claude Code 子进程的 `ANTHROPIC_BASE_URL` 指向该 gateway，并用 gateway 记录构造 `Interaction`。
- `terminal-rl/generate.py`：`claude-code` 只有在外部 Anthropic 后端时默认 non-trainable；`sglang` 后端默认 trainable。
- `terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh`：新增 `CLAUDE_CODE_LLM_BACKEND`、`CLAUDE_CODE_QWEN_GATEWAY_MODEL`；`sglang` 后端默认 `CLAUDE_CODE_MARK_NON_TRAINABLE=0`。
- `terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_claude_code_pu.sh`：显式默认 `CLAUDE_CODE_LLM_BACKEND=sglang`、`CLAUDE_CODE_MARK_NON_TRAINABLE=0`。

## 数据流

1. rollout 选择 `harness_option=claude-code`。
2. `ClaudeCodeAgent` 启动 Claude Code CLI。
3. `ClaudeCodeAgent` 同时启动本地 gateway，例如 `http://127.0.0.1:<port>`。
4. Claude Code CLI 通过 `ANTHROPIC_BASE_URL` 把模型请求发到本地 gateway。
5. gateway 用当前 `SGLangTurnClient` 调 Qwen/SGLang `/generate`，请求包含 `return_logprob=True`。
6. gateway 把 Qwen 输出转换回 Anthropic Messages response 给 Claude Code CLI。
7. agent 从 gateway 记录中构造 `Interaction.output_token_ids` 和 `Interaction.output_token_logprobs`。
8. `_build_samples()` 写入 `Sample.rollout_log_probs`，进入现有 GRPO/DAPO loss。

## 运行示例

```bash
export CLAUDE_CODE_CLI=/root/.local/bin/claude
export CLAUDE_CODE_LLM_BACKEND=sglang
export CLAUDE_CODE_MARK_NON_TRAINABLE=0

bash terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_claude_code_pu.sh
```

sglang 模式不需要 Anthropic API key；Claude 子进程会被强制使用 dummy key 并指向本地 gateway。

## 注意事项

- 这不是训练 Claude；训练对象仍是当前 Qwen/SGLang policy。
- Claude Code CLI 的系统提示和 tool schema 是 Anthropic 风格，Qwen 是否能稳定产出 Claude Code 可执行的 tool_use 行为仍需要实跑观察。
- gateway 会尝试用现有 `tool_call_parser` 解析 Qwen 工具调用并转换为 Anthropic `tool_use` block。
- 如果切到 `CLAUDE_CODE_LLM_BACKEND=anthropic`，默认仍应保持 `CLAUDE_CODE_MARK_NON_TRAINABLE=1`。

## 验证

已执行：

```bash
python3 -m py_compile terminal-rl/agent/claude_code_agent.py terminal-rl/agent/claude_code_qwen_gateway.py terminal-rl/agent/claude_code_mcp_server.py terminal-rl/agent_runner.py terminal-rl/generate.py
bash -n terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh
bash -n terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_claude_code_pu.sh
timeout 90s /mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python -m pytest terminal-rl/tests/test_claude_code_agent.py terminal-rl/tests/test_agent_runner_harness_option.py terminal-rl/tests/test_harness_option_routing.py -v
MAX_CKPT_KEEP=0 DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_claude_code_pu.sh
```

结果：相关 pytest `13 passed`；dry-run 通过；带假 `ANTHROPIC_API_KEY=supersecret` 的 dry-run 日志未检出密钥串。

