## 结论

`terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh` 是 a3s-code + DAPO SetA baseline wrapper，默认只覆盖 harness/data/algo/DAPO 关键变量，其余训练逻辑复用 `terminal-rl_qwen3-8b_pu.sh`，减少脚本漂移。

## 使用方式

```bash
# 只检查最终 train_async 命令
DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh

# 正式运行示例
WORKER_URLS=http://<worker-ip>:18081 \
bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh
```

## 关键默认值

| 变量 | 默认 | 位置 / 说明 |
|---|---|---|
| `HARNESS_OPTION` | `a3s-code` | `terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh:16` |
| `DATASET` | `seta` | `terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh:17` |
| `ALGO` | `dapo` | `terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh:18` |
| `SETA_SAFETY` | `clawsentry` | SetA + safety reward baseline |
| `SAFETY_REWARD_COEF` | `0.3` | 与现有 SetA safety baseline 对齐 |
| `MAX_TURN` | `10` | 与现有 DAPO SetA run tag 对齐 |
| `DAPO_EPS_CLIP_LOW/HIGH` | `0.2 / 0.28` | DAPO clipping |
| `DAPO_CALCULATE_PER_TOKEN_LOSS` | `1` | token-level loss |
| `DAPO_DYNAMIC_SAMPLING` | `1` | dynamic sampling |
| `A3S_CODE_REPO_ROOT` | `/mnt/shared-storage-user/puyuan/code/a3s-lab/Code` | `terminal-rl/terminal-rl_qwen3-8b_pu.sh:274` |
| `A3S_CODE_CACHE_DIR` | `/mnt/shared-storage-user/puyuan/.cache/a3s-code-cp312-x86_64` | shared native cache; bootstrap appends `/3.3.0` |
| `A3S_CODE_TURN_TIMEOUT_SEC` | `900` | SDK outer turn timeout |
| `A3S_CODE_TOOL_TIMEOUT_MS` | `7200000` | SDK external tool timeout |
| `A3S_CODE_MAX_TOOL_ROUNDS` | `10` | SDK inner tool rounds |
| `A3S_CODE_OUTPUT_TOKENS` | `8192` | bridge-generated config output limit |

## Pipeline 兼容性

| 模块 | 兼容性结论 |
|---|---|
| `router_server` | SetA 仍会 reset/evaluate terminal env；a3s SDK external queue 通过 `env_client.exec_tool(lease_id, ...)` 进入同一 lease |
| SGLang | SDK model call 走本地 OpenAI-compatible bridge，再调用 terminal-rl `SGLangTurnClient` |
| ClawSentry | reward-level 流程保留；pre-action 不能在 SDK 内部 action 前拦截，但 trajectory 中会保留 SDK tool calls |
| PRM | `generate.py` 在无外部 `tool_call_requests` 时用 `sdk_tool_calls` 作为 PRM tool-call 输入 |
| trajectory 保存 | `generate.py:1596` 写入 `harness_option`、`sdk_model_turns`、`sdk_tool_calls`，旧 `tool_calls` 字段也回填 |
| lease 回收 | `generate.py:2104` 先 `agent_runner.close()`，再 `env_client.close(lease_id)`，释放顺序明确 |

## 推荐命令

```bash
# a3s-code DAPO SetA baseline
HARNESS_OPTION=a3s-code \
DATASET=seta \
ALGO=dapo \
WORKER_URLS=http://<worker-ip>:18081 \
bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh

# 覆盖 SDK 来源
A3S_CODE_REPO_ROOT=/path/to/a3s-lab/Code \
A3S_CODE_PIP_PACKAGE=a3s-code==3.3.0 \
bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh
```

## 离线 GPU Worker

`a3s-code==3.3.0` 的 PyPI 包会在首次 `import a3s_code` 时下载 native wheel。GPU worker 无网时，先在能联网且共享同一 `/mnt/shared-storage-user` 的 CPU 节点预热 cache：

```bash
export TRAIN_PYTHON=/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python
export A3S_CODE_CACHE_DIR=/mnt/shared-storage-user/puyuan/.cache/a3s-code-cp312-x86_64

$TRAIN_PYTHON -m pip install a3s-code==3.3.0
A3S_CODE_CACHE_DIR=$A3S_CODE_CACHE_DIR $TRAIN_PYTHON -c "import a3s_code"
find "$A3S_CODE_CACHE_DIR/3.3.0" -name '_native.*' -maxdepth 1 -print
```

GPU worker 使用同一个 cache 即可离线 import：

```bash
export A3S_CODE_CACHE_DIR=/mnt/shared-storage-user/puyuan/.cache/a3s-code-cp312-x86_64
A3S_CODE_CACHE_DIR=$A3S_CODE_CACHE_DIR python3 -c "import a3s_code"
```

## 运行前检查

| 检查项 | 命令 |
|---|---|
| wrapper dry-run | `DRY_RUN=1 bash terminal-rl/terminal-rl_qwen3-8b_a3s_pu.sh` |
| SDK import | `${TRAIN_PYTHON:-python3} -c "import a3s_code"` |
| native cache | `find "$A3S_CODE_CACHE_DIR/3.3.0" -name '_native.*' -maxdepth 1 -print` |
| pool server | `curl http://<worker-ip>:18081/healthz` |
| harness tests | `/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin/python -m pytest terminal-rl/tests/test_a3s_code_agent.py terminal-rl/tests/test_agent_runner_harness_option.py terminal-rl/tests/test_harness_option_routing.py -v` |
