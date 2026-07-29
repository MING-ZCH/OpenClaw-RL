# SWE-bench Verified 官方评测适配与使用说明

## 1. 范围与结论

本适配在现有 SWE-smith `Docker worker` 基础上增加
SWE-bench Verified 的标准评测链路，目标模型为 `Qwen/Qwen3-8B` 基模，
默认开启 thinking mode。

代码分支 `zch/sweverified-terminal-rl-eval` 最初从 SWE-smith PR #18 的
head `97518bc8` 切出，之后通过提交 `3780606d` 同步上游最新 harness。
因此 PR #22 依赖 SWE-smith 的通用 Docker lifecycle；合并顺序仍应为
SWE-smith 在前、SWE-bench Verified 在后。

SWE-bench Verified 是 **eval benchmark**，不是训练数据集。完整流程分为：

1. 固定版本的 500 条 Verified 数据转换与任务目录生成；
2. `terminal-rl` + Qwen3-8B 在隔离的 Docker workspace 中生成代码修改；
3. worker 导出官方 prediction schema 所需的 `model_patch`；
4. 使用固定版本的官方 `swebench.harness.run_evaluation` 计算最终分数。

worker 内部不会自行宣称 `resolved`，也不会用自定义测试结果替代官方分数。
只有第 4 步输出的官方 harness report 才是最终评测结果。

固定版本：

- dataset：`princeton-nlp/SWE-bench_Verified`
- split：`test`
- dataset revision：`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`
- instance 数：`500`
- SWE-bench：`4.1.0`
- harness commit：`f7bbbb2ccdf479001d6467c9e34af59e44a840f9`

官方参考：

- [SWE-bench evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/)
- [SWE-bench harness reference](https://www.swebench.com/SWE-bench/reference/harness/)
- [SWE-bench Verified dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)

### 1.1 “官方配置”的边界

这里区分三类配置：

- **SWE-bench 官方标准**：500 条 Verified test split、官方 prediction
  schema、官方 Docker image 与 `swebench.harness.run_evaluation`；
- **Qwen 官方推理建议**：thinking mode 下
  `temperature=0.6/top_p=0.95/top_k=20/min_p=0`，避免 greedy decoding；
- **Terminal-RL agent scaffold**：`camel-agent`、工具协议、最大 turn 和总
  token budget。SWE-bench 官方不规定 agent scaffold，因此这些参数以固定
  profile 记录，而不宣称为 SWE-bench 官方参数。

## 2. 关键文件

| 文件 | 用途 |
|---|---|
| `terminal-rl/data_utils/convert_sweverified_to_terminal_rl.py` | 固定版本数据转换、官方 image 映射、任务目录生成 |
| `terminal-rl/data_utils/download_sweverified.sh` | `smoke/full` 数据准备入口和 artifact lock |
| `terminal-rl/remote/run_pool_server_sweverified_pu.sh` | 端口 `18083` 的独立 worker 启动与 500 条只读预检 |
| `terminal-rl/scripts/smoke_swe_worker.py` | SWE-smith / SWE-Verified 通用 worker smoke client |
| `terminal-rl/terminal-rl_qwen3-8b_eval_pu.sh` | 通用 Qwen3-8B eval-only launcher |
| `terminal-rl/scripts/run_sweverified_qwen3_8b_base_think_eval.sh` | 4 卡 Qwen3-8B Verified 正式评测入口 |
| `terminal-rl/swebench_report.py` | 导出 `predictions.jsonl` 与 generation coverage，不计算官方分数 |
| `terminal-rl/scripts/run_swebench_verified_official_harness.sh` | 固定官方 commit 的最终评分入口 |

## 3. 数据准备

在共享文件系统中的 repo 根目录执行：

```bash
cd /path/to/OpenClaw-RL

MODE=full \
PYTHON_BIN="$(command -v python3)" \
bash terminal-rl/data_utils/download_sweverified.sh
```

正式转换强制从固定 Hugging Face revision 读取完整 500 条数据，不允许
`INPUT_JSONL` 或行数截断。预期输出：

```text
terminal-rl/dataset/sweverified_convert/test.jsonl
terminal-rl/dataset/sweverified_convert/official_test.jsonl
terminal-rl/dataset/sweverified_convert/convert_stats.json
terminal-rl/dataset/sweverified_env/<instance_id>/
```

`test.jsonl` 的预期 SHA256：

```text
4282529dbcc1b9253fa91da35b9f1768a2002b391cc90ac6a4e64575d59cfbf3
```

供官方 harness 评分的 `official_test.jsonl` 预期 SHA256：

```text
f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c
```

两个文件均由同一固定 Hugging Face revision 在同一转换事务中生成，并在
正式 generation preflight 中校验 ID 集合一致。正式 run 会把
`official_test.jsonl` 复制到 `runs/<run_id>/config/`，确保后续评分不再读取
Hugging Face 的可变默认 revision。

需要重建时显式增加 `OVERWRITE=1`。转换和 worker 共享
`.sweverified_artifact.lock`，避免 worker 读取发布中的 artifact。

## 4. 启动 Docker Worker

在有 Docker 的开发机执行：

```bash
cd /path/to/OpenClaw-RL

python3.12 -m venv .venv-swesmith-worker
.venv-swesmith-worker/bin/python -m pip install \
  -r terminal-rl/remote/requirements-swesmith-worker.txt

POOL_SERVER_PYTHON="$PWD/.venv-swesmith-worker/bin/python" \
ENV_SERVER_PORT=18083 \
WORKER_MAX_TASKS=4 \
WORKER_MAX_RUNS_PER_TASK=2 \
WORKER_MAX_CONCURRENT_BUILDS=1 \
WORKER_MAX_CONCURRENT_RESETS=4 \
WORKER_MAX_CONCURRENT_CLOSES=8 \
WORKER_MIN_DOCKER_FREE_GB=120 \
CONTAINER_MEMORY_LIMIT=16g \
CONTAINER_PIDS_LIMIT=256 \
bash terminal-rl/remote/run_pool_server_sweverified_pu.sh
```

该服务使用独立的 `TERMINAL_RL_POOL_NAMESPACE=sweverified` 和端口
`18083`。默认关闭 broad cleanup；SETA、SWE-smith 和 SWE-Verified 的
Compose start/down、watchdog network cleanup 通过同一个 host file lock
串行化，可与端口 `18081/18082` 的服务共存。

仅检查完整 500 条 artifact 而不启动服务：

```bash
WORKER_PREFLIGHT_ONLY=1 \
POOL_SERVER_PYTHON="$PWD/.venv-swesmith-worker/bin/python" \
bash terminal-rl/remote/run_pool_server_sweverified_pu.sh
```

`POOL_SERVER_PYTHON` 必须能真实导入
`terminal-rl.remote.pool_server`，并满足 pinned dependency 版本。不能仅因为
`fastapi` 可导入就复用系统 Python。受限网络环境应由使用者在执行命令前
配置可访问 GitHub 与 PyPI 的代理或 package index。

## 5. Worker Smoke

从 GPU 节点执行：

```bash
cd /path/to/OpenClaw-RL

python3 terminal-rl/scripts/smoke_swe_worker.py \
  --suite sweverified \
  --worker-url http://<docker-worker-host>:18083 \
  --index 0 \
  --ensure-image-timeout 3600 \
  --reset-session-timeout 900
```

成功条件包括：

- `/healthz`、`/allocate`、`/reset`、`/exec_tool`、`/evaluate`、`/close`
  全链路成功；
- agent 工作目录固定为 `/testbed`；
- `/evaluate` 返回 `grader=swebench_prediction_export`；
- 导出的 `model_patch` 包含 smoke probe；
- worker 不运行本地自定义 grader。

## 6. 4 卡 Qwen3-8B 正式生成

在 4 卡 H20/H200 GPU 节点执行：

```bash
cd /path/to/OpenClaw-RL

WORKER_URLS=http://<docker-worker-host>:18083 \
HF_CKPT=/path/to/Qwen3-8B \
REF_LOAD=/path/to/Qwen3-8B_torch_dist \
TRAIN_PYTHON="$(command -v python3)" \
WANDB_MODE=offline \
bash terminal-rl/scripts/run_sweverified_qwen3_8b_base_think_eval.sh
```

固定正式配置：

- 500 instances，`n_samples=1`；
- 4 张 rollout GPU；
- 2 个 `TP=2` SGLang engine；
- `EVAL_MAX_CONCURRENCY=4`；
- thinking mode；
- profile：`qwen3_official_think_64k_terminal_ref_v1`；
- Qwen 官方 sampling：
  `temperature=0.6`、`top_p=0.95`、`top_k=20`、`min_p=0`；
- `prompt=32768`、`response=32768`、总 context `65536`；
- Qwen 官方 YaRN 方案：`factor=2.0`、
  `original_max_position_embeddings=32768`；
- Terminal-RL scaffold：`camel-agent`、`MAX_TURN=200`、
  `max_total_tokens=65536`；
- worker 仅导出 patch，不在生成阶段计算 `resolved`。

使用 64K YaRN 的原因是 coding agent 的多轮工具历史可能超过 32K。Qwen
官方同时说明 static YaRN 可能影响短文本，因此该 profile 只用于需要长历史的
SWE-bench agent eval，不应自动套用到普通短文本评测。launcher 对以上参数
fail-closed，外部覆盖任一固定值都会退出。

生成阶段只有同时满足以下条件才返回成功：

- `submitted=500`；
- `incomplete=0`；
- `unexpected=0`；
- `technical_failures=0`；
- 500 个 instance 全部处于 `pending_official_grading`。

关键输出位于：

```text
runs/<run_id>/swebench_official/predictions.jsonl
runs/<run_id>/swebench_official/prediction_coverage.json
runs/<run_id>/swebench_official/instance_audit.json
runs/<run_id>/swebench_official/score_summary.json
```

其中 `score_summary.json` 的 `authoritative_score` 必须为 `null`，用于明确
表示官方评分尚未执行。

## 7. 官方 Harness 最终评分

在有 Docker 且能拉取官方 SWE-bench image 的机器执行：

```bash
cd /path/to/OpenClaw-RL

RUN_DIR=/path/to/OpenClaw-RL/runs/<run_id> \
MAX_WORKERS=4 \
EVAL_TIMEOUT=1800 \
bash terminal-rl/scripts/run_swebench_verified_official_harness.sh
```

脚本会：

1. checkout 固定 commit，并以 editable mode 安装 `swebench==4.1.0`；
2. 校验安装来源 commit；
3. 校验 `predictions.jsonl` 恰好包含 500 条、ID 唯一、字段严格为
   `instance_id/model_name_or_path/model_patch`；
4. 在启动 Docker 前 import 完整 harness，防止普通 wheel 漏打包
   `constants/fixtures`；
5. 使用 run 内固定且带 SHA256 的 `official_test.jsonl`，调用官方
   `swebench.harness.run_evaluation`；
6. 显式使用官方 `namespace=swebench`、`instance_image_tag=latest`。

官方 harness 本身以 `latest` 标识 instance image，未提供官方 digest
manifest。脚本会在 `evaluation_manifest.json` 记录 dataset、prediction
SHA256、harness commit 和 image tag，但不能把 upstream `latest` 描述为
内容不可变。

为避免官方 harness 复用旧 `report.json`，评分目录非空时脚本默认拒绝运行。
需要有意识地覆盖旧结果时，显式设置 `OVERWRITE_OFFICIAL_RESULTS=1`。

最终结果位于：

```text
runs/<run_id>/swebench_official/harness/
```

以该目录中的官方 report 统计 `resolved` 与 `resolved rate`。生成阶段的
`reward=0` 只是 deferred grading 占位值，不能当作模型得分。

## 8. 本地验证结果

针对当前代码已完成：

- 固定 Hugging Face revision 完整转换 `500/500`；
- task dir 生成 `500/500`，canonical JSONL SHA256 为
  `4282529dbcc1b9253fa91da35b9f1768a2002b391cc90ac6a4e64575d59cfbf3`；
- 官方评分 JSONL 生成 `500/500`，SHA256 为
  `f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c`；
- SWE-Verified、SWE-smith converter、Docker lifecycle、router 与 close
  lifecycle focused regression：`102 passed`；
- 当前分支完整 `terminal-rl/tests`（显式配置
  `TAU2_BENCH_ROOT`）：`176 passed`；
- shell `bash -n`、Python compile、`git diff --check`：通过；
- 正式 launcher `EVAL_DRY_RUN=1`：通过；实际校验
  Qwen3-8B revision `b968826d...`、模型 artifact manifest、500 条数据指纹和
  全部 runtime path，并输出固定 `64K` profile；
- tokenizer audit：500 条初始 prompt 最大约 `8406` token，1 条超过
  `8192`，0 条超过 `32768`；多轮工具历史由 64K total-token budget 约束；
- 多轮 trajectory 回归：只在最后一 turn 导出一个 `model_patch`，不会因
  同一 instance 的 turn samples 触发 duplicate prediction；
- 4 卡正式 topology 为 `2 x TP=2`，Megatron eval-only 占位 topology 为
  `world=1 / TP=1`，避免二者误复用导致 world-size assertion。

旧实验
`runs/eval_qwen3-8b_prerl_sweverified_fixed_2026-07-22_234611`
运行到约 `251/500`，证明 Qwen3-8B、SGLang、remote env 和 Docker workspace
主链路可持续工作，但没有生成完整 500 条 prediction，也没有官方 aggregate
report，因此不计为当前 commit 的完整 benchmark 结果。

完整任务的验收标准是：

1. generation 阶段导出完整 500 条 predictions 且无技术失败；
2. 固定官方 harness 完成 500 条评分；
3. 保存官方 aggregate report 和 instance logs。
