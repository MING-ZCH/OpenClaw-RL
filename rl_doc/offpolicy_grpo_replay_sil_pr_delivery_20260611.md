# Offpolicy-GRPO / Replay-buffer / Self-imitation Learning PR 交付说明

面向：leo
目标分支：`MING-ZCH/OpenClaw-RL:dev-agenticrl-safety-exploration-harness`
本次 PR 范围：仅覆盖任务 1、任务 2，即 `terminal-rl` + `seta` 的 offpolicy-GRPO / Replay-buffer / SPEAR-style self-imitation learning。`swe-smith` 适配属于任务 3，不混入本次 PR。

## 1. 动机

当前 `terminal-rl` 的 on-policy GRPO/DAPO 每轮样本生成成本高，并且 Docker worker 容量会显著限制吞吐。off-policy 方案的目标是把历史 rollout 轨迹以 Replay-buffer 形式复用，降低有效样本成本，同时通过 importance weighting、staleness 控制和 self-imitation learning 降低旧策略样本带来的偏差。

本次实现从第一性原理拆成三层：

1. **可复用历史样本**：把 rollout 生成的 `rollout_log_probs`、`policy_version`、reward、response 等保存到 in-process replay buffer。
2. **可校正 off-policy 偏差**：训练时使用 `decoupled_policy_loss`、importance weight clip、proximal logp recompute、M2PO filtering、version staleness metrics。
3. **可做高价值轨迹自模仿**：SPEAR/SIL 模式只把高 reward 历史轨迹放入 `SILBuffer`，训练 batch 中用渐进系数混入 self-imitation 样本。

## 2. 代码结构

新增/调整的关键文件：

| 路径 | 作用 |
| --- | --- |
| `terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh` | 新增 offpolicy 训练入口，支持 `OFFPOLICY_MODE=dapo/per/topr/spear/all3`，兼容旧变量 `OFFPOLICY_V2_MODE` |
| `slime_offpolicy/train_async.py` | off-policy 训练入口，与 wrapper 的 `SLIME_DIR` 对接 |
| `slime_offpolicy/slime/utils/per_buffer.py` | Replay-buffer / PER 数据结构 |
| `slime_offpolicy/slime/utils/buffer_sampling_strategies.py` | replay sampling 策略，包括 random / PER 等 |
| `slime_offpolicy/slime/utils/offpolicy_utils.py` | importance weighting、staleness、policy version 等工具 |
| `slime_offpolicy/slime/utils/proximal_logp_utils.py` | proximal logp recompute / approximation metric |
| `slime_offpolicy/slime/utils/topr_utils.py` | TOPR 权重修正 |
| `slime_offpolicy/slime/utils/sil_buffer.py` | SPEAR-style self-imitation buffer |
| `slime_offpolicy/slime/ray/rollout_data_source.py` | 生成样本入 replay/SIL buffer，过滤不完整 `rollout_log_probs` |
| `slime_offpolicy/slime/ray/rollout.py` | 从 replay/SIL buffer 取样，SIL 样本替换 batch tail，保持固定 global batch size |
| `slime_offpolicy/slime/backends/megatron_utils/actor.py` | SIL 样本 precomputed advantage override，避免被主 batch whitening 稀释 |
| `slime_offpolicy/slime/utils/data.py` | 训练数据传递 `sil_sample_flags`、`sil_precomputed_advantages` |
| `slime_offpolicy/slime/utils/types.py` | 与 harness 当前 `Sample` 类型对齐，并保留 `policy_version` |

## 3. 启动方式

基础 offpolicy-DAPO：

```bash
OFFPOLICY_MODE=dapo \
ALGO=dapo \
DATASET=seta \
MAX_CKPT_KEEP=2 \
SETA_SAFETY=clawsentry \
SAFETY_REWARD_COEF=0.3 \
CUSTOM_CONFIG_PATH=/mnt/shared-storage-user/puyuan/zhangchenhao/OpenClaw-RL/terminal-rl/configs/rollout_qwen3_think.yaml \
MAX_TURN=10 \
bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
```

PER：

```bash
OFFPOLICY_MODE=per ALGO=dapo DATASET=seta bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
```

TOPR：

```bash
OFFPOLICY_MODE=topr ALGO=dapo DATASET=seta bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
```

SPEAR / self-imitation learning：

```bash
OFFPOLICY_MODE=spear \
ALGO=dapo \
DATASET=seta \
OFFPOLICY_SPEAR_BUF=2048 \
OFFPOLICY_SPEAR_THRESH=1.0 \
OFFPOLICY_SPEAR_COEF=0.001 \
OFFPOLICY_SPEAR_STEPS=200 \
OFFPOLICY_SPEAR_DECAY=-1.0 \
bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
```

组合 DAPO + PER + TOPR：

```bash
OFFPOLICY_MODE=all3 ALGO=dapo DATASET=seta bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
```

## 4. 0609-0611 本地训练 case 结论

| 日志目录 | 配置 | 代码结论 | 非代码结论 |
| --- | --- | --- | --- |
| `tmp_doc_2026-06-09_012056` | `OFFPOLICY_MODE=dapo` + `DATASET=seta` | 早期存在 `Sample.Status.FAILED` 缺失，导致异常分支 `AttributeError`；当前 harness 的 `slime/slime/utils/types.py` 已有 `FAILED`，本 PR 的 `slime_offpolicy` 也保留 `FAILED` | 后续仍遇到 `TASK_SLOTS_EXHAUSTED` |
| `tmp_doc_2026-06-09_015638` | `OFFPOLICY_MODE=dapo` + `DATASET=seta` | 未见新的算法参数解析错误 | worker reset/allocate 多次失败，属于 Docker worker 容量/状态问题 |
| `tmp_doc_2026-06-09_022157` | `OFFPOLICY_MODE=dapo` + `DATASET=seta` | 出现过训练侧 `NoneType.inner`，根因是 rollout data ref 在 worker 压力/异常返回下变成 `None`；本 PR 保留数据源侧 filtering，并在文档中记录需继续观察 | 大量 `TASK_SLOTS_EXHAUSTED`、`evaluate` 失败 |
| `tmp_doc_2026-06-09_024533` | `DATASET=swesmith` | 属于任务 3，不进入本 PR | 主要失败为 worker reset/allocate 达到重试上限 |
| `tmp_doc_2026-06-10_015921` | `OFFPOLICY_MODE=spear` + `DATASET=seta` | SPEAR 参数成功进入训练命令；老版本可见 `SIL Buffer initialized`、`SIL Push/Mixed`，说明 self-imitation 路径已打通 | 最后被 worker capacity 限制打断 |
| `tmp_doc_2026-06-11_003304` | `OFFPOLICY_MODE=spear` + `DATASET=seta` | 新代码已包含 SIL batch-tail replacement、precomputed advantage override、无效 `rollout_log_probs` 过滤；未见算法参数解析错误 | 最后为 `ALL_WORKERS_UNAVAILABLE_OR_PRESSURED` / `TASK_SLOTS_EXHAUSTED` |

## 5. 已解决的关键问题

1. **`OFFPOLICY_V2_MODE` 命名兼容**
   新 wrapper 统一使用 `OFFPOLICY_MODE`，但继续接受 `OFFPOLICY_V2_MODE`，避免旧文档命令失效。

2. **Replay/SIL 与固定 batch size 冲突**
   早期 append SIL 样本会改变 Megatron 侧 batch shape。当前实现改为替换 batch tail，保持 global batch size 不变。

3. **SIL advantage 被 whitening 稀释**
   SPEAR 的 self-imitation 样本使用预计算 advantage，并在 actor 侧按 `sil_sample_flags` 覆盖对应样本 advantage。

4. **不完整 rollout logprob 进入 replay**
   `rollout_log_probs is None` 或长度与 `response_length` 不一致的样本不进入 SIL buffer，避免训练时 shape/logprob 对不齐。

5. **harness `Sample` 类型兼容**
   `slime_offpolicy/slime/utils/types.py` 对齐当前 harness 的 `Sample` 字段，同时增加 `policy_version`，避免 `terminal-rl/generate.py` 访问新字段时报错。

## 6. 当前限制

- 训练 pod 的失败大多来自 Docker worker pool 容量不足：`Worker at task capacity: 16/16`、`ALL_WORKERS_UNAVAILABLE_OR_PRESSURED`、`Max retries reached`。这不是 offpolicy/SIL 算法代码问题。
- 本 PR 不包含 `swe-smith` 数据集/环境适配。需要在任务 3 的 PR 中单独进入，避免 review 面过大。
- `OFFPOLICY_MODE=all3` 是 DAPO + PER + TOPR，不包含 SPEAR；SPEAR/SIL 单独跑更利于观察 replay auxiliary loss。

## 7. 本地验证清单

本次 PR 已执行：

```bash
bash -n terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
bash -n terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh
bash -n terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh
python -m py_compile \
  slime_offpolicy/train_async.py \
  slime_offpolicy/slime/utils/types.py \
  slime_offpolicy/slime/ray/rollout.py \
  slime_offpolicy/slime/ray/rollout_data_source.py \
  slime_offpolicy/slime/backends/megatron_utils/actor.py \
  slime_offpolicy/slime/utils/data.py \
  slime_offpolicy/slime/utils/per_buffer.py \
  slime_offpolicy/slime/utils/sil_buffer.py \
  slime_offpolicy/slime/utils/offpolicy_utils.py \
  slime_offpolicy/slime/utils/topr_utils.py
DRY_RUN=1 OFFPOLICY_MODE=spear ALGO=dapo DATASET=seta \
  bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
DRY_RUN=1 OFFPOLICY_MODE=per ALGO=dapo DATASET=seta \
  bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
DRY_RUN=1 OFFPOLICY_MODE=dapo ALGO=dapo DATASET=seta \
  bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
DRY_RUN=1 OFFPOLICY_MODE=topr ALGO=dapo DATASET=seta \
  bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
DRY_RUN=1 OFFPOLICY_MODE=all3 ALGO=dapo DATASET=seta \
  bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
PYTHONPATH="$PWD/slime_offpolicy:$PWD" python -m pytest \
  slime_offpolicy/tests/test_dynamic_sampling_filters.py -q
git diff --check
```

结果：

- `bash -n` 通过。
- `py_compile` 通过。
- `DRY_RUN` 覆盖 `dapo/per/topr/spear/all3/none`，确认训练命令指向 `slime_offpolicy/train_async.py`，并按 mode 注入对应 flags。
- `test_dynamic_sampling_filters.py` 通过：`3 passed, 2 warnings`。
- `test_http_buffer_integration.py` 当前 `pytest` 收集到 `0 items`，因此不作为本 PR gate。
- `git diff --check` 通过。

建议 review 时至少复查：

```bash
bash -n terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
python -m py_compile \
  slime_offpolicy/train_async.py \
  slime_offpolicy/slime/utils/types.py \
  slime_offpolicy/slime/ray/rollout.py \
  slime_offpolicy/slime/ray/rollout_data_source.py \
  slime_offpolicy/slime/backends/megatron_utils/actor.py \
  slime_offpolicy/slime/utils/data.py
DRY_RUN=1 OFFPOLICY_MODE=spear ALGO=dapo DATASET=seta \
  bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
```

预期 dry-run 中应出现：

- `SLIME_DIR=.../slime_offpolicy`
- `train_async.py` 来自 `slime_offpolicy`
- `--loss-type decoupled_policy_loss`
- `--enable-proximal-policy-storage`
- `--enable-trajectory-replay`（仅 `OFFPOLICY_MODE=spear`）
- `--buffer-sampling-strategy per`（仅 `OFFPOLICY_MODE=per/all3`）
- `--use-topr`（仅 `OFFPOLICY_MODE=topr/all3`）

## 8. Review 重点

- 训练入口保持为 wrapper，不覆盖 harness 现有稳定 baseline。
- offpolicy 后端隔离在 `slime_offpolicy`，默认 on-policy 训练不受影响。
- 所有新增算法开关默认关闭，只有 wrapper 设置 `SLIME_DIR` 和 `EXTRA_ALGO_ARGS` 后才启用。
- 代码侧已处理 0609-0611 发现的真实兼容问题；剩余训练中断主要依赖 Docker worker pool 扩容/清理。
