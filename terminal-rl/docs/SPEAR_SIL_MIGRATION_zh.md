# SPEAR / SIL Self-Imitation Learning 适配说明

面向：leo

本说明只覆盖本 PR 中的 `SPEAR-style Self-Imitation Learning` 适配。当前实现已合并到 integrated `slime`，不再依赖独立 `slime_offpolicy` 目录。

## 1. 目标

SPEAR 的核心思想是 “先探索，再信任高分胜利轨迹”。在 `terminal-rl` 场景中，真实环境 rollout 昂贵，因此高 reward trajectory 不应只用一次。本实现把高分样本写入 `SILBuffer`，后续训练时从 buffer 中抽取少量样本混入 batch，用渐进系数做 self-imitation。

## 2. 代码路径

| 路径 | 职责 |
| --- | --- |
| `slime/slime/utils/sil_buffer.py` | 固定容量 `SILBuffer`，按 reward threshold 接收高分 trajectory |
| `slime/slime/rollout/data_source.py` | rollout 样本入 replay buffer 时同步尝试写入 `SILBuffer` |
| `slime/slime/ray/rollout.py` | 从 `SILBuffer` 采样并替换 batch tail，写入 `sil_sample_flags` 与 `sil_precomputed_advantages` |
| `slime/slime/backends/megatron_utils/actor.py` | 训练前用 `sil_precomputed_advantages` 覆盖 SIL 样本 advantage |
| `terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh` | `OFFPOLICY_MODE=spear` 时注入 SPEAR/SIL 参数 |

## 3. 参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--enable-trajectory-replay` | off | 启用 SPEAR/SIL trajectory replay |
| `--trajectory-buffer-size` | 2048 | SIL buffer 容量 |
| `--trajectory-score-threshold` | 1.0 | 入库 reward 阈值 |
| `--replay-loss-coef` | 0.001 | 最终 self-imitation loss 系数 |
| `--max-replay-loss-steps` | 200 | loss 系数 warmup 步数 |
| `--weight-decay-trajectory-replay` | -1.0 | SIL advantage 重估模式 |
| `--enable-trajectory-posadv` | off | 仅保留正 advantage trajectory |

## 4. 启动命令

```bash
cd /path/to/OpenClaw-RL
export WORKER_URLS="http://<worker-a>:18081,http://<worker-b>:18081"

OFFPOLICY_MODE=spear \
ALGO=dapo \
DATASET=seta \
MAX_TURN=10 \
MAX_CKPT_KEEP=0 \
bash terminal-rl/scripts/run_offpolicy_seta_spear_latest_20260617.sh
```

也可以直接调用 wrapper：

```bash
OFFPOLICY_MODE=spear \
OFFPOLICY_SPEAR_THRESH=1.0 \
OFFPOLICY_SPEAR_BUF=2048 \
OFFPOLICY_SPEAR_COEF=0.001 \
OFFPOLICY_SPEAR_STEPS=200 \
OFFPOLICY_SPEAR_DECAY=-1.0 \
bash terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh
```

## 5. 验收信号

训练日志中应至少看到：

```text
SPEAR SIL buffer enabled
--enable-trajectory-replay
--trajectory-buffer-size
--replay-loss-coef
Ray job succeeded
```

当 buffer 中已有满足阈值的 trajectory 后，应继续观察：

```text
Mixed ... SIL samples into train batch
Overrode advantages for ... SIL samples
```

如果只出现 `TASK_SLOTS_EXHAUSTED`、`LEASE_EXPIRED`、`/allocate` 或 `/reset` 相关错误，应先按 remote env worker 容量和 lease 生命周期排查；这类错误不代表 SPEAR/SIL 算法路径未接入。
