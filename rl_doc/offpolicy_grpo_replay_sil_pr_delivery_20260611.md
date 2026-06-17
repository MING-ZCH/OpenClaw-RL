# Offpolicy-GRPO / Replay-buffer / SPEAR PR 交付说明

面向：leo

本 PR 范围仅覆盖 `terminal-rl` + `seta` 上的 `offpolicy-GRPO/DAPO`、`Replay buffer`、`PER`、`TOPR` 与 `SPEAR-style Self-Imitation Learning`。不包含 `Docker worker` 运维改造、`SWE-Smith` 数据适配、`world_model`、`Agent57` 或评测脚本改造。

## 1. 动机

`terminal-rl` 的 agentic rollout 成本高，纯 on-policy GRPO/DAPO 会在每次 policy update 后丢弃历史 trajectory。本 PR 的目标是把可复用历史样本纳入训练，同时用 `policy_version`、staleness-aware sampling、importance weighting 和 proximal policy correction 控制 off-policy 偏差。

核心思路：

- `Replay buffer` 保存历史 rollout group，并记录 `rollout_log_probs` 与 `policy_version`。
- `decoupled_policy_loss` 将训练 ratio 拆成 `pi_theta / pi_prox` 与 `pi_prox / pi_behav`，对 replay 样本做 bounded importance correction。
- `PER` 用 reward deviation 等可提前获得的信号做优先采样，并把 IS weight 写入 loss。
- `TOPR` 用 sequence-level importance weight 稳定长 response 的 off-policy 修正。
- `SPEAR/SIL` 只把高 reward trajectory 放入 `SILBuffer`，以小比例混入后续 batch 做 self-imitation。

## 2. 主要实现

| 路径 | 作用 |
| --- | --- |
| `slime/slime/rollout/data_source.py` | integrated `slime` 的 in-process replay buffer、staleness-aware sampling、SPEAR/SIL 入库 |
| `slime/slime/ray/rollout.py` | 标记 `policy_version`，输出 `per_is_weights`、`policy_versions`，混入 SIL 样本 |
| `slime/slime/backends/megatron_utils/loss.py` | 新增 `decoupled_policy_loss`，支持 PER IS weight、TOPR、M2PO、staleness metrics |
| `slime/slime/backends/megatron_utils/actor.py` | 训练前计算 `proximal_log_probs`，SIL 样本使用预计算 advantage |
| `slime/slime/backends/megatron_utils/model.py` | 训练 batch 透传 offpolicy 字段 |
| `slime/slime/utils/buffer_sampling_strategies.py` | FIFO/LIFO/random/PER/hybrid sampling strategy |
| `slime/slime/utils/per_buffer.py` | `Prioritized Experience Replay` 策略和 IS weight |
| `slime/slime/utils/offpolicy_utils.py` | importance weight、M2PO、staleness 工具函数 |
| `slime/slime/utils/proximal_logp_utils.py` | proximal logprob recompute/approximation helper |
| `slime/slime/utils/topr_utils.py` | TOPR sequence-level correction |
| `slime/slime/utils/sil_buffer.py` | SPEAR-style self-imitation trajectory buffer |
| `slime/train.py`、`slime/train_async.py` | 支持每个 rollout 后多次 replay train |
| `terminal-rl/terminal-rl_qwen3-8b_offpolicy_seta_pu.sh` | offpolicy wrapper，默认使用 integrated `slime` |
| `terminal-rl/scripts/run_offpolicy_seta_*_latest_20260617.sh` | 各算法一键启动入口 |
| `terminal-rl/scripts/run_offpolicy_pr_check_4gpu_no_spear_20260617.sh` | 4GPU PR smoke runner，默认跳过已单独验证的 SPEAR |
| `terminal-rl/scripts/run_offpolicy_per_prcheck_4gpu_20260618.sh` | PER 单项 4GPU smoke runner |

## 3. 启动方式

先设置 worker：

```bash
cd /path/to/OpenClaw-RL
export WORKER_URLS="http://<worker-a>:18081,http://<worker-b>:18081"
```

单算法启动：

```bash
bash terminal-rl/scripts/run_offpolicy_seta_baseline_latest_20260617.sh
bash terminal-rl/scripts/run_offpolicy_seta_dapo_latest_20260617.sh
bash terminal-rl/scripts/run_offpolicy_seta_per_latest_20260617.sh
bash terminal-rl/scripts/run_offpolicy_seta_topr_latest_20260617.sh
bash terminal-rl/scripts/run_offpolicy_seta_spear_latest_20260617.sh
bash terminal-rl/scripts/run_offpolicy_seta_all3_latest_20260617.sh
```

通用入口：

```bash
OFFPOLICY_MODE=dapo  bash terminal-rl/scripts/run_offpolicy_seta_onekey_latest_20260617.sh dapo
OFFPOLICY_MODE=per   bash terminal-rl/scripts/run_offpolicy_seta_onekey_latest_20260617.sh per
OFFPOLICY_MODE=topr  bash terminal-rl/scripts/run_offpolicy_seta_onekey_latest_20260617.sh topr
OFFPOLICY_MODE=spear bash terminal-rl/scripts/run_offpolicy_seta_onekey_latest_20260617.sh spear
```

4GPU PR smoke：

```bash
CHECK_MODES="baseline dapo per topr all3" \
MAX_TURN=3 NUM_ROLLOUT=4 ROLLOUT_BATCH_SIZE=4 N_SAMPLES=4 \
bash terminal-rl/scripts/run_offpolicy_pr_check_4gpu_no_spear_20260617.sh
```

PER 单项 smoke：

```bash
MAX_TURN=3 NUM_ROLLOUT=4 ROLLOUT_BATCH_SIZE=4 N_SAMPLES=4 \
bash terminal-rl/scripts/run_offpolicy_per_prcheck_4gpu_20260618.sh
```

## 4. 已验证结果

本地训练 pod 已完成以下 4GPU smoke 验证：

| 模式 | 结果 | 验收信号 |
| --- | --- | --- |
| `dapo` | 通过 | `Ray job succeeded`，4 次 `finish_rollout`，4 次 `actor_train`，出现 `importance_weight_*`、`mean_staleness` |
| `topr` | 通过 | `Ray job succeeded`，出现 `topr_w_seq_mean/max/min`、`topr_blend_lambda` |
| `spear` | 通过 | SPEAR/SIL 路径已单独验证，启动参数包含 `--enable-trajectory-replay` 并进入训练流程 |
| `per` | 通过 | `Ray job succeeded`，`buffer_sampling_strategy=per`，出现 `per_is_weight_mean/min/max` |

最近一次 PER 验证目录：

```text
runs/offpolicy_per_prcheck_4gpu_20260618_020720
```

该 run 的 `summary.log` 显示 `END mode=per status=0` 与 `DONE overall_status=0`。日志中出现过 `/reset` 的 `410 Gone / LEASE_EXPIRED` traceback，但训练继续并成功结束，归类为 remote env lease 生命周期噪声，不是 PER/offpolicy 算法错误。

## 5. 非范围说明

本 PR 不声明解决：

- `Docker worker` 容量、reset、lease 或网络波动问题；
- `SWE-Smith` 数据集/环境接入；
- `world_model`、`Agent57`、额外评测统计脚本；
- 上游集群私有 `WORKER_URLS` 默认值。

这些内容应在独立 PR 或实验分支中处理，避免污染本次 offpolicy replay-buffer / SPEAR 算法实现。
