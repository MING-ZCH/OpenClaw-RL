# Agent57 for Agentic RL 适配说明

本文档记录当前 `terminal-rl` 中面向 agentic-RL 的 Agent57 / Agent57-Lite 适配实现。当前实现遵循向后兼容优先原则：所有新增能力默认关闭或保持 legacy 行为，只有显式配置后才改变探索策略。

## 1. 架构总览

当前实现把原版 Agent57 的三个核心思想拆成可逐步接入的模块：

| 原版 Agent57 模块 | 当前 terminal-rl 对应实现 | 当前状态 |
| --- | --- | --- |
| NGU（Never Give Up）双时间尺度内在奖励 | episode-local command signature novelty + run-level lifelong novelty | 已实现轻量版 |
| 多个 `(beta, gamma)` actor | K 个标量探索 arm，beta 调整 intrinsic reward 权重 | 已实现；尚未接 LoRA head |
| Meta-controller UCB bandit | `assign_group_arms()` 基于滑窗 UCB 选择 arm | 已实现 |
| RND lifelong novelty | action signature + coarse observation + exit_code 计数 | 已实现轻量替代 |
| Episodic memory KNN | `EpisodicMemoryBackend` 可插拔后端 | 已新增接口与 count/simhash 后端 |
| Retrace / n-step return | 暂未接入；当前训练仍走 GRPO/DAPO 标量 reward | Roadmap |
| Actor-Learner 异步架构 | 复用 slime / OpenClaw-RL 的 rollout worker + trainer 解耦 | 已复用基础设施 |

数据流：

```text
rollout sample group
  -> sglang_rollout.py: assign_group_arms()
  -> generate.py: execute terminal task and collect turn_records
  -> explore_agent57_lite.py:
       compute_lifelong_bonus()
       compute_ngu_lite_bonus()
       record_arm_event()
  -> rollout_log.py: structured metrics / wandb aggregation
  -> trainer: GRPO/DAPO reward and advantage computation
```

与原版 Agent57 的主要差异：

- 当前没有 value function、Q-learning、Retrace learner，也没有 `(beta, gamma)` 条件价值函数。
- 当前 K 个 arm 共享同一个语言模型策略，只通过 reward 权重和可选采样参数形成探索差异；LoRA head 仍是后续阶段。
- Lifelong novelty 使用稳定计数键替代 RND predictor，优先保证 terminal rollout 的低开销和可解释性。
- Episodic memory 新增为可插拔接口，但默认仍使用现有 command signature novelty，避免改变旧实验。

## 2. 核心组件

### 2.1 Intrinsic / Extrinsic Reward

入口文件：

- `terminal-rl/generate.py`
- `terminal-rl/explore_agent57_lite.py`

外在 reward 仍来自任务 verifier / safety reward / PRM 等既有路径。Agent57-Lite 额外提供：

- `compute_lifelong_bonus()`：跨 run 计数 novelty，按 arm beta 和 `EXPLORE_AGENT57_LIFELONG_COEF` 缩放。
- `compute_ngu_lite_bonus()`：NGU-lite product mode，使用 `episodic_novelty * lifelong_modulator`。
- `record_arm_event()`：将 arm 的 reward、success、parse error、truncation 等写入 local 或 sqlite backend，供 UCB 使用。

默认 `EXPLORE_AGENT57_COMBINE_MODE=add`，即 lifelong bonus 作为附加项；设置为 `ngu_lite` 后才使用乘法组合。

### 2.2 Episodic Memory

入口文件：

- `terminal-rl/agent57_episodic_memory.py`

统一接口：

```python
class EpisodicMemoryBackend:
    def add(self, state: Any) -> None: ...
    def compute_novelty(self, state: Any) -> float: ...
    def reset(self) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
```

三种 backend 的语义：

| Backend | 配置名 | 适用场景 | Novelty 语义 | 注意事项 |
| --- | --- | --- | --- | --- |
| Legacy signature novelty | `legacy` | 当前默认 terminal rollout | 每集内 command signature 的 `1/sqrt(count)` | 保持旧行为，不创建 memory 对象 |
| Count-based | `count` | 离散状态、可哈希 action/state | `1/sqrt(count + 1)` | 支持 decay、capacity、reset 清空策略 |
| SimHash-KNN | `simhash_knn` / `knn` | embedding 或连续向量状态 | SimHash 桶内 KNN 距离映射到 `[0, 1]` | 近似检索；bucket 为空时 novelty=1 |

切换示例：

```bash
EPISODIC_MEMORY_BACKEND=count \
EXPLORE_AGENT57_EPISODIC_CAPACITY=4096 \
EXPLORE_AGENT57_EPISODIC_COUNT_DECAY=0.99 \
bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

```bash
EPISODIC_MEMORY_BACKEND=simhash_knn \
EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS=64 \
EXPLORE_AGENT57_EPISODIC_BUCKET_CAPACITY=256 \
EXPLORE_AGENT57_EPISODIC_K=5 \
EXPLORE_AGENT57_EPISODIC_DISTANCE=cosine \
EXPLORE_AGENT57_EPISODIC_RANDOM_SEED=123 \
bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

当前默认仍是 `legacy`，所以不会自动替换 `_explore_episode_signature_novelty()`。后续若要让 Agent57 NGU-lite 使用新 memory，应在 rollout episode 初始化时创建 backend，并在每个 action/observation 后调用 `compute_novelty()` 与 `add()`。

### 2.3 Meta-controller UCB Bandit

入口文件：

- `terminal-rl/explore_agent57_lite.py`
- `slime/slime/rollout/sglang_rollout.py`

核心函数：

- `assign_group_arms(group_size, evaluation=False, dataset=None)`
- `record_arm_event(...)`
- `_ucb_scores(...)`

UCB score 当前支持四种 value：

- `legacy`：`success + 0.25 * base - 0.5 * parse_rate - 0.5 * trunc_rate`
- `success`：只看 success，并惩罚 parse/truncation
- `base`：使用原始 base score
- `normalized_base`：按数据集把 reward 归一到 `[0, 1]`

新增 `EXPLORE_AGENT57_UCB_RANDOM_SEED` 后，UCB 的随机性隔离到独立 `np.random.Generator`：

- 初始 arm tie-breaking
- 分数相同 arm 的随机排序
- epsilon exploration 的随机 arm 选择

未设置 seed 时保持旧行为，继续使用 Python 全局 `random`。

### 2.4 Retrace / n-step Return

当前没有实现 Agent57 原版 Retrace / n-step return。原因是 terminal-rl 当前训练主路径是 GRPO/DAPO：

- reward 是 trajectory/turn 聚合后的标量或少量分项；
- 没有显式 Q network / value head；
- 没有 replay buffer learner 读取 off-policy transition。

后续若接入 Retrace，需要先引入 value function 或 critic adapter，并定义 terminal step 粒度的 transition schema。

### 2.5 Actor-Learner 架构

当前复用 OpenClaw-RL / slime 的异步架构：

- rollout actor：`slime/slime/rollout/sglang_rollout.py`
- terminal execution / reward：`terminal-rl/generate.py`
- trainer：既有 GRPO/DAPO 训练脚本
- remote pool / watchdog：`terminal-rl/remote/`

Agent57-Lite 的 arm 选择发生在 rollout sample group 分配阶段；arm 结果通过 trajectory metadata 和 reward metrics 回传，供下一轮 UCB 更新。

## 3. 完整配置项清单

| 配置项 | 默认值 | 取值 | 作用 |
| --- | --- | --- | --- |
| `EXPLORE_AGENT57_LITE` | `0` | `0/1` | 打开 Agent57-Lite 总开关 |
| `EXPLORE_AGENT57_LITE_ENABLED` | 同 `EXPLORE_AGENT57_LITE` | `0/1` | runtime 生效开关 |
| `EXPLORE_AGENT57_K` | `8` | 正整数 | arm 数量 |
| `EXPLORE_AGENT57_ARM_BETAS` | `0,0.003,0.006,0.01,0.015,0.02,0.03,0.04` | 逗号分隔 float | 每个 arm 的 intrinsic beta |
| `EXPLORE_AGENT57_COMBINE_MODE` | `add` | `add/ngu_lite` | reward 组合方式 |
| `EXPLORE_AGENT57_NGU_MOD_CLIP` | `5.0` | `>=1` | NGU lifelong modulator 上限 |
| `EXPLORE_AGENT57_NGU_EPISODIC_SOURCE` | `signature_intrinsic` | `signature_intrinsic/intrinsic` | NGU episodic 来源 |
| `EXPLORE_AGENT57_MAX_BONUS` | `0` | `>=0` | Agent57 bonus 绝对值裁剪，0 表示不裁剪 |
| `EXPLORE_AGENT57_CONTROLLER` | `fixed` | `fixed/ucb` | arm 选择器 |
| `EXPLORE_AGENT57_UCB_C` | `0.5` | `>=0` | UCB exploration bonus 系数 |
| `EXPLORE_AGENT57_UCB_WINDOW` | `256` | 正整数 | 滑窗事件数 |
| `EXPLORE_AGENT57_UCB_EPSILON` | `0` | `[0,1]` | 强制随机探索概率 |
| `EXPLORE_AGENT57_UCB_MIN_PER_ARM` | `0` | `>=0` | 每个 arm 最少样本数，不足时优先探索 |
| `EXPLORE_AGENT57_UCB_VALUE` | `legacy` | `legacy/success/base/normalized_base` | UCB value 定义 |
| `EXPLORE_AGENT57_UCB_DATASET_AWARE` | `0` | `0/1` | 是否按数据集分别统计 UCB |
| `EXPLORE_AGENT57_UCB_RANDOM_SEED` | 空 | int | UCB 独立随机种子；空则 legacy 行为 |
| `EXPLORE_AGENT57_KEEP_BASELINE` | `1` | `0/1` | 每组 rollout 是否固定保留 arm 0 |
| `EXPLORE_AGENT57_LIFELONG` | `0` | `0/1` | lifelong novelty 开关 |
| `EXPLORE_AGENT57_LIFELONG_ENABLED` | 同 `EXPLORE_AGENT57_LIFELONG` | `0/1` | runtime 生效开关 |
| `EXPLORE_AGENT57_LIFELONG_COEF` | `0.01` | `>=0` | lifelong bonus 缩放 |
| `EXPLORE_AGENT57_LIFELONG_CLIP` | `2.0` | `>=0` | lifelong raw novelty 裁剪 |
| `EXPLORE_AGENT57_LIFELONG_WARMUP` | `64` | `>=0` | warmup 轨迹数 |
| `EXPLORE_AGENT57_LIFELONG_BACKEND` | `local` | `local/sqlite` | lifelong count 存储 |
| `EXPLORE_AGENT57_LIFELONG_KEY_VERSION` | `v1` | `v1/v2` | lifelong key schema |
| `EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET` | `1` | `0/1` | v2 key 是否包含 dataset |
| `EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK` | `0` | `0/1` | v2 key 是否包含 task bucket |
| `EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN` | `0` | `0/1` | v2 key 是否包含 turn bucket |
| `EXPLORE_AGENT57_STATE_PATH` | 自动推导 | path | sqlite state 文件 |
| `EXPLORE_AGENT57_SUCCESS_THRESHOLD` | `0.0` | float | UCB success 判定阈值 |
| `EPISODIC_MEMORY_BACKEND` | `legacy` | `legacy/count/simhash_knn/knn` | episodic memory backend 通用配置 |
| `EXPLORE_AGENT57_EPISODIC_BACKEND` | 同 `EPISODIC_MEMORY_BACKEND` | 同上 | Agent57 专用 episodic backend |
| `EXPLORE_AGENT57_EPISODIC_CAPACITY` | `4096` | `>=0` | count backend 容量；0 表示不限制 |
| `EXPLORE_AGENT57_EPISODIC_COUNT_DECAY` | `1.0` | `[0,1]` | count backend add/reset decay |
| `EXPLORE_AGENT57_EPISODIC_CLEAR_ON_RESET` | `1` | `0/1` | count backend reset 是否清空 |
| `EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS` | `64` | 正整数 | SimHash 位数 |
| `EXPLORE_AGENT57_EPISODIC_BUCKET_CAPACITY` | `256` | 正整数 | 每个 SimHash 桶的向量上限 |
| `EXPLORE_AGENT57_EPISODIC_K` | `5` | 正整数 | KNN 的 K |
| `EXPLORE_AGENT57_EPISODIC_DISTANCE` | `cosine` | `cosine/l2/hamming` | 桶内距离度量 |
| `EXPLORE_AGENT57_EPISODIC_VECTOR_DIM` | `128` | 正整数 | 非数值状态 hashing trick 维度 |
| `EXPLORE_AGENT57_EPISODIC_RANDOM_SEED` | 空 | int | SimHash hyperplane 随机种子 |

## 4. 使用指南

### 4.1 启用当前稳态 Agent57-Lite

```bash
EXPLORE_AGENT57_LITE=1 \
EXPLORE_AGENT57_LIFELONG=1 \
EXPLORE_AGENT57_LIFELONG_BACKEND=sqlite \
EXPLORE_AGENT57_CONTROLLER=fixed \
EXPLORE_AGENT57_ARM_BETAS="0,0.002,0.004,0.006,0.008,0.01,0.015,0.02" \
EXPLORE_AGENT57_LIFELONG_COEF=0.005 \
EXPLORE_AGENT57_LIFELONG_WARMUP=64 \
bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

### 4.2 启用 UCB 并固定复现性

```bash
EXPLORE_AGENT57_LITE=1 \
EXPLORE_AGENT57_LIFELONG=1 \
EXPLORE_AGENT57_LIFELONG_BACKEND=sqlite \
EXPLORE_AGENT57_CONTROLLER=ucb \
EXPLORE_AGENT57_UCB_VALUE=normalized_base \
EXPLORE_AGENT57_UCB_DATASET_AWARE=1 \
EXPLORE_AGENT57_UCB_RANDOM_SEED=20260604 \
bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

### 4.3 切换 episodic backend

当前 reward 主路径不会自动替换 legacy episodic novelty；以下配置用于后续 agent 初始化或实验性接入：

```bash
EXPLORE_AGENT57_EPISODIC_BACKEND=count
```

```bash
EXPLORE_AGENT57_EPISODIC_BACKEND=simhash_knn
EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS=64
EXPLORE_AGENT57_EPISODIC_DISTANCE=cosine
```

## 5. 已知限制

- SimHash-KNN 只在同一 hash bucket 内检索，不做多探针近邻搜索；高维向量下可能漏召回。
- Count-based backend 对连续 embedding 需要外部先做离散化或稳定签名，否则相近状态不会合并。
- UCB seed 只隔离 Agent57 UCB 自身随机性，不保证模型采样、Ray 调度、Docker 任务环境完全确定。
- 当前没有 LoRA head、多 value head、Retrace learner、prioritized replay。
- `knn` 当前是 `simhash_knn` 的兼容别名，不代表已有精确全量 KNN backend。

## 6. Roadmap

1. 将 `EpisodicMemoryBackend` 接入 rollout episode 生命周期，使 `ngu_lite` 可以选择 `legacy/count/simhash_knn` 作为 episodic source。
2. 引入 dataset/task aware episodic reset 策略，避免跨任务污染。
3. 增加多探针 SimHash 或轻量 ANN，提升 embedding KNN recall。
4. 接入 LoRA head 或 sampling profile head，实现真正 policy-space diversity。
5. 在 critic/value path 成熟后再实现 n-step / Retrace。

## 7. 关键代码入口

| 文件 | 作用 |
| --- | --- |
| `terminal-rl/agent57_episodic_memory.py` | episodic memory 抽象基类、count backend、SimHash-KNN backend、factory |
| `terminal-rl/explore_agent57_lite.py` | Agent57-Lite config、lifelong novelty、NGU-lite、UCB、arm event |
| `terminal-rl/generate.py` | terminal rollout reward 组合和 Agent57 metrics 写入 |
| `slime/slime/rollout/sglang_rollout.py` | sample group arm 分配和采样参数覆盖 |
| `terminal-rl/rollout_log.py` | Agent57 structured metrics / wandb 聚合 |
| `terminal-rl/tests/test_agent57_episodic_memory.py` | episodic backend 单元测试 |
| `terminal-rl/tests/test_explore_agent57_lite.py` | UCB / lifelong key / sqlite migration 测试 |

测试命令见 `terminal-rl/docs/agent57_test_commands.md`。
