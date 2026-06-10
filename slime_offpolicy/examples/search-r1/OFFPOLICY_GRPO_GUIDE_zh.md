# Off-Policy GRPO 实现指南

SLIME 中 off-policy 群体相对策略优化（GRPO）的实用指南，用于通过样本复用训练语言模型。

---

## 目录

1. [概述](#概述)
2. [核心架构](#核心架构)
3. [关键组件](#关键组件)
4. [安装与配置](#安装与配置)
5. [使用方法](#使用方法)
6. [配置指南](#配置指南)
7. [监控调试](#监控调试)

---

## 概述

### 什么是 Off-Policy GRPO？

Off-policy GRPO 通过允许**样本复用**来提升样本效率，可以使用前几个版本策略生成的数据进行训练。

**核心特性**：
- ✅ **样本复用**：每个样本可复用 2-8 次
- ✅ **陈旧度控制**：通过 `max_staleness` 配置
- ✅ **重要性采样**：修正分布偏移
- ✅ **M2PO 过滤**：移除高方差 token
- ✅ **灵活策略**：支持随机、LIFO、优先级、混合采样

### Off-Policy Loss 公式

**Decoupled PPO 目标函数**：
```
L = E[w · min(u·A, clip(u, 1-ε, 1+ε)·A)]

其中：
  w = π_prox / π_behav   (重要性权重，修正分布偏移)
  u = π_θ / π_prox       (近端比率，标准 PPO 组件)
  A = advantage（优势函数）
  ε = eps_clip (0.2)

关键策略：
  π_θ:      当前正在优化的策略
  π_prox:   前一个策略（正则化锚点）
  π_behav:  生成数据的行为策略（可能陈旧）
```

### On-Policy vs Off-Policy

| 方面 | On-Policy | Off-Policy |
|------|-----------|------------|
| **数据来源** | 仅当前策略 | 多个历史策略 |
| **样本复用** | ❌ 无 | ✅ 2-8 倍 |
| **Buffer** | ❌ 禁用 | ✅ 必需 |
| **Loss 类型** | `policy_loss` | `decoupled_policy_loss` |
| **重要性权重** | ❌ 不需要 | ✅ w = π_prox / π_behav |
| **陈旧度** | 0 | 可配置 (2-16) |

---

## 核心架构

### 数据流

```
1. ROLLOUT（生成） → 生成样本并标记 policy_version

2. BUFFER（存储） → 去重存储 (group_index, policy_version)

3. SAMPLING（采样） → 过滤 staleness ≤ max_staleness
                   → 过滤 reuse_count < buffer_reuse_samples
                   → 应用策略 (random/LIFO/priority/hybrid)

4. TRAINING（训练） → 计算重要性权重: w = π_prox / π_behav
                   → 应用 M2PO 过滤: m2 = (log π_behav - log π_prox)²
                   → 计算 loss: L = w · min(u·A, clip(u)·A)

5. UPDATE（更新） → 递增 policy_version
                 → 更新 proximal policy
```

### 策略版本追踪

**三层追踪机制**：
1. **RolloutManager**：全局 `current_policy_version`（训练后递增）
2. **DataSource**：追踪当前版本用于计算陈旧度
3. **Sample**：生成时标记 `policy_version`

**陈旧度计算**：`staleness = current_policy_version - sample.policy_version`

---

## 关键组件

### 1. Decoupled Policy Loss

**公式**：
```
L = E[w · min(u·A, clip(u, 1-ε, 1+ε)·A)]

其中：
  w = π_prox / π_behav   (重要性权重，裁剪至 [0.5, 2.0])
  u = π_θ / π_prox       (近端比率)
  A = advantage
```

**四层陈旧数据保护机制**：

| 层级 | 机制 | 参数 | 效果 |
|------|------|------|------|
| 1 | 陈旧度过滤 | `max_staleness=4` | 拒绝 `staleness > 4` 的样本 |
| 2 | 重要性权重裁剪 | `[0.5, 2.0]` | 限制权重范围 |
| 3 | M2PO 过滤 | `threshold=0.16` | 移除高方差 token |
| 4 | 极端值过滤 | `cap=5.0` | 排除 `w > 5.0` 的 token |

### 2. Buffer 管理

**结构**：`List[List[Sample]]`（外层：样本组，内层：每个 prompt 的多个 response）

**采样策略**：
- **Random**：均匀采样（默认，高多样性）
- **LIFO**：优先最新（最小陈旧度）
- **Priority**：按奖励排序（更快收敛）
- **Hybrid**：20% LIFO + 80% priority

### 3. M2PO 过滤

基于以下指标移除高方差 token：
```python
m2 = (log(π_behav) - log(π_prox))²
```

**阈值缩放规则**：`m2po_threshold = 0.04 × max_staleness`

---

## 安装与配置

### 环境

**H 集群 Docker 镜像**：
```bash
docker pull registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/lightrft:slime-20251118
```

### 数据准备（仅在从头配置环境时需要，使用上述镜像则不需要）

```bash
cd /mnt/shared-storage-user/puyuan/code/
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1/

WORK_DIR=/mnt/shared-storage-user/puyuan/code/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train

# 合并训练数据集
DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA

# 合并测试数据集
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA
```

### Retriever Server（在单独的终端中运行）

```bash
# 激活 retriever 环境
conda activate /mnt/shared-storage-user/puyuan/conda_envs/retriever

# 配置路径
save_path=/mnt/shared-storage-user/puyuan/code/Search-R1/Index
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path="/mnt/shared-storage-user/puyuan/model/e5-base-v2"

# 启动服务器（使用 GPU 0）
export CUDA_VISIBLE_DEVICES=0
python /mnt/shared-storage-user/puyuan/code/slime/examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu
```

---

## 使用方法

### 快速开始

```bash
# 退出 retriever 环境
conda deactivate

# 设置训练 GPU
export CUDA_VISIBLE_DEVICES=1,2
cd /mnt/shared-storage-user/puyuan/code/slime

# 运行训练（默认：offpolicy_random）
bash examples/search-r1/run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

### 训练模式

```bash
# 默认：Off-policy 随机采样
bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# 混合采样
MODE=offpolicy_hybrid bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# Vanilla 基线（无 off-policy 矫正）
MODE=offpolicy_vanilla bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# On-policy 基线（最稳定）
MODE=onpolicy bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

| 模式 | 陈旧度 | Loss | Buffer | 使用场景 |
|------|--------|------|--------|---------|
| **offpolicy_random** | 4 | decoupled | random | 标准基线 |
| **offpolicy_hybrid** | 4 | decoupled | hybrid | 生产环境（更快） |
| **offpolicy_vanilla** | 4 | policy | random | 消融实验 |
| **onpolicy** | 0 | policy | 禁用 | 稳定性参考 |

### 自定义参数

```bash
# 自定义陈旧度
MAX_STALENESS=8 M2PO_THRESHOLD=0.32 bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# 自定义 buffer
BUFFER_SIZE=2048 BUFFER_REUSE=8 bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# 组合多个参数
MODE=offpolicy_random MAX_STALENESS=8 BUFFER_STRATEGY=priority \
  bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

---

## 配置指南

### 关键参数

| 参数 | 保守 | 标准 | 激进 |
|------|------|------|------|
| `MAX_STALENESS` | 2 | 4 | 8 |
| `M2PO_THRESHOLD` | 0.08 | 0.16 | 0.32 |
| `BUFFER_SIZE` | 512 | 1024 | 2048 |
| `BUFFER_REUSE` | 2 | 4 | 8 |
| `IMP_WEIGHT_MIN` | 0.7 | 0.5 | 0.3 |
| `IMP_WEIGHT_MAX` | 1.5 | 2.0 | 3.0 |

**陈旧度缩放规则**：`M2PO_THRESHOLD = 0.04 × MAX_STALENESS`

### 推荐配置

**首次训练**：
```bash
# 使用默认配置
bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

**生产训练**：
```bash
MODE=offpolicy_hybrid bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

**高样本效率**：
```bash
MAX_STALENESS=8 M2PO_THRESHOLD=0.32 BUFFER_SIZE=2048 \
  bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

**最大稳定性**：
```bash
MAX_STALENESS=2 M2PO_THRESHOLD=0.08 IMP_WEIGHT_MAX=1.5 \
  bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

---

## 监控调试

### 关键指标（WandB）

**Buffer 健康度**：
- `buffer/avg_staleness`：应 ≤ `max_staleness`
- `buffer/utilization`：目标 0.7-0.9

**训练稳定性**：
- `train/importance_weight_mean`：应在 0.8-1.2
- `train/effective_sample_size`：应 > `batch_size × 0.5`
- `train/m2po_filter_rate`：应在 0.1-0.3
- `train/grad_norm`：应 < 5.0

### 健康检查

✅ **健康的训练**：
```
importance_weight_mean ∈ [0.8, 1.2]
effective_sample_size > batch_size × 0.5
m2po_filter_rate ∈ [0.1, 0.3]
grad_norm < 5.0
```

⚠️ **警告信号**：
```
importance_weight_mean > 2.0 或 < 0.5  → 降低陈旧度
effective_sample_size < batch_size × 0.3 → 收紧权重裁剪
m2po_filter_rate > 0.5 → 提高阈值
grad_norm > 10.0 → 训练不稳定，降低陈旧度
```

---

## 快速参考


### 样本生命周期

```
t=0: 生成样本 (policy_version=0)
t=1: 训练 (staleness=1, reuse_count=1)
t=2: 训练 (staleness=2, reuse_count=2)
t=3: 训练 (staleness=3, reuse_count=3)
t=4: 训练 (staleness=4, reuse_count=4) → 移除（达到复用上限）
```

### 重要性采样直觉理解

**问题**：使用旧策略数据（π_behav）训练，更新当前策略（π_θ）

**解决方案**：通过 `w = π_prox / π_behav` 加权样本
- `w > 1`：当前行为更可能 → 增加权重
- `w < 1`：当前行为不太可能 → 降低权重
- `w ≈ 1`：行为相似 → 保持原权重

**裁剪作用**：用偏差换取方差减小

---

## 参考资料

- **AReaL 论文**
- **Loss 实现**：`slime/backends/megatron_utils/loss.py`
- **Buffer 策略**：`slime/utils/buffer_sampling_strategies.py`
- **Off-Policy 工具**：`slime/utils/offpolicy_utils.py`

---

**脚本位置**：`/mnt/shared-storage-user/puyuan/code/slime/examples/search-r1/run_qwen3_4B_2xgpu_offpolicy_unified.sh`

**版本**：1.1（简化版）
**最后更新**：2026-03-06
