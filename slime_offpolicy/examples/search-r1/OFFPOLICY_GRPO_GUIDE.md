# Off-Policy GRPO Implementation Guide

A practical guide to off-policy Group Relative Policy Optimization (GRPO) in SLIME for training language models with sample reuse.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Key Components](#key-components)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Monitoring](#monitoring)

---

## Overview

### What is Off-Policy GRPO?

Off-policy GRPO enables **sample reuse** across multiple policy updates, improving sample efficiency by training on data from previous policy versions.

**Key Features**:
- ✅ **Sample Reuse**: 2-8x reuse per sample
- ✅ **Staleness Control**: Configurable via `max_staleness`
- ✅ **Importance Sampling**: Corrects distribution shift
- ✅ **M2PO Filtering**: Removes high-variance tokens
- ✅ **Flexible Strategies**: Random, LIFO, priority, hybrid sampling

### Off-Policy Loss Formula

**Decoupled PPO Objective**:
```
L = E[w · min(u·A, clip(u, 1-ε, 1+ε)·A)]

where:
  w = π_prox / π_behav   (importance weight, corrects distribution shift)
  u = π_θ / π_prox       (proximal ratio, standard PPO component)
  A = advantage
  ε = eps_clip (0.2)

Key policies:
  π_θ:      Current policy being optimized
  π_prox:   Previous policy (regularization anchor)
  π_behav:  Behavior policy that generated the data (potentially stale)
```

### On-Policy vs Off-Policy

| Aspect | On-Policy | Off-Policy |
|--------|-----------|------------|
| **Data Source** | Current policy only | Multiple historical policies |
| **Sample Reuse** | ❌ None | ✅ 2-8x |
| **Buffer** | ❌ Disabled | ✅ Required |
| **Loss Type** | `policy_loss` | `decoupled_policy_loss` |
| **Importance Weight** | ❌ Not needed | ✅ w = π_prox / π_behav |
| **Staleness** | 0 | Configurable (2-16) |

---

## Core Architecture

### Data Flow

```
1. ROLLOUT → Generate samples with policy_version tag

2. BUFFER → Store with deduplication (group_index, policy_version)

3. SAMPLING → Filter by staleness ≤ max_staleness
            → Filter by reuse_count < buffer_reuse_samples
            → Apply strategy (random/LIFO/priority/hybrid)

4. TRAINING → Compute importance weights: w = π_prox / π_behav
            → Apply M2PO filtering on m2 = (log π_behav - log π_prox)²
            → Compute loss: L = w · min(u·A, clip(u)·A)

5. UPDATE → Increment policy_version
          → Update proximal policy
```

### Policy Version Tracking

**Three-level tracking**:
1. **RolloutManager**: Global `current_policy_version` (incremented after training)
2. **DataSource**: Tracks current version for staleness calculation
3. **Sample**: Tagged with `policy_version` at generation time

**Staleness**: `staleness = current_policy_version - sample.policy_version`

---

## Key Components

### 1. Decoupled Policy Loss

**Formula**:
```
L = E[w · min(u·A, clip(u, 1-ε, 1+ε)·A)]

where:
  w = π_prox / π_behav   (importance weight, clipped to [0.5, 2.0])
  u = π_θ / π_prox       (proximal ratio)
  A = advantage
```

**Four-layer protection against stale data**:

| Layer | Mechanism | Parameter | Effect |
|-------|-----------|-----------|--------|
| 1 | Staleness filter | `max_staleness=4` | Reject samples with `staleness > 4` |
| 2 | Importance clipping | `[0.5, 2.0]` | Limit weight range |
| 3 | M2PO filtering | `threshold=0.16` | Remove high-variance tokens |
| 4 | Extreme value cap | `cap=5.0` | Exclude extreme weights |

### 2. Buffer Management

**Structure**: `List[List[Sample]]` (outer: groups, inner: per-prompt responses)

**Sampling strategies**:
- **Random**: Uniform sampling (default, high diversity)
- **LIFO**: Newest first (minimal staleness)
- **Priority**: Sort by reward (faster convergence)
- **Hybrid**: 20% LIFO + 80% priority (recommended)

### 3. M2PO Filtering

Removes high-variance tokens based on:
```python
m2 = (log(π_behav) - log(π_prox))²
```

**Threshold scaling**: `m2po_threshold = 0.04 × max_staleness`

---

## Installation & Setup

### Environment

**H Cluster Docker image**:
```bash
docker pull registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/lightrft:slime-20251118
```

### Data Preparation (Only used in setting env from scrath, not nessesay  in above docker image)

```bash
cd /mnt/shared-storage-user/puyuan/code/
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1/

WORK_DIR=/mnt/shared-storage-user/puyuan/code/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train

DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA

DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA
```

### Retriever Server (in one separate terminal)

```bash
# Activate retriever environment
conda activate /mnt/shared-storage-user/puyuan/conda_envs/retriever

save_path=/mnt/shared-storage-user/puyuan/code/Search-R1/Index
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path="/mnt/shared-storage-user/puyuan/model/e5-base-v2"

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

## Usage

### Quick Start

```bash
# Exit retriever environment
conda deactivate

# Set GPUs for training
export CUDA_VISIBLE_DEVICES=1,2
cd /mnt/shared-storage-user/puyuan/code/slime

# Run training (default: offpolicy_random)
bash examples/search-r1/run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

### Training Modes

```bash
# Default: Off-policy with random sampling
bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# Hybrid sampling (production recommended)
MODE=offpolicy_hybrid bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# Vanilla baseline (no off-policy correction)
MODE=offpolicy_vanilla bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# On-policy baseline (most stable)
MODE=onpolicy bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

| Mode | Staleness | Loss | Buffer | Use Case |
|------|-----------|------|--------|----------|
| **offpolicy_random** | 4 | decoupled | random | Standard baseline |
| **offpolicy_hybrid** | 4 | decoupled | hybrid | Production (faster) |
| **offpolicy_vanilla** | 4 | policy | random | Ablation study |
| **onpolicy** | 0 | policy | disabled | Stability reference |

### Custom Parameters

```bash
# Custom staleness
MAX_STALENESS=8 M2PO_THRESHOLD=0.32 bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# Custom buffer
BUFFER_SIZE=2048 BUFFER_REUSE=8 bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# Combine overrides
MODE=offpolicy_random MAX_STALENESS=8 BUFFER_STRATEGY=priority \
  bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

---

## Configuration

### Key Parameters

| Parameter | Conservative | Standard | Aggressive |
|-----------|--------------|----------|------------|
| `MAX_STALENESS` | 2 | 4 | 8 |
| `M2PO_THRESHOLD` | 0.08 | 0.16 | 0.32 |
| `BUFFER_SIZE` | 512 | 1024 | 2048 |
| `BUFFER_REUSE` | 2 | 4 | 8 |
| `IMP_WEIGHT_MIN` | 0.7 | 0.5 | 0.3 |
| `IMP_WEIGHT_MAX` | 1.5 | 2.0 | 3.0 |

**Staleness scaling rule**: `M2PO_THRESHOLD = 0.04 × MAX_STALENESS`

### Recommended Configurations

**First-time training**:
```bash
# Use defaults
bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

**Production training**:
```bash
MODE=offpolicy_hybrid bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

**High sample efficiency**:
```bash
MAX_STALENESS=8 M2PO_THRESHOLD=0.32 BUFFER_SIZE=2048 \
  bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

**Maximum stability**:
```bash
MAX_STALENESS=2 M2PO_THRESHOLD=0.08 IMP_WEIGHT_MAX=1.5 \
  bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

---

## Monitoring

### Key Metrics (WandB)

**Buffer health**:
- `buffer/avg_staleness`: Should be ≤ `max_staleness`
- `buffer/utilization`: Target 0.7-0.9

**Training stability**:
- `train/importance_weight_mean`: Should be 0.8-1.2
- `train/effective_sample_size`: Should be > `batch_size × 0.5`
- `train/m2po_filter_rate`: Should be 0.1-0.3
- `train/grad_norm`: Should be < 5.0

### Health Checks

✅ **Healthy training**:
```
importance_weight_mean ∈ [0.8, 1.2]
effective_sample_size > batch_size × 0.5
m2po_filter_rate ∈ [0.1, 0.3]
grad_norm < 5.0
```

⚠️ **Warning signs**:
```
importance_weight_mean > 2.0 or < 0.5  → Reduce staleness
effective_sample_size < batch_size × 0.3 → Tighten weight clipping
m2po_filter_rate > 0.5 → Increase threshold
grad_norm > 10.0 → Training unstable, reduce staleness
```

### Quick Fixes

**Training unstable**:
```bash
MAX_STALENESS=2 IMP_WEIGHT_MAX=1.5 M2PO_THRESHOLD=0.32
```

**Low sample efficiency**:
```bash
MAX_STALENESS=8 BUFFER_SIZE=2048 BUFFER_STRATEGY=priority
```

**High variance**:
```bash
IMP_WEIGHT_MIN=0.7 IMP_WEIGHT_MAX=1.5 BEHAV_IMP_WEIGHT_CAP=3.0
```

---

## Quick Reference

### Common Commands

```bash
# Default
bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# Production
MODE=offpolicy_hybrid bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# Ablation
MODE=offpolicy_vanilla bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
MODE=onpolicy bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# Custom staleness
MAX_STALENESS=8 M2PO_THRESHOLD=0.32 bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
```

### Sample Lifecycle

```
t=0: Generate sample (policy_version=0)
t=1: Train (staleness=1, reuse_count=1)
t=2: Train (staleness=2, reuse_count=2)
t=3: Train (staleness=3, reuse_count=3)
t=4: Train (staleness=4, reuse_count=4) → Remove (reuse limit)
```

### Importance Sampling Intuition

**Problem**: Training on old policy data (π_behav), updating current policy (π_θ)

**Solution**: Weight samples by `w = π_prox / π_behav`
- `w > 1`: Current behavior more likely → Upweight
- `w < 1`: Current behavior less likely → Downweight
- `w ≈ 1`: Similar behaviors → Original weight

**Clipping**: Trade bias for variance reduction

---

## References

- **AReaL Paper**: Adaptive Reuse of Experience for LLM Policy Optimization
- **Loss Implementation**: `slime/backends/megatron_utils/loss.py`
- **Buffer Strategies**: `slime/utils/buffer_sampling_strategies.py`
- **Off-Policy Utils**: `slime/utils/offpolicy_utils.py`

---

**Script Location**: `/mnt/shared-storage-user/puyuan/code/slime/examples/search-r1/run_qwen3_4B_2xgpu_offpolicy_unified.sh`

**Version**: 1.1 (Simplified)
**Last Updated**: 2026-03-06
