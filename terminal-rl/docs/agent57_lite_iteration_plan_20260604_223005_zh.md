# Agent57-Lite 工作交接文档

## 项目概览

当前实现是在 Terminal-RL rollout/reward 链路中加入 Agent57/NGU 风格的轻量探索：用按 arm 分配的探索权重和采样参数增加组内多样性，用 episodic signature novelty 与 lifelong count novelty 形成 intrinsic reward，并可用 UCB 在 arms 间做简单选择；代码已进入可跑实验阶段，但主要瓶颈仍是 env allocate/reset/evaluate/close 稳定性，尚未证明训练收益。

## 核心原理摘要

- Intrinsic reward：主路径仍以 terminal action/tool signature 的 episode novelty 为 episodic 项；lifelong novelty 用本地或 sqlite 计数近似全局新颖性。
- NGU-lite：`add` 模式保留各探索奖励相加；`ngu_lite` 模式已实现 `episodic * lifelong_modulator` 的乘法 bonus，并避免把 standalone intrinsic 重复计入 total。
- Meta-controller：arms 当前不是独立策略，只是不同 beta 和可选 sampling params；controller 支持 fixed round-robin 与 sqlite/local sliding-window UCB。
- 与完整 Agent57 的边界：没有 replay/retrace、RND predictor、Universal Value Function、gamma-conditioned policy/value heads 或 LoRA heads；当前是 reward/sampling 层的轻量近似。

## 已实现的功能清单

| 功能 | 当前说明 | 代码位置 | 与原计划差异 |
| --- | --- | --- | --- |
| Agent57 配置入口 | 从 env 解析 arm 数、betas、combine mode、UCB、lifelong、episodic backend 等参数。 | `terminal-rl/explore_agent57_lite.py:98-227` `Agent57LiteConfig` / `config_from_env()` | 原文把 `ngu_lite`、UCB epsilon/min/value 等写成待实现；当前已落地。 |
| sqlite/local lifelong state | 默认可按 `RUN_DIR` 推导 sqlite 路径，维护 `lifelong_counts`、`meta`、`arm_events`，并做 schema migration。 | `terminal-rl/explore_agent57_lite.py:75-94` `_default_state_path()`；`terminal-rl/explore_agent57_lite.py:253-305` `_connect()` / `_sqlite_next_counts()` | 原文只把 dataset-aware arm events 当未来扩展；当前 schema 已有 `dataset`、`normalized_base_score`。 |
| Lifelong key v1/v2 | v1 使用 action/obs/exit；v2 可包含 dataset/split/task/turn、command family、test/filemod flags、signature、obs、exit。 | `terminal-rl/explore_agent57_lite.py:499-665` `V1LifelongKeyBuilder` / `V2LifelongKeyBuilder` / `lifelong_keys()` | 原文计划的 v2 key 已实现；task/turn 仍为 opt-in。 |
| Lifelong bonus | 更新 counts，计算 `mean(1/sqrt(count+1))`，按 arm beta 与 coef 缩放，并按 status、parse error、warmup、state error 等原因抑制。 | `terminal-rl/explore_agent57_lite.py:673-771` `compute_lifelong_bonus()` | counts 在 eligibility suppression 前更新，仍会让 bad attempts 消耗 novelty；这是当前实际行为。 |
| NGU-lite product bonus | 在 `combine_mode=ngu_lite` 且 lifelong eligible 时计算乘法 bonus，支持 `EXPLORE_AGENT57_MAX_BONUS` cap。 | `terminal-rl/explore_agent57_lite.py:774-815` `compute_ngu_lite_bonus()`；`terminal-rl/generate.py:2477-2564` reward 集成 | 原文把 product mode 和 max bonus cap 写成下一步；当前已实现并接入主 reward block。 |
| Episode signature novelty | 用已有 turn/action signature novelty 作为 episodic source；`ngu_lite` 下用 `_explore_episode_signature_novelty()`。 | `terminal-rl/generate.py:272-316` `_explore_intrinsic_bonus()` / `_explore_episode_signature_novelty()` | 未使用神经 embedding/k-NN；这是当前主路径。 |
| Arm 分配与 UCB | 支持 fixed round-robin、UCB、baseline arm、epsilon random、min-per-arm、dataset-aware stats、value=`legacy/success/base/normalized_base`、random seed。 | `terminal-rl/explore_agent57_lite.py:828-1095` `_sqlite_arm_stats()` / `_ucb_scores()` / `assign_group_arms()` / `record_arm_event()` | 原文列的 UCB 改进大多已实现；但真实 UCB 训练 run 尚未验证。 |
| Rollout metadata 与采样参数 | rollout group 提交前写入 `agent57_arm_id`、group position、dataset；可按 arm 覆盖 temperature/top_p/top_k。 | `slime/slime/rollout/sglang_rollout.py:260-347` `_apply_agent57_sampling_params()` / `_assign_agent57_arms()` / `_annotate_rollout_groups()`；调用点 `slime/slime/rollout/sglang_rollout.py:751` | 当前是 sampling/reward mixing，不是 LoRA heads 或独立 policies。 |
| Reward 写回与 arm event | 生成结束后把 exploration total 写入 sample reward/metadata，并记录 arm outcome。 | `terminal-rl/generate.py:2441-2640` reward block；`terminal-rl/generate.py:2564` `_agent57_record_arm_event()` | `ngu_lite` 下 `_intr_for_total=0.0`，避免 standalone intrinsic double-count。 |
| Agent57 metrics 汇总 | 汇总 lifelong/NGU/arm/suppression/per-dataset 指标，格式化 Agent57 table，并在 structured record 中带 Agent57 字段。 | `terminal-rl/rollout_log.py:468-655` `_agent57_summary()` / `_add_agent57_debug_metrics()`；`terminal-rl/rollout_log.py:900-954` record 字段；`terminal-rl/rollout_log.py:1602-1646` table | 代码具备 metrics 路径；已检查的 Agent57 runs 未观察到实际 `metrics.jsonl` 文件。 |
| Episodic memory backend 模块 | 已实现 count backend 与 SimHash-KNN backend，含 factory/env aliases 和单测。 | `terminal-rl/agent57_episodic_memory.py:126-218` `CountBasedEpisodicMemory`；`terminal-rl/agent57_episodic_memory.py:231-377` `SimHashKNNEpisodicMemory`；`terminal-rl/agent57_episodic_memory.py:379-470` factory | 模块已实现，但 `generate.py` 主训练路径尚未调用 factory。 |
| 启动脚本 env 导出 | exploration wrapper 和主训练脚本已导出 Agent57/episodic/lifelong/UCB env，并写入 `run_config.json`。 | `terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh:140-182` defaults、`405-408` exports；`terminal-rl/terminal-rl_qwen3-8b_pu.sh:171-213` defaults、`1293-1335` run_config | 原文建议的多数 env 已存在；默认 `MAX_CKPT_KEEP=0`，不自动保存 checkpoint。 |
| 单元测试文件 | 覆盖 NGU clamp、UCB min/dataset/random seed、lifelong key v1/v2、sqlite migration、episodic memory backend。 | `terminal-rl/tests/test_explore_agent57_lite.py:24-253`；`terminal-rl/tests/test_agent57_episodic_memory.py:22-125` | 测试文件存在；当前 shell 环境缺 pytest，未能在本次执行完整跑通。 |

## 尚未实现 / 部分实现的部分

| 状态 | 项目 | 当前缺口 / 阻塞点 |
| --- | --- | --- |
| 部分实现 | Episodic backend 接入训练主路径 | `agent57_episodic_memory.py` 已实现并有测试，但 `generate.py` 仍使用 legacy signature novelty；还缺 per-episode lifecycle、reset/clear 策略和 metrics 对齐。 |
| 部分实现 | Structured metrics 文件落盘验证 | `rollout_log.py` 和脚本配置了 `TERMINAL_METRICS_JSONL`，但已检查的 Agent57 run 目录没有 `logs/metrics.jsonl`；需要确认 writer 是否在当前训练入口被调用。 |
| 有 stub | Multi-attempt reflection | `terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh:357-362` 只导出 env 并打印警告：`agent_runner` 支持未实现。 |
| 未开始 | 完整 Agent57 replay/retrace/value learning | 当前没有 replay buffer、retrace target、UVFA 或 gamma-conditioned value/policy heads；Terminal-RL 仍走现有 rollout reward/advantage 链路。 |
| 未开始 | Neural RND predictor | 没有 random target/predictor network、predictor loss、同步或 running normalization；当前 lifelong 是 count-based。 |
| 未开始 | LoRA heads / 真正 policy-space mixing | arms 只影响 reward beta 和 sampling params，没有独立 adapter/head。 |
| 未开始 | Agent57 专用分析脚本 | 未发现 `terminal-rl/scripts/analyze_agent57_lite.py`；当前只能靠日志、sqlite 和通用脚本手动分析。 |
| 已实现但未实验验证 | UCB controller 真实训练效果 | UCB 代码已具备 epsilon/min/dataset-aware/normalized-base，但已有 run 均为 `controller=fixed`。 |

## 测试与运行命令

说明：测试文件未发现专用 `pytest.mark` marker；建议用文件路径或 `-k agent57` 选择。

### 单元测试

```bash
python3 -m pytest \
  terminal-rl/tests/test_explore_agent57_lite.py \
  terminal-rl/tests/test_agent57_episodic_memory.py
```

用途：验证 Agent57 config、NGU-lite clamp、UCB、lifelong keys/sqlite migration、episodic memory backends。
预期产出：pytest pass/fail 输出；不产生训练 run。
本次状态：`python3 -m pytest --version` 失败，当前系统 Python 无 pytest；`.venv/bin/python` 是指向 `/root/.local/share/uv/.../python3.12` 的失效 symlink。

可选筛选：

```bash
python3 -m pytest -k "agent57 or episodic" terminal-rl/tests
```

### 轻量 import smoke

```bash
python3 -c "import sys; sys.path.insert(0, 'terminal-rl'); import explore_agent57_lite as a57; import agent57_episodic_memory as mem; print(a57.config_from_env().combine_mode); print(mem.resolve_episodic_backend_name('knn'))"
```

用途：确认当前 Python 至少能导入核心 Agent57 模块和 episodic backend factory。
本次结果：通过，输出 `add` 和 `simhash_knn`。

### 端到端 smoke 训练

```bash
DEBUG_MODE=1 \
DATASET=seta \
ALGO=dapo \
HARNESS_OPTION=camel-agent \
EXPLORATION_PROFILE=spear_lite \
EXPLORE_AGENT57_LITE=1 \
EXPLORE_AGENT57_LIFELONG=1 \
EXPLORE_AGENT57_LIFELONG_BACKEND=sqlite \
EXPLORE_AGENT57_COMBINE_MODE=ngu_lite \
EXPLORE_AGENT57_LIFELONG_KEY_VERSION=v2 \
EXPLORE_AGENT57_MAX_BONUS=0.05 \
EXPLORE_ADVANTAGE_BONUS=1 \
EXPLORE_ADVANTAGE_BONUS_COMPONENTS=explore_intrinsic_scaled,explore_agent57_ngu_bonus \
MAX_CKPT_KEEP=0 \
bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

用途：用 debug 小 batch 验证 rollout annotation、reward 写回、sqlite lifelong state 与日志是否连通。
默认/关键参数：主脚本默认 `MAX_TURN=10`、debug 下 `ROLLOUT_BATCH_SIZE=4`、`N_SAMPLES=2`；Agent57 wrapper 默认 `K=8`、`controller=fixed`、`lifelong_warmup=64`。
预期产出：`runs/<RUN_ID>/config/run_config.json`、`runs/<RUN_ID>/logs/train.log`、`runs/<RUN_ID>/trajectories/`、`runs/<RUN_ID>/agent57_lite.sqlite3`；如 metrics writer 生效，还应有 `runs/<RUN_ID>/logs/metrics.jsonl`。默认 `MAX_CKPT_KEEP=0`，不保存 checkpoint。

### 常规 NGU-lite 训练启动

```bash
DATASET=seta \
ALGO=dapo \
HARNESS_OPTION=camel-agent \
EXPLORATION_PROFILE=spear_lite \
EXPLORE_INTRINSIC=1 \
EXPLORE_INTRINSIC_COEF=0.015 \
EXPLORE_INTRINSIC_SCHEDULE=cosine \
EXPLORE_INTRINSIC_DECAY_STEPS=120 \
EXPLORE_CDE_ACTOR=1 \
EXPLORE_CDE_ACTOR_OMEGA=0.02 \
EXPLORE_AGENT57_LITE=1 \
EXPLORE_AGENT57_K=8 \
EXPLORE_AGENT57_ARM_BETAS="0,0.002,0.004,0.006,0.008,0.01,0.015,0.02" \
EXPLORE_AGENT57_CONTROLLER=fixed \
EXPLORE_AGENT57_COMBINE_MODE=ngu_lite \
EXPLORE_AGENT57_MAX_BONUS=0.05 \
EXPLORE_AGENT57_LIFELONG=1 \
EXPLORE_AGENT57_LIFELONG_BACKEND=sqlite \
EXPLORE_AGENT57_LIFELONG_COEF=0.005 \
EXPLORE_AGENT57_LIFELONG_WARMUP=64 \
EXPLORE_AGENT57_LIFELONG_KEY_VERSION=v2 \
EXPLORE_AGENT57_UCB_RANDOM_SEED=20260605 \
EXPLORE_ADVANTAGE_BONUS=1 \
EXPLORE_ADVANTAGE_BONUS_COMPONENTS=explore_intrinsic_scaled,explore_agent57_ngu_bonus \
MAX_CKPT_KEEP=0 \
bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

用途：复现最近一次 NGU-lite run 的核心配置。
预期产出同上；checkpoint 只有在 `MAX_CKPT_KEEP>0` 时才会保留。

## 已运行过的实验记录

公共前缀：以下 run 均在 `runs/terminal-rl_qwen3-8b_8gpu_seta_dapo_harness-camel-agent_explore_spear_lite_int_cosine120_a57_life0.005_postnorm_cdeact0.02_2026-06-04_*`。

| run 后缀 | commit / 启动时间 | 配置摘要 | 产物 | 关键结论 |
| --- | --- | --- | --- | --- |
| `134734` | `e699d1a3`；`2026-06-04T13:47:47Z` | SetA + DAPO + camel-agent；additive 时代配置：`controller=fixed`、sqlite lifelong、advantage 只含 `explore_intrinsic_scaled`。 | `..._134734/config/run_config.json`、`..._134734/logs/train.log`、`..._134734/trajectories/`；未发现 sqlite 和 `metrics.jsonl`。 | 早停/失败：尾部为 env reset timeout，dynamic sampling 取消 pending rollout。 |
| `140928` | `e699d1a3`；`2026-06-04T14:09:34Z` | 同 additive 时代配置。 | `..._140928/config/run_config.json`、`..._140928/logs/train.log`、`..._140928/trajectories/`；未发现 sqlite 和 `metrics.jsonl`。 | 短尝试/待定：尾部仍在生成，并出现 tool-call JSON parse error；无可比训练指标。 |
| `141253` | `e699d1a3`；`2026-06-04T14:12:59Z` | additive 时代配置；sqlite lifelong enabled；advantage 只含 `explore_intrinsic_scaled`。 | `..._141253/agent57_lite.sqlite3`、大量 `..._141253/trajectories/`、`..._141253/logs/train.log`；未发现 `metrics.jsonl`。 | 长 run 有轨迹和 sqlite，但尾部出现 evaluate HTTP 500、container missing、close 500；结果被 env 稳定性主导，训练结论待定。 |
| `163947` | `47f772c6`；`2026-06-04T16:39:56Z` | 最新 NGU-lite 配置：`combine_mode=ngu_lite`、`max_bonus=0.05`、`lifelong_key_version=v2`、`ucb_random_seed=20260605`、`controller=fixed`、advantage 含 `explore_intrinsic_scaled,explore_agent57_ngu_bonus`。 | `..._163947/agent57_lite.sqlite3`、大量 `..._163947/trajectories/`、`..._163947/logs/train.log`；未发现 `metrics.jsonl`。 | NGU-lite 主路径已实际跑到 rollout/轨迹产出；尾部为 terminal env allocate timeout，仍无法得出收益结论。 |

可比指标状态：

| 指标 | 当前观察 |
| --- | --- |
| `metrics.jsonl` | 四个 Agent57 run 的预期路径存在于 `run_config.json`，但文件未观察到。 |
| sqlite state | 仅 `141253` 与 `163947` 观察到 `agent57_lite.sqlite3`。 |
| checkpoint | 现有 run 未在当前 sandbox 内确认 checkpoint；脚本默认 `MAX_CKPT_KEEP=0`。 |
| 主要失败模式 | env reset/allocate timeout、evaluate/close HTTP 500、container missing、tool-call parse error。 |

## 后续迭代计划（按可行性排序）

| 优先级 | 项目 | 动机 | 改动范围 | 验证方式 | 风险点 |
| --- | --- | --- | --- | --- | --- |
| 1 | 恢复可跑测试环境 | 当前单测文件存在但 shell 环境缺 pytest，无法快速防回归。 | 主要是运行环境；必要时检查 `.venv` 或项目依赖声明。 | 成功运行 `terminal-rl/tests/test_explore_agent57_lite.py` 与 `terminal-rl/tests/test_agent57_episodic_memory.py`。 | 不要污染线上训练 conda/uv 环境；先确认训练实际使用的 Python。 |
| 2 | 先做低并发 env 稳定性 smoke | 已有 run 的主要失败不是 reward 逻辑，而是 env allocate/reset/evaluate/close。 | `terminal-rl/terminal-rl_qwen3-8b_pu.sh` 参数、env worker 配置；必要时再看 `terminal-rl/generate.py` timeout/close 路径。 | `DEBUG_MODE=1` Agent57 run 能完整产出轨迹、sqlite、日志，且无持续 500/timeout。 | 训练吞吐和远端 Docker 状态噪声大；需要避免把 infra 问题误判为探索问题。 |
| 3 | 确认并修复 `metrics.jsonl` 落盘 | 没有结构化指标时无法比较 arms、suppression 和 reward scale。 | `terminal-rl/rollout_log.py`、训练入口 metrics writer、可新增 `terminal-rl/scripts/analyze_agent57_lite.py`。 | 新 run 出现 `runs/<RUN_ID>/logs/metrics.jsonl`，脚本能打印 arm coverage、bonus、suppression、state error。 | 日志频率过高可能增加 IO；schema 变更需兼容已有分析脚本。 |
| 4 | 接入 `agent57_episodic_memory` backend | 已实现的 count/SimHash-KNN backend 尚未被训练主路径使用，可先作为可选 episodic source。 | `terminal-rl/generate.py`、`terminal-rl/agent57_episodic_memory.py`、对应 tests。 | 在 `EXPLORE_AGENT57_EPISODIC_BACKEND=count|simhash_knn` 下跑单测与 import smoke，再跑 debug rollout 比较 novelty/latency。 | per-episode reset、容量和并发状态管理容易引入隐性差异。 |
| 5 | 做 fixed add vs fixed NGU-lite 小规模 A/B | 当前代码已支持两种 combine mode，但没有稳定对照。 | 主要是运行配置；必要时只调整 wrapper 默认值。 | 同一 dataset/seed/低并发下比较 parse/truncation、lifelong eligible、NGU bonus、task reward、trajectory 成功率。 | env 抖动会掩盖 reward 差异；必须先完成第 2、3 项。 |
| 6 | 在 fixed 稳定后验证 UCB | UCB 代码已实现但未真实训练验证。 | 配置为主：`EXPLORE_AGENT57_CONTROLLER=ucb`、`EXPLORE_AGENT57_UCB_EPSILON`、`MIN_PER_ARM`、`VALUE=normalized_base`、`DATASET_AWARE=1`。 | 观察 arm coverage 不塌缩、baseline arm 保留、per-arm normalized base 不恶化。 | UCB 可能过早偏向高噪声 arm；mixed dataset reward scale 仍需小心。 |

更新时间：2026-06-05 15:50:44 HKT；基于 commit `6be53312eb52f2c52be27ae209e99feaec3c6bdf` 的代码状态。
