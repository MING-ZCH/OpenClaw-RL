# Agent57 测试命令清单

本文档列出当前 Agent57 / Agent57-Lite 相关测试命令。默认工作目录：

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL
```

## 1. 语法与 shell 检查

用途：快速确认 Python 文件可编译、训练脚本 shell 语法正确。

预计时长：10-30 秒。

预期产出：无输出且退出码为 0。

```bash
python3 -m py_compile \
  terminal-rl/agent57_episodic_memory.py \
  terminal-rl/explore_agent57_lite.py \
  terminal-rl/generate.py \
  terminal-rl/rollout_log.py \
  terminal-rl/tests/test_agent57_episodic_memory.py \
  terminal-rl/tests/test_explore_agent57_lite.py

bash -n \
  terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh \
  terminal-rl/terminal-rl_qwen3-8b_pu.sh
```

## 2. 单元测试

用途：覆盖 episodic memory backend、UCB seed、lifelong key、sqlite schema migration。

预计时长：30-90 秒。

预期产出：pytest 全部通过。

```bash
PYTHONPATH=terminal-rl python3 -m pytest \
  terminal-rl/tests/test_agent57_episodic_memory.py \
  terminal-rl/tests/test_explore_agent57_lite.py
```

如果当前环境没有安装 pytest，可先运行下面的 smoke test。

## 3. Episodic backend smoke test

用途：不依赖 pytest，直接验证 count 和 SimHash-KNN 的 add/query/serialize 基本路径。

预计时长：5-10 秒。

预期产出：输出 `episodic memory smoke ok`。

```bash
PYTHONPATH=terminal-rl python3 - <<'PY'
from agent57_episodic_memory import CountBasedEpisodicMemory, SimHashKNNEpisodicMemory

m = CountBasedEpisodicMemory()
assert m.compute_novelty("x") == 1.0
m.add("x")
assert m.compute_novelty("x") < 1.0

state = m.state_dict()
restored = CountBasedEpisodicMemory()
restored.load_state_dict(state)
assert restored.compute_novelty("x") == m.compute_novelty("x")

k = SimHashKNNEpisodicMemory()
assert k.compute_novelty([1, 0, 0]) == 1.0
k.add([1, 0, 0])
assert k.compute_novelty([1, 0, 0]) == 0.0

print("episodic memory smoke ok")
PY
```

## 4. UCB seed 复现性验证

用途：确认 `EXPLORE_AGENT57_UCB_RANDOM_SEED` 控制 UCB tie-breaking 和 epsilon 随机选择。

预计时长：5-10 秒。

预期产出：输出 `ucb seed smoke ok`，两次同 seed 序列一致。

```bash
PYTHONPATH=terminal-rl python3 - <<'PY'
import os
import explore_agent57_lite as a57

os.environ["EXPLORE_AGENT57_LITE"] = "1"
os.environ["EXPLORE_AGENT57_CONTROLLER"] = "ucb"
os.environ["EXPLORE_AGENT57_K"] = "6"
os.environ["EXPLORE_AGENT57_KEEP_BASELINE"] = "0"
os.environ["EXPLORE_AGENT57_UCB_EPSILON"] = "0.5"
os.environ["EXPLORE_AGENT57_UCB_RANDOM_SEED"] = "123"

a57._LOCAL_ARM_EVENTS.clear()
a57._reset_ucb_rng_for_tests()
first = [a57.assign_group_arms(6) for _ in range(4)]

a57._reset_ucb_rng_for_tests()
second = [a57.assign_group_arms(6) for _ in range(4)]

assert first == second, (first, second)
print("ucb seed smoke ok", first[0])
PY
```

## 5. 不同 episodic backend 对比

用途：验证 factory 环境变量和 backend 行为差异。

预计时长：5-10 秒。

预期产出：输出三行 backend 类型和 novelty。

```bash
PYTHONPATH=terminal-rl python3 - <<'PY'
import os
from agent57_episodic_memory import create_episodic_memory_backend

for backend in ["legacy", "count", "simhash_knn"]:
    os.environ["EXPLORE_AGENT57_EPISODIC_BACKEND"] = backend
    memory = create_episodic_memory_backend()
    if memory is None:
        print(backend, "legacy path")
        continue
    before = memory.compute_novelty("pytest failure")
    memory.add("pytest failure")
    after = memory.compute_novelty("pytest failure")
    print(backend, type(memory).__name__, before, after)
PY
```

## 6. 端到端训练 smoke test

用途：启动最小 Agent57-Lite 配置，确认 env 转发、run_config 落盘、worker 侧导入不报错。

预计时长：取决于集群和模型启动，通常 10-30 分钟以上。建议只在 GPU/Ray 环境完整时运行。

预期产出：

- `logs/<run_name>/train.log` 中出现 `ucb_seed=20260604 episodic=legacy`
- `runs/<run_name>/config/run_config.json` 中包含 `explore_agent57_ucb_random_seed`
- rollout metrics 中包含 `explore_agent57_ucb_random_seed`

```bash
EXPLORE_AGENT57_LITE=1 \
EXPLORE_AGENT57_LIFELONG=1 \
EXPLORE_AGENT57_LIFELONG_BACKEND=sqlite \
EXPLORE_AGENT57_CONTROLLER=ucb \
EXPLORE_AGENT57_UCB_VALUE=normalized_base \
EXPLORE_AGENT57_UCB_DATASET_AWARE=1 \
EXPLORE_AGENT57_UCB_RANDOM_SEED=20260604 \
EXPLORE_AGENT57_COMBINE_MODE=ngu_lite \
EPISODIC_MEMORY_BACKEND=legacy \
MAX_CKPT_KEEP=1 \
TRAJECTORY_SAVE_INTERVAL=10 \
EXTRA_DAPO_ARGS="--dynamic-sampling-max-groups 8 --dynamic-sampling-max-seconds 600 --rollout-abort-wait-timeout 300" \
bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

## 7. Count backend 实验性 smoke

用途：检查 count backend 配置能进入 run config；当前不会自动替换 legacy reward path，除非后续代码显式接入 factory。

预计时长：与端到端训练 smoke 相同。

预期产出：`run_config.json` 中 `explore_agent57_episodic_backend=count`。

```bash
EXPLORE_AGENT57_LITE=1 \
EXPLORE_AGENT57_LIFELONG=1 \
EXPLORE_AGENT57_EPISODIC_BACKEND=count \
EXPLORE_AGENT57_EPISODIC_CAPACITY=4096 \
EXPLORE_AGENT57_EPISODIC_COUNT_DECAY=0.99 \
EXPLORE_AGENT57_UCB_RANDOM_SEED=20260604 \
bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```

## 8. SimHash-KNN backend 实验性 smoke

用途：检查 SimHash-KNN 配置解析、seed 和 run config 落盘。

预计时长：与端到端训练 smoke 相同。

预期产出：`run_config.json` 中 `explore_agent57_episodic_backend=simhash_knn`。

```bash
EXPLORE_AGENT57_LITE=1 \
EXPLORE_AGENT57_LIFELONG=1 \
EXPLORE_AGENT57_EPISODIC_BACKEND=simhash_knn \
EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS=64 \
EXPLORE_AGENT57_EPISODIC_BUCKET_CAPACITY=256 \
EXPLORE_AGENT57_EPISODIC_K=5 \
EXPLORE_AGENT57_EPISODIC_DISTANCE=cosine \
EXPLORE_AGENT57_EPISODIC_RANDOM_SEED=20260604 \
bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh
```
