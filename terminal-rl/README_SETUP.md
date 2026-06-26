# Terminal RL

RL training for terminal agents. The agent interacts with Docker-hosted environments and is trained with GRPO (optional PRM).

This workflow has two independent components:

- **Training machine** runs task router + Ray + training scripts, and connects to workers via `WORKER_URLS`
- **Remote workers:** run the pool server and execute tasks (Docker required): [remote/README.md](remote/README.md)

---

## Prerequisites

- **Training machine:** GPU node/cluster with the required training dependencies.
- **Remote workers:** Docker-capable hosts reachable from the training machine (default pool server port **18081**). Setup: [remote/README.md](remote/README.md).

---

## Instructions

### 0. Start remote workers (pool server)

Follow [remote/README.md](remote/README.md) on each worker to start `pool_server` (it should be reachable at e.g. `http://<worker-ip>:18081`).

### 1. Clone the repo

From a directory of your choice:

```bash
git clone https://github.com/Gen-Verse/OpenClaw-RL.git
cd OpenClaw-RL
```

### 2. Prepare dataset (download + convert)

Download a supported dataset under `terminal-rl/dataset/`:

```bash
export DATASET_DIR="terminal-rl/dataset"
python terminal-rl/data_utils/download.py seta_env
```

Convert tasks into training JSONL:

```bash
python terminal-rl/data_utils/convert_task_to_dataset.py \
  --tasks_dir terminal-rl/dataset/seta_env
```

The `seta_env` dataset corresponds to the task dataset published in: [camel-ai/seta-env](https://github.com/camel-ai/seta-env/tree/main/Dataset).

### 3. Run training

On the training machine, set the required environment variables:

```bash
# Hugging Face cache / model paths
export HF_HOME="/path/to/huggingface"
export MODEL_CKPT="/path/to/model"
export REF_LOAD="/path/to/reference_model_dir"
export SAVE_CKPT="/path/to/save/checkpoints"

# Dataset + workers
export ROLLOUT_PROMPT_DATA="/path/to/train.jsonl"
export WORKER_URLS="http://worker1:18081,http://worker2:18081"

# Logging
export WANDB_KEY="your-wandb-key"
```

Then run (from repo root):

```bash
bash terminal-rl/terminal_qwen3_8b_rl.sh
```

---

### Qwen3-8B eval-only on puyuan 集群

在 4-GPU 节点上从 `/mnt/shared-storage-user/puyuan/code/OpenClaw-RL` 运行以下命令。`mock` suite 会评测 AgentSafetyBench train 和 AgentHarm val，并使用本地 mock env，因此不需要 Docker worker。

```bash
EVAL_SUITE=mock EVAL_CKPT=init \
  bash terminal-rl/terminal-rl_qwen3-8b_eval_pu.sh

EVAL_SUITE=mock EVAL_CKPT=step119 \
  bash terminal-rl/terminal-rl_qwen3-8b_eval_pu.sh
```

使用每个 dataset 两条样本做 smoke test：

```bash
EVAL_SUITE=mock EVAL_CKPT=init EVAL_LIMIT=2 \
  bash terminal-rl/terminal-rl_qwen3-8b_eval_pu.sh
```

Docker worker ready 后运行 SETA：

```bash
EVAL_SUITE=seta EVAL_CKPT=init WORKER_URLS=http://<worker-host>:18081 \
  bash terminal-rl/terminal-rl_qwen3-8b_eval_pu.sh

EVAL_SUITE=seta EVAL_CKPT=step119 WORKER_URLS=http://<worker-host>:18081 \
  bash terminal-rl/terminal-rl_qwen3-8b_eval_pu.sh
```

SETA worker ready 后，也可以一次运行全部三个 benchmark：

```bash
EVAL_SUITE=all EVAL_CKPT=init WORKER_URLS=http://<worker-host>:18081 \
  bash terminal-rl/terminal-rl_qwen3-8b_eval_pu.sh

EVAL_SUITE=all EVAL_CKPT=step119 WORKER_URLS=http://<worker-host>:18081 \
  bash terminal-rl/terminal-rl_qwen3-8b_eval_pu.sh
```

输出目录为 `runs/eval_qwen3-8b_<ckpt>_<suite>_<timestamp>/`。主要文件包括 `logs/eval.log`、`logs/ray_job.log`、`logs/metrics.jsonl`、`eval_summary.json` 和 `eval_summary.tsv`。在 summary 文件中，internal dataset label `security` 会被归一化为 `agent_safetybench`。

当前 Qwen3-8B eval 完成状态和 score table 记录在：
`terminal-rl/docs/qwen3_8b_eval_status_20260609.md`.

---

### PRM training (optional)

To enable PRM scoring with the 2-node script, add:

```bash
export PRM_ENABLE=1
export PRM_MODEL_PATH="/path/to/prm-model"
export PRM_M=3
export PRM_STEP_COEF=1.0
export PRM_TEMPERATURE=0.0
export PRM_MAX_NEW_TOKENS=4096
# Optional: use an external PRM endpoint instead of framework-hosted engines
export PRM_SGLANG_URL="http://<prm-router-ip>:<prm-router-port>"
```

Then run:

```bash
bash terminal-rl/terminal_qwen3_8b_prm_rl_2nodes.sh
```

---

## Notes

- `WORKER_URLS` must point to already-running pool servers.
- As an example, one rollout agent implementation in this repo is based on **CAMEL** (see `terminal-rl/agent/camel_agent.py` and [CAMEL](https://github.com/camel-ai/camel)).
