# CPU Worker Docker 运维手册

本文档覆盖 terminal-rl 训练中 CPU worker 的两条主要运维路径：

- 场景 A：从零配置一台新的 CPU worker。
- 场景 B：已配置机器上的 Docker / pool_server 异常后的恢复。

所有命令默认在 CPU worker 上执行，仓库路径默认：

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL
```

## 概述与前置依赖

CPU worker 负责运行 `pool_server`，由它管理 Docker 容器并执行 seta terminal task。GPU worker 通过 `WORKER_URLS=http://<cpu-ip>:18081` 调用它。

核心脚本：

| 脚本 | 职责 |
|---|---|
| `terminal-rl/remote/setup_new_worker.sh` | 场景 A 入口：安装 Docker/Compose，写 daemon 配置，代理加固，安装 watchdog，验证构建 |
| `terminal-rl/remote/fix_dockerd_and_proxy.sh` | 场景 B 入口：修复 Docker 挂掉、代理丢失、base image apt 代理缺失 |
| `terminal-rl/remote/docker_worker_doctor.sh` | 训练日志感知的诊断/修复入口：统计 `/reset`、`/evaluate`、Docker exit code、`Errno 11`，并采集 CPU worker Docker 状态 |
| `terminal-rl/remote/restart_docker_force.sh` | 低层强制重启 dockerd，绕过卡住的 `systemctl restart docker` |
| `terminal-rl/remote/run_pool_server_pu_v2.sh` | 启动 pool_server，自动加载 `/etc/seta_build_proxy.env` |
| `terminal-rl/remote/docker_watchdog_v2.sh` | watchdog 主循环，监控 dockerd、pool_server、容器/网络压力 |
| `terminal-rl/remote/docker-watchdog.service` | watchdog 的 systemd unit |
| `terminal-rl/remote/prebuild_proxied_base_images.sh` | 给常用 base image 注入 apt proxy |
| `terminal-rl/remote/diag_docker_failures_lite.sh` | 训练期间可运行的轻量诊断 |
| `terminal-rl/remote/cleanup_docker_cache.sh` | 清理 stopped containers、build cache、dangling volumes/networks |
| `terminal-rl/remote/safevo_docker_storage_doctor.sh` | safevo/rlinfra Docker data-root 深度诊断：区分 overlay2 image layer、container writable layer、BuildKit/orphan、json log，并支持保守修复 |
| `terminal-rl/remote/docker_overlay2_orphan_audit.py` | overlay2 离线可达性审计：从 layerdb 出发沿 lower 链找不可达目录，支持停 Docker 后 quarantine 回滚式处理 |
| `terminal-rl/remote/docker_storage_gc.py` | Docker data-root 自动 GC：阈值触发、dry-run、保留白名单、按旧 image 逐个删除直到目标水位 |

前置要求：

- Ubuntu 20.04+。
- 当前用户可 `sudo`。
- Docker data root 所在分区建议不少于 150 GB，长跑建议 500 GB+。
- CPU worker 的 `18081` 端口能被 GPU worker 访问。
- 需要访问 `ghcr.io` / `docker.io` / Ubuntu apt 源；pjlab 环境默认代理为 `http://httpproxy-headless.kubebrain.svc.pjlab.local:3128`。

常用变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DOCKER_DATA_ROOT` | `/var/lib/docker` in setup, `/data` in repair | Docker data root；`DOCKER_ROOT` 仍作为兼容别名 |
| `PROXY_URL` | pjlab proxy | Docker pull/build/apt 使用的代理 |
| `NO_PROXY_LIST` | 内网与本机地址 | 代理绕过列表 |
| `ENV_SERVER_PORT` | `18081` | pool_server 端口 |
| `SKIP_VERIFY` | `0` | 设为 `1` 可跳过构建验证 |

## 场景 A：从零配置 CPU Worker

适用：新机器、未安装 Docker、未配置代理、未安装 watchdog。

### A1. 执行一键初始化

推荐使用大盘作为 Docker data root：

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

sudo env DOCKER_DATA_ROOT=/data \
  PROXY_URL=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128 \
  bash terminal-rl/remote/setup_new_worker.sh
```

该脚本会执行：

- 安装 Docker 和 Docker Compose V2。
- 写入 `/etc/docker/daemon.json`，包括 `data-root`、address pool、ulimit、日志限制。
- 写入 Docker systemd proxy drop-in。
- 预拉常用 base image。
- 创建 `.venv` 并安装 pool_server 依赖。
- 调用 `fix_dockerd_and_proxy.sh` 写入 `/etc/seta_build_proxy.env` 并注入 base image apt proxy。
- 安装并启动 `docker-watchdog.service`。
- 构建一个 seta task 做最小验证。

可选参数：

```bash
# 非交互通过低磁盘告警
sudo env ASSUME_YES=1 DOCKER_DATA_ROOT=/data bash terminal-rl/remote/setup_new_worker.sh

# 只做基础安装，不跑代理修复和构建验证
sudo env DOCKER_DATA_ROOT=/data RUN_PROXY_FIX=0 SKIP_VERIFY=1 bash terminal-rl/remote/setup_new_worker.sh

# 不安装 watchdog
sudo env DOCKER_DATA_ROOT=/data INSTALL_WATCHDOG=0 bash terminal-rl/remote/setup_new_worker.sh
```

### A2. 验证基础设施

```bash
timeout 10 docker info >/dev/null && echo "[OK] docker"
docker compose version
test -f /etc/seta_build_proxy.env && echo "[OK] proxy env"
systemctl is-active docker-watchdog && echo "[OK] watchdog"
journalctl -u docker-watchdog -n 20 --no-pager
```

检查 base image 是否已注入 apt proxy：

```bash
docker run --rm ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624 \
  sh -c 'test -f /etc/apt/apt.conf.d/95proxies && echo "[OK] apt proxy"'
```

### A3. 启动 pool_server

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

# 前台运行，便于观察日志
bash terminal-rl/remote/run_pool_server_pu_v2.sh
```

后台运行：

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

nohup bash terminal-rl/remote/run_pool_server_pu_v2.sh \
  > /tmp/cpu_pool.log 2>&1 &
echo $! > /tmp/cpu_pool.pid
```

验证：

```bash
curl --noproxy '*' http://127.0.0.1:18081/healthz
curl --noproxy '*' http://127.0.0.1:18081/status | python3 -m json.tool
```

### A4. 在 GPU worker 上连接 CPU worker

在 CPU worker 上查询 IP：

```bash
hostname -I | awk '{print $1}'
```

在 GPU worker 上配置：

```bash
export WORKER_URLS="http://<cpu-worker-ip>:18081"

DATASET=mixed \
MIX_SETA_RATIO=1 \
MIX_SAFETY_RATIO=1 \
bash /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/terminal-rl/terminal-rl_qwen3-8b_pu.sh
```

## 场景 B：Docker 服务挂掉后的修复

适用：机器已配置过，但出现以下现象：

- `docker info` 卡住或失败。
- `pool_server /healthz` 不通。
- GPU worker 日志中出现 `/allocate`、`/reset` 502/500。
- seta build 失败，常见为 apt timeout、exit status 17、proxy 不生效。
- watchdog 日志出现 `Health check failed`、`deep probe failed`、address-pool 风险。

### B1. 诊断

推荐优先使用 `docker_worker_doctor.sh`。它会同时读取 GPU 侧 `train.log` 和 CPU worker 本机 Docker / pool_server 状态，输出 `SUMMARY.md` 和打包后的诊断归档：

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

bash terminal-rl/remote/docker_worker_doctor.sh diagnose \
  --train-log /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/runs/<run>/logs/train.log
```

如果 GPU 日志不在 CPU worker 本机，可以先只做本机诊断：

```bash
bash terminal-rl/remote/docker_worker_doctor.sh diagnose
```

先跑轻量诊断，不会做 build 探针，训练中也可以运行：

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

bash terminal-rl/remote/diag_docker_failures_lite.sh
cat tmp_doc_latest/cpu_diag_summary.txt
```

手动检查：

```bash
timeout 10 docker info
systemctl status docker --no-pager
systemctl status docker-watchdog --no-pager
journalctl -u docker-watchdog -n 80 --no-pager
curl --noproxy '*' --max-time 5 http://127.0.0.1:18081/healthz
ss -tlnp | grep ':18081 '
```

### B2. 一键修复

如果 `docker_worker_doctor.sh diagnose` 的摘要里出现大量 `/reset 500`、`exit status 125`、`exit status 17/2` 或 `Errno 11`，推荐直接走完整恢复：

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

sudo env DOCKER_DATA_ROOT=/data \
  PROXY_URL=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128 \
  bash terminal-rl/remote/docker_worker_doctor.sh full-repair \
  --train-log /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/runs/<run>/logs/train.log
```

如果只是轻微堆积、Docker API 仍响应，可以先做保守恢复：

```bash
sudo env DOCKER_DATA_ROOT=/data \
  bash terminal-rl/remote/docker_worker_doctor.sh soft-repair
```

推荐优先使用完整恢复入口：

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

sudo env DOCKER_DATA_ROOT=/data \
  PROXY_URL=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128 \
  bash terminal-rl/remote/fix_dockerd_and_proxy.sh
```

该脚本会：

- 停止 watchdog，避免它和人工修复抢 dockerd。
- 强制清理 stale `dockerd`、`containerd-shim`、docker pid/sock。
- 清理 stale container state 和 network kv。
- 重写 Docker / watchdog / 用户级 proxy 配置。
- 写入 `/etc/seta_build_proxy.env`。
- 启动 Docker 并验证 API。
- 预构建带 apt proxy 的 base image。
- 验证 `seta_env/0` 构建。
- 按需恢复 watchdog。

如果只需要快速拉起 Docker，不改代理配置：

```bash
sudo env DOCKER_DATA_ROOT=/data bash terminal-rl/remote/restart_docker_force.sh
```

如果怀疑 cache 或 dangling network 过多：

```bash
bash terminal-rl/remote/cleanup_docker_cache.sh
```

如果 Docker data root 已接近打满且 `cleanup_docker_cache.sh` 无法释放足够空间，先做只读诊断：

```bash
DOCKER_DATA_ROOT=/data \
python3 terminal-rl/remote/docker_storage_gc.py --diagnose-only
```

如果需要解释 `/data/overlay2` 到底由 image layer、container writable layer、BuildKit cache 还是 Docker json log 贡献，优先运行 doctor 脚本。默认只读，不删除任何对象：

```bash
DOCKER_DATA_ROOT=/data \
bash terminal-rl/remote/safevo_docker_storage_doctor.sh
```

如果 `/data/overlay2` 目录很多、全量 `du` 太慢，可以跳过全量扫描，只做抽样：

```bash
DOCKER_DATA_ROOT=/data RUN_OVERLAY_DU=0 RUN_OVERLAY_SAMPLE=1 OVERLAY_SAMPLE_N=200 \
bash terminal-rl/remote/safevo_docker_storage_doctor.sh
```

如需预览保守修复动作：

```bash
MODE=repair APPLY=0 DOCKER_DATA_ROOT=/data \
bash terminal-rl/remote/safevo_docker_storage_doctor.sh
```

确认后执行保守修复。该模式不会删除已 tag 的 image，只清理 stopped container、unused network、旧 builder cache、dangling image；volume 和日志截断仍需单独打开：

```bash
MODE=repair APPLY=1 DOCKER_DATA_ROOT=/data \
bash terminal-rl/remote/safevo_docker_storage_doctor.sh
```

如确认当前没有 build/pull 任务，并且希望优先释放非 image 的构建缓存，可只把 builder cache 全清，仍不删除 tagged image：

```bash
MODE=repair APPLY=1 DOCKER_DATA_ROOT=/data BUILDER_CACHE_UNTIL=all PRUNE_TIMEOUT=900 \
bash terminal-rl/remote/safevo_docker_storage_doctor.sh
```

如果清理后 Docker 账本和文件系统仍明显对不上，并且 doctor 显示大量 overlay2 目录无法从 layerdb/lower 链到达，先做只读 orphan 审计：

```bash
DOCKER_DATA_ROOT=/data \
python3 terminal-rl/remote/docker_overlay2_orphan_audit.py --docker-root /data --top-n 80
```

只有在确认 pool_server/watchdog/Docker 都停掉之后，才允许把候选目录移动到 quarantine。该步骤不是删除，便于回滚：

```bash
systemctl stop docker-watchdog || true
pkill -f run_pool_server_pu_v2.sh || true
systemctl stop docker

python3 terminal-rl/remote/docker_overlay2_orphan_audit.py \
  --docker-root /data \
  --quarantine \
  --quarantine-dir /data/overlay2.orphan-quarantine.$(date +%Y%m%d_%H%M%S)

systemctl start docker
df -h /data
docker info
```

如果 Docker 启动或 image/build 验证失败，停 Docker 后将 quarantine 目录里的 `overlay2/*` 和 `l/*` 移回 `/data/overlay2/` 即可回滚。

如需更重的统计，再显式打开慢操作；`docker system df -v` 和 `du` 在 layer 很多时可能较慢：

```bash
DOCKER_DATA_ROOT=/data \
python3 terminal-rl/remote/docker_storage_gc.py --diagnose-only --run-docker-df --run-du
```

注意：`/data/overlay2` 是 Docker overlay2 storage driver 的核心目录，image layer、build cache 关联 layer、container writable layer 都可能落在这里。看到 overlay2 占比大不等于可以直接删 overlay2 子目录；在线手删 overlay2 很容易破坏 Docker 元数据。

默认 GC 不删除已 tag 的旧 image，只会清理 stopped containers、dangling volumes、dangling images、builder cache。预览旧 image 候选：

```bash
DOCKER_DATA_ROOT=/data \
DOCKER_GC_DRY_RUN=1 \
DOCKER_GC_TRIGGER_USED_PCT=85 \
DOCKER_GC_TARGET_USED_PCT=70 \
DOCKER_GC_KEEP_PATTERNS='ghcr.io/laude-institute/t-bench/*,ubuntu:*,python:*' \
python3 terminal-rl/remote/docker_storage_gc.py
```

确认确实需要删除旧 image 后，必须显式开启：

```bash
DOCKER_DATA_ROOT=/data \
DOCKER_GC_TRIGGER_USED_PCT=85 \
DOCKER_GC_TARGET_USED_PCT=70 \
DOCKER_GC_DELETE_OLD_IMAGES=1 \
DOCKER_GC_KEEP_PATTERNS='ghcr.io/laude-institute/t-bench/*,ubuntu:*,python:*' \
python3 terminal-rl/remote/docker_storage_gc.py
```

该脚本不会删除任何仍被 Docker container 引用的 image；默认还会保护匹配白名单的基础镜像。开启 `DOCKER_GC_DELETE_OLD_IMAGES=1` 后，它才会按 image created time 从旧到新删除未引用 image，到目标水位即停止。

如果日志里明确出现 `/data/overlay2/... no space left on device`：

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

# 默认保守清理：stopped container、unused network、旧 build cache、dangling image
bash terminal-rl/remote/fix_docker_overlay2_no_space.sh

# 空间仍不足时：删除所有未被容器引用的 image，后续可能需要重新 build/pull
AGGRESSIVE=1 bash terminal-rl/remote/fix_docker_overlay2_no_space.sh

# dockerd 已经不响应时
sudo RESTART_DOCKER=1 AGGRESSIVE=1 bash terminal-rl/remote/fix_docker_overlay2_no_space.sh
```

注意：`docker system df` 在镜像、layer、BuildKit cache 很多或 dockerd 元数据锁竞争时可能卡很久。上述 overlay2 修复脚本和新版 `cleanup_docker_cache.sh` 默认不会执行重型 `docker system df`；如确实需要统计，可显式设置：

```bash
RUN_HEAVY_DF=1 bash terminal-rl/remote/fix_docker_overlay2_no_space.sh
```

### B3. 修复后验证

```bash
timeout 10 docker info >/dev/null && echo "[OK] docker"
test -f /etc/seta_build_proxy.env && echo "[OK] proxy env"
systemctl is-active docker-watchdog && echo "[OK] watchdog"

docker run --rm ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624 \
  sh -c 'apt-get update >/dev/null && echo "[OK] apt through proxy"'
```

重启 pool_server：

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

nohup bash terminal-rl/remote/run_pool_server_pu_v2.sh \
  > /tmp/cpu_pool.log 2>&1 &
echo $! > /tmp/cpu_pool.pid

curl --noproxy '*' http://127.0.0.1:18081/healthz
```

## 常见问题排查

### 1. `systemctl restart docker` 卡住

不要继续反复运行 `systemctl restart docker`。使用：

```bash
sudo env DOCKER_DATA_ROOT=/data bash terminal-rl/remote/restart_docker_force.sh
```

如果同时伴随 proxy/build 失败，直接运行完整恢复：

```bash
sudo env DOCKER_DATA_ROOT=/data bash terminal-rl/remote/fix_dockerd_and_proxy.sh
```

### 2. `setup_new_worker.sh` 卡在 Docker already installed 后

常见原因是脚本探测 Docker daemon 时执行了 `docker info`，而 dockerd 已经假死或 `/var/run/docker.sock` 是 stale socket。新版脚本已经给 `docker info` 加了 timeout，并会进入 Step 4 的 restart/force-restart fallback。

如果你正在旧脚本里卡住：

```bash
# 在卡住的终端 Ctrl-C；如果 Ctrl-C 不生效，另开终端执行：
pkill -f 'terminal-rl/remote/setup_new_worker.sh' || true

cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL
sudo env DOCKER_DATA_ROOT=/data \
  PROXY_URL=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128 \
  bash terminal-rl/remote/setup_new_worker.sh
```

如果想先单独恢复 Docker：

```bash
sudo env DOCKER_DATA_ROOT=/data bash terminal-rl/remote/restart_docker_force.sh
timeout 10 docker info
```

### 3. `apt-get update` 在 Docker build 内 timeout

原因通常是 Ubuntu apt 不读取普通 `HTTP_PROXY` 环境变量。重新注入 base image apt proxy：

```bash
sudo bash terminal-rl/remote/prebuild_proxied_base_images.sh
```

验证：

```bash
docker run --rm ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624 \
  sh -c 'cat /etc/apt/apt.conf.d/95proxies && apt-get update >/dev/null'
```

### 4. GPU worker 看到 `/allocate` 502

先在 CPU worker 上确认 pool_server：

```bash
curl --noproxy '*' http://127.0.0.1:18081/healthz
curl --noproxy '*' http://127.0.0.1:18081/status | python3 -m json.tool
```

如果不通，重启 pool_server；如果 Docker 不通，执行场景 B。

### 5. `port 18081 already in use`

```bash
ss -tlnp | grep ':18081 '
kill -9 <PID>
bash terminal-rl/remote/run_pool_server_pu_v2.sh
```

### 6. Docker 网络或容器残留太多

```bash
docker ps -a | head
docker network ls | wc -l
bash terminal-rl/remote/cleanup_docker_cache.sh
```

### 6.1 `/data/overlay2` no space left on device

**症状**：pool_server `/reset` 返回 500，日志中 `docker compose build` 失败，包含：

```text
mkdir /data/overlay2/<id>: no space left on device
```

**原因**：Docker data root 所在分区满了，通常由 on-demand build 产生的 BuildKit cache、task image、stopped container、dangling network/volume 累积导致。镜像太多是可能原因之一，但不是唯一原因；BuildKit cache 和 overlay2 layer 元数据同样常见。

**处理**：

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

# 先判因：默认只读，报告写到 tmp_doc_latest/docker_storage_doctor/<host>/<run_id>/
DOCKER_DATA_ROOT=/data bash terminal-rl/remote/safevo_docker_storage_doctor.sh

# 再做保守修复：不删除 tagged images
MODE=repair APPLY=1 DOCKER_DATA_ROOT=/data bash terminal-rl/remote/safevo_docker_storage_doctor.sh

# 如果已经确认可以删除所有未引用 image，再使用旧 emergency 脚本的 AGGRESSIVE 模式
AGGRESSIVE=1 bash terminal-rl/remote/fix_docker_overlay2_no_space.sh
```

如果脚本输出停在 `Docker system df`，说明用的是旧脚本或手动执行了重型统计。新版脚本默认跳过该步骤；也可以直接 Ctrl-C 后重新运行上面的命令。

**预防**：新版 `docker_watchdog_v2.sh` 已加入 Docker data-root 磁盘压力监控，不调用 `docker system df`，只用 `df` 快速判断容量并渐进清理。常用阈值：

```bash
DISK_WARN_PCT=80
DISK_EMERGENCY_PCT=92
DISK_MIN_FREE_GB=20
DISK_INODE_WARN_PCT=80
DISK_INODE_EMERGENCY_PCT=90
DISK_BUILD_CACHE_UNTIL=12h
WATCHDOG_AGGRESSIVE_IMAGE_PRUNE=0
WATCHDOG_DOCKER_STORAGE_GC=1
DOCKER_GC_TRIGGER_USED_PCT=85
DOCKER_GC_TARGET_USED_PCT=70
DOCKER_GC_DELETE_OLD_IMAGES=0
DOCKER_GC_KEEP_PATTERNS='ghcr.io/laude-institute/t-bench/*,ubuntu:*,python:*'
POOL_STOP_ON_DISK_EMERGENCY=1
```

如果 worker 磁盘较小且允许自动删除未使用 image，推荐保留 `WATCHDOG_DOCKER_STORAGE_GC=1`。旧的无差别 unused-image prune 仍可通过以下方式启用，但不推荐作为默认：

```bash
WATCHDOG_DOCKER_STORAGE_GC=0
WATCHDOG_AGGRESSIVE_IMAGE_PRUNE=1
```

safevo/rlinfra 可共用同一脚本，仅通过阈值区分。safevo 900G 更容易触顶，建议更早触发；rlinfra 1T 可使用相同或略宽松阈值：

```bash
# safevo
DOCKER_DATA_ROOT=/data
DOCKER_GC_TRIGGER_USED_PCT=85
DOCKER_GC_TARGET_USED_PCT=70
DOCKER_GC_MIN_FREE_GB=120
DOCKER_GC_DELETE_OLD_IMAGES=0

# rlinfra
DOCKER_DATA_ROOT=/data
DOCKER_GC_TRIGGER_USED_PCT=88
DOCKER_GC_TARGET_USED_PCT=75
DOCKER_GC_MIN_FREE_GB=100
DOCKER_GC_DELETE_OLD_IMAGES=0
```

也可以用 cron 做独立巡检，和 watchdog 互补：

```cron
*/15 * * * * cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL && DOCKER_DATA_ROOT=/data DOCKER_GC_TRIGGER_USED_PCT=85 DOCKER_GC_TARGET_USED_PCT=70 DOCKER_GC_LOG_FILE=/var/log/openclaw-docker-gc.log python3 terminal-rl/remote/docker_storage_gc.py
```

systemd timer 示例：

```ini
# /etc/systemd/system/openclaw-docker-gc.service
[Unit]
Description=OpenClaw Docker storage GC

[Service]
Type=oneshot
WorkingDirectory=/mnt/shared-storage-user/puyuan/code/OpenClaw-RL
Environment=DOCKER_DATA_ROOT=/data
Environment=DOCKER_GC_TRIGGER_USED_PCT=85
Environment=DOCKER_GC_TARGET_USED_PCT=70
Environment=DOCKER_GC_DELETE_OLD_IMAGES=0
Environment=DOCKER_GC_KEEP_PATTERNS=ghcr.io/laude-institute/t-bench/*,ubuntu:*,python:*
ExecStart=/usr/bin/python3 terminal-rl/remote/docker_storage_gc.py

# /etc/systemd/system/openclaw-docker-gc.timer
[Unit]
Description=Run OpenClaw Docker storage GC periodically

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min
Unit=openclaw-docker-gc.service

[Install]
WantedBy=timers.target
```

新版 `pool_server` 也会在 `/allocate` 和 `/reset` 前做 Docker data-root admission check。默认阈值：

```bash
WORKER_MIN_DOCKER_FREE_GB=50
WORKER_MAX_DOCKER_USED_PCT=85
WORKER_MAX_DOCKER_INODE_PCT=80
```

超过阈值时，`/healthz` 返回 503，`/allocate`/`/reset` 返回 `WORKER_DOCKER_DISK_PRESSURE`，避免继续 build 写爆 `/data`。

### 7. watchdog 没有启动

```bash
sudo cp terminal-rl/remote/docker-watchdog.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now docker-watchdog
journalctl -u docker-watchdog -n 50 --no-pager
```

### 8. 已废弃脚本

当前 `terminal-rl/remote/` 下没有删除任何脚本。历史旧版脚本如 `setup.sh`、`run_pool_server.sh` 等如果存在，应视为 deprecated，仅以 `terminal-rl/remote/README.md` 和本文档列出的 active scripts 为准。
