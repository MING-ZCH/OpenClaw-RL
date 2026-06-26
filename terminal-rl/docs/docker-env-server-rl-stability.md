# Docker Env Server RL 稳定性问题记录

## Overview

本文记录近期 `docker-env-server` 在 RL 长时间训练中暴露的关键稳定性问题，以及已经落地的核心修复点。内容基于当前代码仓库的近期 commit、当前 working tree diff 和对应模块实现整理，重点覆盖 `safevo`/`rlinfra` 上的 Docker worker、`pool_server`、`TerminalEnv` 和 watchdog 生命周期管理。

相关变更主要来自：

- `294f9541 fix(pu): Stabilize Docker reset timeouts for RL rollouts`
- `e0e81b9f fix(pu): Stabilize reset timeout and Docker cleanup`
- `8c2b8313 fix(pu): Stabilize Terminal-RL reset workflow`
- `72931a7 fix(pu): Force container recreation to fix Docker API hang`
- 当前 working tree 的 P0 优化：`evaluate` 降级返回、`close backlog` timeout 拆分
- 2026-06-11 运行复盘后的 P0 优化：fast close 优先 Docker 物理删除、cleanup shield/detach、pool-server 周期性 orphan Docker sweep
- 2026-06-12 运行复盘后的 P0 优化：多容器 compose 固定名残留清理、lease-aware sweep、按 task 的并发上限 override

## 目录

1. 长生命周期容器导致 Docker API hang
2. reset coroutine 被重复 await
3. reset/image build timeout 预算过短
4. close backlog 影响长期训练
5. evaluate timeout/parse 失败造成重试放大
6. watchdog 孤儿容器回收
7. 2026-06-11 `Up 3 hours` 容器复盘
8. 2026-06-12 多容器 compose 固定名冲突复盘
9. 验证要点

## 1. 长生命周期容器导致 Docker API hang

### 问题现象

训练运行一段时间后，`docker ps` 中可见较早启动的 task 容器长期保持 `Up`；后续 reset 出现长时间卡住或 timeout。表面看像是任务仍在运行，实际关键卡点可能发生在 Docker API 查询阶段。

### 根因分析

关键定位点在 `terminal-rl/remote/docker_compose_utils.py:522`：

```python
# terminal-rl/remote/docker_compose_utils.py:522
container = compose_manager._client.containers.get(container_name)
```

近期定位结论是：`compose up` 本身可能较快完成，但长期运行的容器会让后续 `containers.get(container_name)` 调 Docker daemon 时变慢甚至卡住。也就是说，reset 的瓶颈不一定是 build 或 compose，而可能是复用老容器后的 Docker API 状态查询。

### 解决方案

在每次 `TerminalEnv.reset()` 创建新 trial 前，先按精确容器名强制移除旧容器，避免复用长期运行容器。

```python
# terminal-rl/remote/terminal_env.py:439, TerminalEnv.reset()
def _sync_reset() -> tuple[str, list[dict[str, Any]]]:
    container_name = f"{self._task_spec.task_name}.{self._run_ctx.uid}.slime-run"
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            timeout=5,
            capture_output=True,
            check=False,
        )
        logger.info(
            "Forced container recreation for %s to avoid Docker API slowdown",
            container_name,
        )
    except Exception as e:
        logger.debug("Container force-remove failed (may not exist): %s", e)
```

### 影响范围

- reset 更倾向于使用新容器，降低 Docker daemon 查询老容器元数据导致的长尾延迟。
- 对已经完成但未清理的旧容器更敏感，需要结合 force cleanup/watchdog 清理孤儿容器。

## 2. reset coroutine 被重复 await

### 问题现象

`cpu_pool.log` 中出现：

```text
RuntimeError: cannot reuse already awaited coroutine
```

随后 reset 失败率升高，worker slot 可能滞留在 `resetting` 或被错误清理。

### 根因分析

`asyncio.wait_for()` 如果直接包裸 coroutine，timeout 后会 cancel 这个 coroutine；裸 coroutine 一旦被 await/cancel 过，后续不能再次 await。之前的 progressive wait 路径可能复用同一个 coroutine，触发 `cannot reuse already awaited coroutine`。

涉及模块：

- `terminal-rl/remote/pool_server.py`
- `WorkerPool._run_reset_once()`

### 解决方案

把 reset coroutine 显式包装成 `asyncio.Task`，warning 阶段只等待 task 是否完成，真正超时阶段用 `asyncio.shield(reset_task)`，避免 `wait_for` 直接消费裸 coroutine。

```python
# terminal-rl/remote/pool_server.py:1185, WorkerPool._run_reset_once()
reset_task = asyncio.create_task(
    run_slot.env.reset(
        task_meta=task_meta,
        task_spec=task_spec,
        run_ctx=run_ctx,
        timeouts=timeouts,
    )
)

done, _ = await asyncio.wait({reset_task}, timeout=warn_timeout)

user_msg, tool_schemas = await asyncio.wait_for(
    asyncio.shield(reset_task),
    timeout=remaining_timeout,
)
```

超时后显式 cancel，并注册 callback 消费取消异常，避免后台 task 异常泄漏：

```python
# terminal-rl/remote/pool_server.py:1231
except asyncio.TimeoutError as exc:
    if reset_task.done():
        raise
    is_timeout_drop = True
    reset_task.cancel()
    reset_task.add_done_callback(_consume_cancelled_reset_task)
    raise TimeoutError(
        f"WORKER_RESET_TIMEOUT lease_id={run_lease_id} after {reset_timeout:.1f}s"
    ) from exc
```

### 影响范围

- 修复 reset timeout 路径本身导致的二次异常。
- timeout 后 slot 会进入明确的 drop/cleanup 路径，而不是因为 coroutine 状态错误进入不一致状态。

## 3. reset/image build timeout 预算过短

### 问题现象

部分任务 reset 阶段包含 image build、compose startup、容器 runtime 限制设置等操作。Docker 压力较高或首次 build 时，旧 timeout 过短会把正常慢操作误判成失败，导致训练中断和 cleanup 风暴。

### 根因分析

timeout 需要同时覆盖：

- image prepare/build
- reset session / compose startup
- Docker API 长尾
- server 和 RL client 的 HTTP 等待时间

如果 RL 端传入较短 timeout override，worker 端不能被客户端意外缩短关键 reset 预算。

### 解决方案

worker 端对关键 timeout override 设置 floor，避免旧客户端把 `ensure_image`/`reset_session` 降到不合理的小值。

```python
# terminal-rl/remote/pool_server.py:28, _parse_timeout_overrides()
def _pick(key: str, default: float, *, minimum: float | None = None) -> float:
    ...
    if minimum is not None and value < minimum:
        logger.debug(
            "Raising client timeout override %s=%.1fs to worker floor %.1fs",
            key,
            value,
            minimum,
        )
        return minimum

return TaskTimeouts(
    ensure_image=_pick("ensure_image", base.ensure_image, minimum=base.ensure_image),
    reset_session=_pick("reset_session", base.reset_session, minimum=base.reset_session),
    close_session=_pick("close_session", base.close_session),
    eval=_pick("eval", base.eval),
)
```

启动脚本把 reset 预算调大，且使 stale repair 晚于 reset operation timeout：

```bash
# terminal-rl/remote/start_server.sh:52
export ENSURE_IMAGE_TIMEOUT="${ENSURE_IMAGE_TIMEOUT:-1200}"
export RESET_SESSION_TIMEOUT="${RESET_SESSION_TIMEOUT:-600}"
export WORKER_RESET_OPERATION_TIMEOUT="${WORKER_RESET_OPERATION_TIMEOUT:-1920}"
export WORKER_RESETTING_TTL="${WORKER_RESETTING_TTL:-2100}"
```

RL 端 HTTP reset timeout 也按阶段预算加 buffer：

```python
# terminal-rl/generate.py:2225
default_reset_http_timeout = (
    float(timeouts.ensure_image) + float(timeouts.reset_session) + 300.0
)
reset_http_timeout = _env_float("ENV_RESET_HTTP_TIMEOUT", default_reset_http_timeout)
```

### 影响范围

- 减少正常 image build / compose 慢操作被误杀。
- `WORKER_RESETTING_TTL` 晚于 reset timeout，避免 repair 线程提前清理仍在合法 reset 的 slot。

## 4. close backlog 影响长期训练

### 问题现象

长时间训练中，`pending_closes` 增多时，close 任务可能排队等待 semaphore。旧实现用一个 `WORKER_CLOSE_TASK_TIMEOUT` 同时覆盖排队等待和真正 `env.close()`，在 backlog 场景下容易误判 close 超时。

### 根因分析

`close` 有两个不同阶段：

1. 等待 `_close_sem`：backlog 排队时间
2. 执行 `run_slot.env.close()`：真实 session 关闭时间

两者共用一个 timeout 会把“排队等待过久”和“close 本身卡住”混在一起，影响后续判断和状态观测。

### 解决方案

当前 P0 优化将 close timeout 拆成 queue/session 两段，并在 `/status` 暴露。

```python
# terminal-rl/remote/pool_server.py:543, WorkerPool.__init__()
legacy_close_task_timeout = _env_float(
    "WORKER_CLOSE_TASK_TIMEOUT",
    max(30.0, float(default_timeouts.close_session) + 30.0),
)
self.close_queue_timeout = _env_float(
    "WORKER_CLOSE_QUEUE_TIMEOUT", legacy_close_task_timeout
)
self.close_session_timeout = _env_float(
    "WORKER_CLOSE_SESSION_TIMEOUT",
    max(30.0, float(default_timeouts.close_session)),
)
self.close_task_timeout = self.close_queue_timeout + self.close_session_timeout
```

```python
# terminal-rl/remote/pool_server.py:753
await asyncio.wait_for(
    self._close_sem.acquire(), timeout=self.close_queue_timeout
)
...
await asyncio.wait_for(
    run_slot.env.close(), timeout=self.close_session_timeout
)
```

启动脚本同步默认值：

```bash
# terminal-rl/remote/start_server.sh:48
export WORKER_CLOSE_TASK_TIMEOUT="${WORKER_CLOSE_TASK_TIMEOUT:-45}"
export WORKER_CLOSE_QUEUE_TIMEOUT="${WORKER_CLOSE_QUEUE_TIMEOUT:-${WORKER_CLOSE_TASK_TIMEOUT}}"
export WORKER_CLOSE_SESSION_TIMEOUT="${WORKER_CLOSE_SESSION_TIMEOUT:-60}"
```

### 影响范围

- close backlog 不会直接挤占真实 close session 的 timeout。
- `/status` 可区分 `pending_closes`、`close_queue_timeout`、`close_session_timeout`，便于判断是 backlog 还是 Docker close 卡住。

## 5. evaluate timeout/parse 失败造成重试放大

### 问题现象

训练后期可能出现 `/evaluate` 500、RL client 重试、rollout 延迟放大的组合。部分任务的 eval 测试本身可能耗时较长，或者 parser 没解析到结果；这类失败不应该让整个 env-server 进入错误风暴。

### 根因分析

旧路径中，terminal test timeout 或 parser exception 会抛 `RuntimeError`，`pool_server` 的 `/evaluate` 端点会返回 HTTP 500。RL client 看到 500 后可能按重试策略继续请求，放大已经拥塞的 env-server。

涉及模块：

- `terminal-rl/remote/terminal_env.py`
- `terminal-rl/remote/pool_server.py`
- `terminal-rl/env_client.py`

### 解决方案

当前 P0 优化把 terminal test eval 的可预期失败降级成 `score=0.0`，并通过 `details.reason` 标注原因。

```python
# terminal-rl/remote/terminal_env.py:671, TerminalEnv.evaluate()
except TimeoutError as exc:
    logger.warning(
        "Evaluation tests timed out for task=%s after %.1fs.",
        task_name,
        test_timeout_sec,
    )
    self._last_eval = {
        "mode": "terminal_tests",
        "score": 0.0,
        "reason": "eval_timeout",
        "task": task_name,
        "timeout_sec": test_timeout_sec,
        "error": str(exc),
    }
    return 0.0
```

parser 失败和无结果同样返回 0 分：

```python
# terminal-rl/remote/terminal_env.py:690
except Exception as exc:
    self._last_eval = {
        "mode": "terminal_tests",
        "score": 0.0,
        "reason": "eval_parse_failed",
        "task": task_name,
        "parser": type(self._parser).__name__,
        "error": str(exc),
    }
    return 0.0

if not parser_results:
    self._last_eval = {
        "mode": "terminal_tests",
        "score": 0.0,
        "reason": "eval_no_results",
        "task": task_name,
        "parser": type(self._parser).__name__,
        "total": 0,
        "passed": 0,
    }
    return 0.0
```

`pool_server` 会把 details 返回给 RL client：

```python
# terminal-rl/remote/pool_server.py:2815
score, details = await POOL.evaluate(str(lease_id), trajectory)
payload: dict[str, Any] = {"ok": True, "score": score}
if details is not None:
    payload["details"] = details
```

RL 侧默认 evaluate retry 降到 1：

```bash
# terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh:372
ENV_EVALUATE_MAX_RETRIES="${ENV_EVALUATE_MAX_RETRIES:-1}"
```

### 影响范围

- eval timeout/parser 异常不再变成 HTTP 500 重试风暴。
- 训练仍能拿到确定 reward：失败 eval 记为 0 分。
- 需要监控 `details.reason`，如果 `eval_parse_failed` 大量出现，说明 parser 或测试输出格式仍需单独修复。

## 6. watchdog 孤儿容器回收

### 问题现象

server 重启、reset timeout、close timeout 或 worker slot 被 drop 后，Docker 层可能残留 task 容器。`docker ps` 中看到老容器 `Up`，但 pool 逻辑状态里不一定还有对应 active run。

### 根因分析

env-server 的逻辑状态和 Docker daemon 的实际容器状态不是强一致的。异常路径下可能出现：

- run slot 已从 pool 中 drop
- force cleanup 超时或被取消
- server 重启丢失内存态
- Docker 容器仍保持 `Up`

### 解决方案

watchdog 新增两类保守回收：

1. reset storm orphan reap：resetting 数量和最大 reset age 异常时，按 active/running gap 清理 unprotected idle 容器。
2. idle orphan reap：running task containers 显著多于 pool active/protected 时，清理老的 unprotected idle 容器。

```bash
# terminal-rl/remote/docker_watchdog_v2.sh:1577
reap_reset_storm_orphan_task_containers() {
    [ "${WATCHDOG_RESET_STORM_ORPHAN_REAP_ENABLED}" = "1" ] || return 0
    pool_status_is_fresh || return 0
    ...
    [ "${LAST_POOL_RESET_MAX_AGE}" -ge "${WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_RESET_AGE}" ] || return 0
    ...
    reap_unprotected_task_containers \
        "reset storm orphan gap running=${running} active=${LAST_POOL_ACTIVE} resetting=${LAST_POOL_RESETTING}" \
        "${limit}" \
        "${WATCHDOG_RESET_STORM_ORPHAN_REAP_MIN_AGE}" \
        "reset_storm_orphan" \
        1 || true
}
```

```bash
# terminal-rl/remote/docker_watchdog_v2.sh:1613
reap_idle_orphan_task_containers() {
    [ "${WATCHDOG_IDLE_REAP_ENABLED}" = "1" ] || return 0
    pool_status_is_fresh || return 0
    [ "${running}" -ge "${WATCHDOG_IDLE_REAP_MIN_CONTAINERS}" ] || return 0
    ...
    reap_unprotected_task_containers \
        "idle orphan gap running=${running} active=${LAST_POOL_ACTIVE} target=${idle_target}" \
        "${limit}" \
        "${WATCHDOG_IDLE_REAP_MIN_AGE}" \
        "idle_orphan" \
        1 || true
}
```

### 影响范围

- `docker ps` 中长期残留但不受 pool 保护的 task 容器会被逐步回收。
- 清理逻辑依赖 fresh pool status 和 protected 容器列表，避免误杀仍在执行的 active run。

## 7. 2026-06-11 `Up 3 hours` 容器复盘

### 问题现象

2026-06-11 的 `terminal-rl_qwen3-8b_8gpu_seta_dapo_nodynamic_think_mt10` 长跑中，CPU worker 上 `docker ps` 出现多个 `Up 2 hours` / `Up 3 hours` 的 task 容器，例如：

```text
105-f1574760-slime-run
1123-8a678567-slime-run
1033-e4d05e4d-slime-run
673-8d8d7971-slime-run
515-8f9b54b9-slime-run
```

这些容器看起来像还在执行，但 server 日志显示对应 run 已收到 `/close`，并且 pool 已尝试释放：

```text
2026-06-11 14:12:18 close_run requested lease=run-a7a6010a41724538 task=515:seta_env/515 ...
2026-06-11 14:13:18 Timed out closing run session run-a7a6010a41724538 after 45.0s ...
2026-06-11 14:14:48 Force cleanup timed out for run session run-a7a6010a41724538 ...

2026-06-11 14:17:32 close_run requested lease=run-f0a0b9dc725d4357 task=1033:seta_env/1033 ...
2026-06-11 14:19:47 Force cleanup timed out for run session run-f0a0b9dc725d4357 ...

2026-06-11 14:18:02 close_run requested lease=run-670c2b18c6bc4cb6 task=105:seta_env/105 ...
2026-06-11 14:20:17 Force cleanup timed out for run session run-670c2b18c6bc4cb6 ...

2026-06-11 14:18:02 close_run requested lease=run-6b9158bad0b5462c task=1123:seta_env/1123 ...
2026-06-11 14:20:17 Force cleanup timed out for run session run-6b9158bad0b5462c ...

2026-06-11 14:21:12 close_run requested lease=run-14ffd32233fb4505 task=673:seta_env/673 ...
2026-06-11 14:23:42 Force cleanup timed out for run session run-14ffd32233fb4505 ...
```

同一时间段 server 日志中还有大量：

```text
Timed out closing run session ... after 45.0s
Force cleanup timed out ... timeout=90.0s
```

这说明问题是系统性 close/cleanup 长尾，而不是某一个 task 的测试逻辑导致。

RL 端最后并未表现为生成服务崩溃。`train.log` tail 仍有 SGLang `/generate` 200 OK 和 rollout tool calls。主要可见异常是 `/evaluate` 500，原因包括测试超时、`PytestParser` 无法解析输出、任务内下载依赖超时等。这会影响 reward/eval 质量，但不是 `Up 3 hours` Docker 容器残留的直接根因。

### 第一性原理诊断

容器生命周期要满足一个基本不变量：

```text
只要 run slot 已从 pool 内存态释放，对应 Docker 容器也必须最终被删除；
如果一次删除失败，系统必须能从 Docker 真实状态重新发现并重试。
```

旧路径违反了这个不变量：

1. `WorkerPool.close_run()` 在无 in-flight op 时先 `_pop_run_slot_locked()`，把 lease 从 `_run_to_task` 和 task slot 中移除。
2. 后续 close 任务再执行 `TerminalEnv.close()` 和 Docker cleanup。
3. 如果 close 排队、session close、`TerminalToolkit.cleanup`、session drain、`terminal.stop()` 或 `docker rm -f` 卡住，外层 timeout 会结束这次 close。
4. run slot 已经从内存索引消失，`repair_stale_runs` / `repair_close_requested_runs` 只能看到 pool 内存态，看不到 Docker daemon 里仍然 `Up` 的容器。
5. 旧日志里的 “Watchdog/preflight cleanup will remove any orphan Docker objects” 只能依赖重启/退出或外部 watchdog；长跑中的 pool server 本身没有按真实 `docker ps -a` 做最终一致性扫描。

本质上，旧实现是 “best-effort delete after losing ownership”。一旦这一次 best-effort 删除超时，就丢失了重试所需的强引用。

### 已落地修复

#### 7.1 fast close 先删 Docker，再跳过易卡的 toolkit drain

`TerminalEnv.close()` 在 `TERMINAL_ENV_FAST_CLOSE=1` 时优先执行 Docker 物理清理：

```python
# terminal-rl/remote/terminal_env.py
if fast_close and terminal is not None:
    await _run_force_cleanup("fast_close")
```

fast close 下不再等待 `TerminalToolkit.cleanup` 和 `_drain_toolkit_sessions`：

```python
if toolkit is not None:
    if fast_close:
        logger.warning(
            "Fast close enabled for %s; skipping TerminalToolkit.cleanup "
            "and session drain; relying on direct Docker cleanup.",
            trial_name,
        )
```

这把 close 的关键目标从“优雅清理所有终端资源”调整为“先确保 Docker 容器被删”。对 RL worker 来说，后者是长期稳定性的硬约束。

#### 7.2 Docker cleanup 使用专用 executor 并 shield

Docker 删除不再走默认 `asyncio.to_thread` executor，而是走专用线程池：

```python
# terminal-rl/remote/terminal_env.py
def _docker_cleanup_executor() -> ThreadPoolExecutor:
    workers = max(1, _env_int("TERMINAL_ENV_DOCKER_CLEANUP_WORKERS", 8))
    return ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="openclaw-docker-cleanup",
    )
```

异步 cleanup 使用 `asyncio.shield(fut)`：

```python
async def _force_remove_docker_objects_async(...):
    fut = loop.run_in_executor(_docker_cleanup_executor(), partial(...))
    try:
        await asyncio.shield(fut)
    except asyncio.CancelledError:
        logger.warning(
            "Docker cleanup detached after cancellation ... cleanup will continue"
        )
        _attach_detached_cleanup_logger(fut, trial_name=trial_name, reason=reason)
        raise
```

这样即使外层 `wait_for` 超时并取消 close coroutine，已经进入 executor 的 `docker rm -f` 仍会继续执行并记录 detached completion。

#### 7.3 pool server 增加周期性 Docker orphan sweep

新增 `force_remove_orphan_docker_objects()`，直接从 Docker daemon 枚举真实状态：

```bash
docker ps -a --format "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}"
```

只选择满足以下条件的容器：

- 名字匹配 `^[0-9]+-[A-Za-z0-9]{8}-slime-run$`，或名字以 `-slime-run` 结尾且 image 以 `tb__` 开头。
- 不在 pool 当前 active container name 集合中。
- `docker ps` status 年龄超过 `WORKER_ORPHAN_DOCKER_SWEEP_MIN_AGE`。

`WorkerPool.periodic_reap()` 每轮调用：

```python
# terminal-rl/remote/pool_server.py
await self._maybe_cleanup_orphan_docker_containers()
```

默认参数：

```bash
WORKER_ORPHAN_DOCKER_SWEEP=1
WORKER_ORPHAN_DOCKER_SWEEP_INTERVAL=60
WORKER_ORPHAN_DOCKER_SWEEP_MIN_AGE=600
WORKER_ORPHAN_DOCKER_SWEEP_MAX_REMOVE=128
WORKER_ORPHAN_DOCKER_SWEEP_TIMEOUT=30
TERMINAL_ENV_DOCKER_CLEANUP_WORKERS=8
```

这使 pool server 自身具备最终一致性能力：即使某次 close 已经丢失 run slot，只要 Docker 里还残留老 task 容器，后续周期扫描仍能重新发现并删除。

#### 7.4 reset/close 并发返回语义修正

完整生命周期测试暴露了一个相关边界：`close` 在 `reset` 进行中被请求时，`_run_reset_once()` finally 会把 in-flight 降到 0 并按 close 请求移除 slot；外层 `reset()` 随后只想缓存 reset 结果，却因为 slot 已被删除报 `WORKER_RESET_STALE`。

修复后，如果 reset future 已成功完成但 lease 已被 close 移除，直接返回 reset 结果，不再把它反报为 500：

```python
# terminal-rl/remote/pool_server.py
except KeyError:
    logger.info(
        "Reset completed after lease=%s was already removed; "
        "returning reset result without caching it.",
        run_lease_id,
    )
    return result
```

这避免了 close/reset 正常竞争被误判成 reset 失败，从而减少额外 cleanup 风暴。

### 启动脚本默认值

`terminal-rl/remote/start_server.sh` 和 `terminal-rl/remote/run_pool_server_pu_v2.sh` 已同步导出：

```bash
TERMINAL_ENV_DOCKER_CLEANUP_WORKERS="${TERMINAL_ENV_DOCKER_CLEANUP_WORKERS:-8}"
WORKER_ORPHAN_DOCKER_SWEEP="${WORKER_ORPHAN_DOCKER_SWEEP:-1}"
WORKER_ORPHAN_DOCKER_SWEEP_INTERVAL="${WORKER_ORPHAN_DOCKER_SWEEP_INTERVAL:-60}"
WORKER_ORPHAN_DOCKER_SWEEP_MIN_AGE="${WORKER_ORPHAN_DOCKER_SWEEP_MIN_AGE:-600}"
WORKER_ORPHAN_DOCKER_SWEEP_MAX_REMOVE="${WORKER_ORPHAN_DOCKER_SWEEP_MAX_REMOVE:-128}"
WORKER_ORPHAN_DOCKER_SWEEP_TIMEOUT="${WORKER_ORPHAN_DOCKER_SWEEP_TIMEOUT:-30}"
```

重启 pool server 后，启动日志应能看到：

```text
force_cleanup=... workers=8 ...
orphan_sweep=1 interval=60s min_age=600s max_remove=128 timeout=30s
```

### 预期效果

- `Up 2 hours` / `Up 3 hours` 的历史容器会在 pool server 重启 preflight 或运行中 orphan sweep 后被回收。
- 单次 close timeout 不再是最终失败；Docker 删除会继续跑，且周期 sweep 会兜底。
- active run 的容器不会被 sweep 误删，因为 sweep 会排除 pool 当前记录的 active container name，并设置 10 分钟默认最小年龄。
- 仍需单独关注大量 `/evaluate` 500 或 `eval_parse_failed`，它们影响 reward 质量，但不应再阻断 Docker lifecycle cleanup。

## 8. 2026-06-12 多容器 compose 固定名冲突复盘

### 问题现象

2026-06-12 复盘发现，部分残留容器不是 `*-slime-run` client，而是 SETA task 的 compose 辅助服务，例如：

```text
tb__892__remote-server
tb__1133__web-server
tb__1133__db-server
tb__1133__app-server
tb__1133__workstation-1
tb__1133__workstation-2
tb__1133__firewall-host
```

这些容器可长期 `Up 4 hours` / `Up 6 hours`。后续同 task reset 会遇到 `container name already in use` 或 `Pool overlaps with other one on this address space`，最终表现为 `/reset` 500、`RUN_SLOTS_EXHAUSTED`、worker pressure。

### 根因分析

SETA 中少数 task 的 `docker-compose.yaml` 对非 client 服务使用固定容器名，例如 `container_name: ${T_BENCH_TASK_DOCKER_NAME_PREFIX}__remote-server`；1133 这类任务还使用固定 `ipam.subnet`。

这类 compose 文件本质不支持同 task 多并发：

```text
同一个 task 并发 8 条 rollout
=> 8 个 compose project 同时创建同名 tb__<task>__service 或同 subnet network
=> Docker compose up 中途失败
=> 可能已经创建部分 service/network/volume
=> 旧 cleanup 只删除 client 容器，辅助服务残留
```

进程是否活跃不是可靠判断标准：`sshd`、`nginx`、`postgres` 或 `sleep infinity` 都可能长期活着。真正的保留条件应是该 Docker 对象是否仍属于 pool 当前 active lease / active task / active compose project。

### 已落地修复

#### 8.1 close/reset 失败路径执行 compose down

`TerminalEnv` 现在记录 compose 文件路径，并在 close、fast close、force cleanup 和 `reset_start_failed` 路径执行：

```bash
docker compose -p <project> -f <docker-compose.yaml> down --remove-orphans -v --timeout 5
```

随后仍保留原有精确 client 删除和 network 删除作为兜底。这样 `compose up` 半途失败后，已创建的 sibling services、network、volume 会被主动回收。

相关开关：

```bash
TERMINAL_ENV_COMPOSE_DOWN_CLEANUP=1
TERMINAL_ENV_COMPOSE_DOWN_ON_CLOSE=1
TERMINAL_ENV_COMPOSE_DOWN_SERVICE_TIMEOUT=5
TERMINAL_ENV_FIXED_SERVICE_CLEANUP_MAX_REMOVE=64
```

#### 8.2 pool-server sweep 从 client-only 扩展为 lease-aware

周期性 orphan sweep 现在会传入：

- active client container names
- active compose project names
- active task ids

清理对象包括：

- 不在 active 集合内的 stale `*-slime-run` client。
- compose project 已不活跃的 task 容器。
- `tb__<task_id>__*` 固定名辅助服务，但前提是该 `task_id` 没有 active lease，也没有 running client。
- stale compose network / volume。

相关开关：

```bash
WORKER_ORPHAN_DOCKER_SWEEP=1
WORKER_ORPHAN_DOCKER_SWEEP_RESOURCES=1
WORKER_ORPHAN_DOCKER_SWEEP_RESOURCE_MAX_REMOVE=128
```

`/status` 也增加了 `active_project_names`、`active_task_ids`，用于判断 cleanup 是否有误删风险。

#### 8.3 默认并发 8，但已知 compose-unsafe task 串行

RL 端 `N_SAMPLES=8` 需要同一 prompt 产生 8 条独立 rollout。正确语义是 8 个隔离 env 实例，而不是 8 条 rollout 共用一个容器状态。

大多数 SETA task 可以并发，所以全局默认保持：

```bash
WORKER_MAX_RUNS_PER_TASK=8
```

但已知 compose-unsafe task 默认串行：

```bash
WORKER_SERIAL_TASK_IDS=892,1133
WORKER_TASK_MAX_RUNS_OVERRIDES=
WORKER_AUTO_SERIALIZE_UNSAFE_COMPOSE=0
```

`WorkerPool.allocate()` 会按 task 计算有效上限：

```text
WORKER_TASK_MAX_RUNS_OVERRIDES 命中 => 使用显式值
WORKER_SERIAL_TASK_IDS 命中 => max_runs=1
WORKER_AUTO_SERIALIZE_UNSAFE_COMPOSE=1 且 compose 含固定 service container_name/ipam/subnet => max_runs=1
否则 => WORKER_MAX_RUNS_PER_TASK
```

`/status` 中每个 task 会显示 `max_runs`，例如 892/1133 应显示 `max_runs=1`，普通 task 应显示 `max_runs=8`。

#### 8.4 router 不把 task-local 满载当作 worker 全局不健康

`RUN_SLOTS_EXHAUSTED` 表示某个 worker 上“该 task 的并发上限已满”，不是整个 worker 坏了。router 现在遇到该错误会尝试下一个 worker，但不会把当前 worker 标记为全局 unhealthy，避免 892/1133 的串行策略拖慢其他 task。

### 预期效果

- 892/1133 这类固定名多容器任务不再在单 worker 上并发互撞。
- 普通可并发 task 仍可使用 `max_runs_per_task=8`，满足 RL 端同 prompt 8 条 rollout 的吞吐需求。
- 即使 compose up 半途失败，也会通过 `compose down` 和 lease-aware sweep 最终回收固定名辅助服务、network、volume。
- 若发现新的 compose-unsafe task，优先追加到 `WORKER_SERIAL_TASK_IDS` 或 `WORKER_TASK_MAX_RUNS_OVERRIDES`，而不是全局降并发。

## 9. 验证要点

修复后建议重点看以下信号：

```bash
# server 语法和 Python 编译
python -m py_compile terminal-rl/remote/terminal_env.py terminal-rl/remote/pool_server.py terminal-rl/env_client.py terminal-rl/router_server.py
bash -n terminal-rl/remote/start_server.sh terminal-rl/remote/run_pool_server_pu_v2.sh terminal-rl/terminal-rl_qwen3-8b_mixed_dapo_nodynamic_pu.sh

# 运行日志关键字
grep -E "Reset exceeds|WORKER_RESET_TIMEOUT|Forced container recreation|eval_timeout|eval_parse_failed|eval_no_results|Timed out closing|Docker cleanup detached|Docker compose down finished|fixed task service|Periodic orphan Docker sweep|Idle orphan reap|Reset-storm orphan reap|RUN_SLOTS_EXHAUSTED" cpu_pool.log

# pool status 关键字段
curl -s http://127.0.0.1:${ENV_SERVER_PORT:-18081}/status

# Docker 侧残留趋势
docker ps --format '{{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}' \
  | grep -E 'tb__[0-9]+__|[0-9]+-[A-Za-z0-9]{8}-slime-run'
```

重点判断：

- `RuntimeError: cannot reuse already awaited coroutine` 是否消失。
- `/evaluate` 是否从 500 重试风暴变为 `ok=true, score=0.0, details.reason=...`。
- `pending_closes` 是否可回落，`pending_close_age_sec` 是否不持续增长。
- `/status` 中普通 task 的 `max_runs` 是否为 8，`892`/`1133` 是否为 1。
- 启动日志是否显示 `serial_task_ids=892,1133`，以及 `max_runs_per_task=8`。
- `docker ps` 中老 task 容器是否能被 reset recreate、force cleanup、pool periodic orphan sweep 或 watchdog orphan reap 回收。
- 是否还能持续出现 `Force cleanup timed out` 后对应容器继续 `Up > 10min`；如果仍存在，优先检查 Docker daemon 是否卡死、`docker rm -f` 是否可手工完成、以及 `WORKER_ORPHAN_DOCKER_SWEEP=1` 是否已在当前 server 进程中生效。
- 是否还出现 `tb__892__*` / `tb__1133__*` 长期残留；如果存在，优先确认对应 task 是否仍有 active lease，再检查 `compose down` 和 fixed service cleanup 日志。
- 正常慢 reset/image build 不应因为过小 timeout 被提前中断。
