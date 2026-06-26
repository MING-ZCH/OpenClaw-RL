from __future__ import annotations

import asyncio
import importlib
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass(frozen=True)
class _TaskSpec:
    task_name: str
    task_path: str
    instruction: str


@dataclass(frozen=True)
class _RunContext:
    uid: str
    group_index: int
    sample_index: int
    log_dir: Path
    rollout_id: int | None = None
    train_step: int | None = None
    rollout_step: int | None = None


@dataclass
class _TaskTimeouts:
    ensure_image: float = 300.0
    reset_session: float = 300.0
    close_session: float = 60.0
    eval: float = 600.0


class _FastAPI:
    def get(self, *_args, **_kwargs):
        return lambda fn: fn

    def post(self, *_args, **_kwargs):
        return lambda fn: fn

    def on_event(self, *_args, **_kwargs):
        return lambda fn: fn


class _JSONResponse(dict):
    def __init__(self, content=None, status_code=200, headers=None):
        super().__init__(content or {})
        self.status_code = status_code
        self.headers = headers or {}


class _Response:
    def __init__(self, content="", media_type=None):
        self.body = content
        self.media_type = media_type


class _DummyEnv:
    def __init__(self) -> None:
        self.reset_started = asyncio.Event()
        self.release_reset = asyncio.Event()
        self.close_count = 0

    async def reset(self, **_kwargs):
        self.reset_started.set()
        await self.release_reset.wait()
        return "user", []

    async def exec_tool(self, _tool_name, _arguments):
        return "observation"

    async def evaluate(self, _trajectory=None):
        return 1.0

    def last_eval_details(self):
        return None

    async def close(self):
        self.close_count += 1

    async def force_cleanup(self, reason="external"):
        self.force_cleanup_reason = reason


def _install_import_stubs(monkeypatch):
    fastapi_mod = types.ModuleType("fastapi")
    fastapi_mod.FastAPI = _FastAPI
    fastapi_mod.Request = object
    responses_mod = types.ModuleType("fastapi.responses")
    responses_mod.JSONResponse = _JSONResponse
    responses_mod.Response = _Response

    custom_types_mod = types.ModuleType("terminal-rl.custom_types")
    custom_types_mod.TaskSpec = _TaskSpec
    custom_types_mod.RunContext = _RunContext
    custom_types_mod.TaskTimeouts = _TaskTimeouts

    request_utils_mod = types.ModuleType("terminal-rl.request_utils")

    async def _json_payload(_request):
        return {}

    request_utils_mod.json_payload = _json_payload

    terminal_env_mod = types.ModuleType("terminal-rl.remote.terminal_env")
    terminal_env_mod.TerminalEnv = _DummyEnv
    terminal_env_mod.force_remove_orphan_docker_objects = lambda **_kwargs: 0

    docker_compose_utils_mod = types.ModuleType("terminal-rl.remote.docker_compose_utils")

    class DockerImageBuildError(RuntimeError):
        pass

    class DockerImagePreparationBacklogError(RuntimeError):
        pass

    class TaskImageBlacklistedError(DockerImageBuildError):
        pass

    docker_compose_utils_mod.DockerImageBuildError = DockerImageBuildError
    docker_compose_utils_mod.DockerImagePreparationBacklogError = (
        DockerImagePreparationBacklogError
    )
    docker_compose_utils_mod.TaskImageBlacklistedError = TaskImageBlacklistedError
    docker_compose_utils_mod.docker_image_build_status = lambda: {
        "active": 0,
        "waiting": 0,
    }

    monkeypatch.setitem(sys.modules, "uvicorn", types.ModuleType("uvicorn"))
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_mod)
    monkeypatch.setitem(sys.modules, "terminal-rl.custom_types", custom_types_mod)
    monkeypatch.setitem(sys.modules, "terminal-rl.request_utils", request_utils_mod)
    monkeypatch.setitem(sys.modules, "terminal-rl.remote.terminal_env", terminal_env_mod)
    monkeypatch.setitem(
        sys.modules, "terminal-rl.remote.docker_compose_utils", docker_compose_utils_mod
    )
    sys.modules.pop("terminal-rl.remote.pool_server", None)
    return importlib.import_module("terminal-rl.remote.pool_server")


def _new_pool(pool_server_mod, env: _DummyEnv, tmp_path: Path):
    class TestWorkerPool(pool_server_mod.WorkerPool):
        def _new_env(self):
            return env

    return TestWorkerPool(
        max_tasks=4,
        max_runs_per_task=4,
        run_idle_ttl=1,
        output_root=str(tmp_path),
        default_timeouts=_TaskTimeouts(),
        max_concurrent_closes=2,
    )


def test_pressure_guard_rejects_allocate_on_low_pids_headroom(monkeypatch):
    pool_server = _install_import_stubs(monkeypatch)
    monkeypatch.setenv("WORKER_DISK_GUARD_ENABLED", "0")
    monkeypatch.setenv("WORKER_PIDS_PAUSE_ALLOCATE_PCT", "99")
    monkeypatch.setenv("WORKER_PIDS_MIN_FREE_ALLOCATE", "6000")
    monkeypatch.setattr(
        pool_server,
        "worker_pressure_stats",
        lambda *args, **kwargs: {
            "procs": 100,
            "tasks": 10000,
            "pids_current": 10000,
            "pids_max": 15511,
            "pids_pct": 20.0,
            "zombies": 0,
            "dockerd": 1,
            "containerd": 1,
            "shim": 0,
            "runc": 0,
            "docker_cli_procs": 0,
            "docker_cli_ok": True,
        },
    )

    try:
        pool_server.assert_worker_has_capacity_for_docker(phase="allocate")
    except pool_server.ResourcePressureError as exc:
        assert exc.code == "WORKER_PIDS_HEADROOM_LOW"
        assert exc.details["pids_free"] == 5511
    else:
        raise AssertionError("expected low pids headroom to reject allocate")


def test_pressure_guard_rejects_reset_on_low_pids_headroom(monkeypatch):
    pool_server = _install_import_stubs(monkeypatch)
    monkeypatch.setenv("WORKER_DISK_GUARD_ENABLED", "0")
    monkeypatch.setenv("WORKER_PIDS_REJECT_RESET_PCT", "99")
    monkeypatch.setenv("WORKER_PIDS_MIN_FREE_RESET", "4000")
    monkeypatch.setattr(
        pool_server,
        "worker_pressure_stats",
        lambda *args, **kwargs: {
            "procs": 100,
            "tasks": 12511,
            "pids_current": 12511,
            "pids_max": 15511,
            "pids_pct": 20.0,
            "zombies": 0,
            "dockerd": 1,
            "containerd": 1,
            "shim": 0,
            "runc": 0,
            "docker_cli_procs": 0,
            "docker_cli_ok": True,
        },
    )

    try:
        pool_server.assert_worker_has_capacity_for_docker(phase="reset")
    except pool_server.ResourcePressureError as exc:
        assert exc.code == "WORKER_PIDS_HEADROOM_LOW"
        assert exc.details["pids_free"] == 3000
    else:
        raise AssertionError("expected low pids headroom to reject reset")


def test_close_allocated_run_cleans_up_without_unpack_error(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]

        assert await pool.close_run(lease_id, reason="test_close") is True
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=False)

        assert lease_id not in pool._run_to_task
        assert env.close_count == 1

    asyncio.run(_case())


def test_idle_reaper_skips_in_flight_reset(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        reset_task = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "do it"},
                {"uid": "u1", "log_dir": str(tmp_path)},
            )
        )
        await env.reset_started.wait()

        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            run_slot.last_used_ts = time.time() - 100
            expired = pool._reap_idle_locked()

        assert expired == []
        assert lease_id in pool._run_to_task
        assert env.close_count == 0

        env.release_reset.set()
        await reset_task
        assert lease_id in pool._run_to_task
        assert env.close_count == 0

    asyncio.run(_case())


def test_reset_admission_backlog_does_not_mark_waiter_resetting(monkeypatch, tmp_path):
    async def _case():
        monkeypatch.setenv("WORKER_MAX_CONCURRENT_RESETS", "1")
        monkeypatch.setenv("WORKER_RESET_ADMISSION_TIMEOUT", "0.05")
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        first = await pool.allocate("task-a")
        second = await pool.allocate("task-b")
        first_reset = asyncio.create_task(
            pool.reset(
                first["lease_id"],
                {"task_name": "a", "task_path": "seta_env/a", "instruction": "x"},
            )
        )
        await env.reset_started.wait()

        try:
            await pool.reset(
                second["lease_id"],
                {"task_name": "b", "task_path": "seta_env/b", "instruction": "x"},
            )
        except pool_server.ResetAdmissionBacklogError:
            pass
        else:
            raise AssertionError("expected reset admission backlog")

        status = await pool.status()
        assert status["phase_counts"].get("resetting") == 1
        assert status["phase_counts"].get("allocated") == 1
        assert status["reset_admission"]["rejected"] == 1

        env.release_reset.set()
        await first_reset

    asyncio.run(_case())


def test_cancelled_reset_waiter_leaves_admission_queue(monkeypatch, tmp_path):
    async def _case():
        monkeypatch.setenv("WORKER_MAX_CONCURRENT_RESETS", "1")
        monkeypatch.setenv("WORKER_RESET_ADMISSION_TIMEOUT", "10")
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        first = await pool.allocate("task-a")
        second = await pool.allocate("task-b")
        first_reset = asyncio.create_task(
            pool.reset(
                first["lease_id"],
                {"task_name": "a", "task_path": "seta_env/a", "instruction": "x"},
            )
        )
        try:
            await env.reset_started.wait()

            second_reset = asyncio.create_task(
                pool.reset(
                    second["lease_id"],
                    {"task_name": "b", "task_path": "seta_env/b", "instruction": "x"},
                )
            )
            for _ in range(100):
                status = await pool.status()
                if status["reset_admission"]["waiting"] == 1:
                    break
                await asyncio.sleep(0.01)
            assert status["reset_admission"]["waiting"] == 1

            second_reset.cancel()
            try:
                await second_reset
            except TimeoutError as exc:
                assert "WORKER_RESET_CANCELLED" in str(exc)
            else:
                raise AssertionError("expected reset cancellation timeout")

            for _ in range(100):
                status = await pool.status()
                if status["reset_admission"]["waiting"] == 0:
                    break
                await asyncio.sleep(0.01)
            assert status["reset_admission"]["waiting"] == 0
            assert status["phase_counts"].get("resetting") == 1
            assert status["phase_counts"].get("allocated") == 1
            async with pool._lock:
                second_slot = pool._get_run_slot(second["lease_id"])
                assert second_slot.reset_future is None
        finally:
            env.release_reset.set()
        await first_reset

    asyncio.run(_case())


def test_close_during_reset_is_deferred_until_in_flight_finishes(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        reset_task = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "do it"},
                {"uid": "u1", "log_dir": str(tmp_path)},
            )
        )
        await env.reset_started.wait()

        assert await pool.close_run(lease_id, reason="test_close") is True
        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            assert run_slot.close_requested is True
            assert run_slot.in_flight_ops == 1
            assert run_slot.phase == "closing_requested"

        env.release_reset.set()
        await reset_task
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=True)

        assert lease_id not in pool._run_to_task
        assert env.close_count == 1

    asyncio.run(_case())


def test_close_during_reset_force_releases_after_delay(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_CLOSE_REQUESTED_FORCE_RELEASE_AFTER", "0")
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        reset_task = asyncio.create_task(
            pool.reset(
                lease_id,
                {"task_name": "task", "task_path": "task", "instruction": "do it"},
                {"uid": "u1", "log_dir": str(tmp_path)},
            )
        )
        await env.reset_started.wait()

        assert await pool.close_run(lease_id, reason="test_close") is True
        await asyncio.sleep(0)
        if pool._force_cleanup_tasks:
            await asyncio.gather(*pool._force_cleanup_tasks, return_exceptions=True)

        assert lease_id not in pool._run_to_task
        assert env.force_cleanup_reason == "close_requested_force_release:test_close"

        reset_task.cancel()
        await asyncio.gather(reset_task, return_exceptions=True)

    asyncio.run(_case())


def test_status_and_readyz_report_stale_allocated_run(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_DISK_GUARD_ENABLED", "0")
        monkeypatch.setenv("WORKER_PRESSURE_GUARD_ENABLED", "0")
        monkeypatch.setenv("WORKER_ALLOCATED_TTL", "10")
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            run_slot.created_ts = time.time() - 20

        status = await pool.status()
        assert status["phase_counts"] == {"allocated": 1}
        assert status["stale_runs"][0]["lease_id"] == lease_id
        assert status["stale_runs"][0]["reason"] == "allocated_ttl_exceeded"

        pool_server.POOL = pool
        response = await pool_server.readyz()
        assert response.status_code == 503
        assert response["code"] == "WORKER_STALE_RUNS"

    asyncio.run(_case())


def test_allocate_auto_repairs_close_requested_task_capacity(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_DISK_GUARD_ENABLED", "0")
        monkeypatch.setenv("WORKER_PRESSURE_GUARD_ENABLED", "0")
        monkeypatch.setenv("WORKER_AUTO_REPAIR_ON_CAPACITY", "1")
        env = _DummyEnv()

        class OneTaskPool(pool_server.WorkerPool):
            def _new_env(self):
                return env

        pool = OneTaskPool(
            max_tasks=1,
            max_runs_per_task=4,
            run_idle_ttl=1,
            output_root=str(tmp_path),
            default_timeouts=_TaskTimeouts(),
            max_concurrent_closes=2,
        )
        pool_server.POOL = pool

        old_lease = (await pool.allocate("old-task"))["lease_id"]
        async with pool._lock:
            old_slot = pool._get_run_slot(old_lease)
            old_slot.phase = "closing_requested"
            old_slot.close_requested = True
            old_slot.close_reason = "test"
            old_slot.close_requested_ts = time.time() - 5
            old_slot.in_flight_ops = 1
            old_slot.active_op = "reset"

        async def _payload(_request):
            return {"task_key": "new-task"}

        pool_server.json_payload = _payload
        response = await pool_server.allocate(object())

        assert response.status_code == 200
        assert response["ok"] is True
        assert response["auto_repair"]["close_requested"]["repaired_count"] == 1
        assert old_lease not in pool._run_to_task
        assert response["lease_id"] in pool._run_to_task
        if pool._force_cleanup_tasks:
            await asyncio.gather(*pool._force_cleanup_tasks, return_exceptions=True)

    asyncio.run(_case())


def test_repair_stale_run_force_cleans_inflight_close_requested(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_CLOSING_REQUESTED_TTL", "10")
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            run_slot.phase = "closing_requested"
            run_slot.close_requested = True
            run_slot.close_reason = "test"
            run_slot.close_requested_ts = time.time() - 20
            run_slot.in_flight_ops = 1
            run_slot.active_op = "exec_tool"

        status = await pool.status()
        assert status["stale_runs"][0]["lease_id"] == lease_id
        result = await pool.repair_stale_runs(reason="test", min_age=0, max_repairs=10)

        assert result["repaired"] is True
        assert result["repaired_count"] == 1
        assert result["repaired_runs"][0]["lease_id"] == lease_id
        assert env.force_cleanup_reason == "repair_stale_runs:test"
        status = await pool.status()
        assert status["total_active_runs"] == 0
        assert status["stale_runs"] == []

    asyncio.run(_case())


def test_reaper_removes_stale_allocated_run(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_ALLOCATED_TTL", "10")
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)

        lease = await pool.allocate("task")
        lease_id = lease["lease_id"]
        async with pool._lock:
            run_slot = pool._get_run_slot(lease_id)
            run_slot.created_ts = time.time() - 20
            expired = pool._reap_idle_locked()

        assert len(expired) == 1
        assert expired[0][1] == lease_id
        assert lease_id not in pool._run_to_task

        for task_key, run_id, run_slot in expired:
            pool._schedule_close(task_key, run_id, run_slot, reason="test stale reap")
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=False)
        assert env.close_count == 1

    asyncio.run(_case())

def test_pending_close_repair_allows_negative_active_limit(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        env = _DummyEnv()
        pool = _new_pool(pool_server, env, tmp_path)
        await pool.allocate("task")

        sleeper = asyncio.create_task(asyncio.sleep(3600))
        pool._closing_tasks.add(sleeper)
        pool._closing_task_started[sleeper] = time.time() - 60
        try:
            result = await pool.repair_pending_closes(
                reason="test",
                max_active_runs=-1,
                cancel_timeout=0.1,
                min_age=0,
            )
        finally:
            sleeper.cancel()
            await asyncio.gather(sleeper, return_exceptions=True)

        assert result["repaired"] is True
        assert result["cancelled"] == 1
        assert result["pending_after"] == 0

    asyncio.run(_case())


def test_rollout_probe_resets_executes_and_closes(monkeypatch, tmp_path):
    async def _case():
        pool_server = _install_import_stubs(monkeypatch)
        monkeypatch.setenv("WORKER_DISK_GUARD_ENABLED", "0")
        monkeypatch.setenv("WORKER_PRESSURE_GUARD_ENABLED", "0")
        env = _DummyEnv()
        env.release_reset.set()
        pool = _new_pool(pool_server, env, tmp_path)
        pool_server.POOL = pool

        async def _payload(_request):
            return {
                "task_key": "probe-task",
                "task_meta": {
                    "task_name": "probe-task",
                    "task_path": "probe-task",
                    "instruction": "probe",
                },
                "run_ctx": {"uid": "probe", "log_dir": str(tmp_path)},
                "tool_call": {"name": "noop", "arguments": {}},
            }

        pool_server.json_payload = _payload
        response = await pool_server.probe_rollout(object())

        assert response.status_code == 200
        assert response["ok"] is True
        assert response["exec"]["tool_name"] == "noop"
        if pool._closing_tasks:
            await asyncio.gather(*pool._closing_tasks, return_exceptions=False)
        assert env.close_count == 1
        assert response["lease_id"] not in pool._run_to_task

    asyncio.run(_case())
