from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import shutil
import subprocess
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from ..custom_types import RunContext, TaskSpec, TaskTimeouts
from ..request_utils import json_payload
from .terminal_env import (
    TerminalEnv,
    force_remove_orphan_docker_objects,
)
from .docker_compose_utils import (
    DockerImageBuildError,
    DockerImagePreparationBacklogError,
    TaskImageBlacklistedError,
    docker_image_build_status,
)

logger = logging.getLogger("terminal.env.worker")
app = FastAPI()

_DOCKER_CLI_FAIL_STREAK = 0
_DOCKER_DEGRADED_UNTIL = 0.0
_DOCKER_DEGRADED_REASON = ""


def _parse_timeout_overrides(
    base: TaskTimeouts, payload: dict[str, Any] | None
) -> TaskTimeouts:
    if not isinstance(payload, dict):
        return base

    def _pick(key: str, default: float, *, minimum: float | None = None) -> float:
        raw = payload.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        if value <= 0:
            return default
        if minimum is not None and value < minimum:
            logger.debug(
                "Raising client timeout override %s=%.1fs to worker floor %.1fs",
                key,
                value,
                minimum,
            )
            return minimum
        return value

    return TaskTimeouts(
        ensure_image=_pick(
            "ensure_image",
            base.ensure_image,
            minimum=base.ensure_image,
        ),
        reset_session=_pick(
            "reset_session",
            base.reset_session,
            minimum=base.reset_session,
        ),
        close_session=_pick("close_session", base.close_session),
        eval=_pick("eval", base.eval),
    )


def _build_task_spec(task_meta: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_name=str(task_meta.get("task_name", "unknown")),
        task_path=str(task_meta.get("task_path", "")),
        instruction=str(task_meta.get("instruction", "")),
    )


def _build_run_ctx(
    run_ctx_payload: dict[str, Any] | None, default_log_dir: Path
) -> RunContext:
    payload = run_ctx_payload if isinstance(run_ctx_payload, dict) else {}
    uid = str(payload.get("uid") or uuid.uuid4().hex[:8])
    try:
        group_index = int(payload.get("group_index") or 0)
    except (TypeError, ValueError):
        group_index = 0
    try:
        sample_index = int(payload.get("sample_index") or 0)
    except (TypeError, ValueError):
        sample_index = 0

    log_dir_raw = payload.get("log_dir")
    if isinstance(log_dir_raw, str) and log_dir_raw:
        log_dir = Path(log_dir_raw).resolve()
    else:
        log_dir = default_log_dir.resolve()

    return RunContext(
        uid=uid,
        group_index=group_index,
        sample_index=sample_index,
        log_dir=log_dir,
    )


class CapacityError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class ResourcePressureError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any]):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


class ResetInProgressError(Exception):
    def __init__(self, run_lease_id: str, request_id: str | None):
        self.run_lease_id = run_lease_id
        self.request_id = request_id
        super().__init__(
            f"Run {run_lease_id} already has a different reset in progress"
        )


class ResetAdmissionBacklogError(Exception):
    def __init__(self, run_lease_id: str, timeout: float, max_concurrent: int):
        self.run_lease_id = run_lease_id
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        super().__init__(
            f"WORKER_RESET_ADMISSION_BACKLOG lease_id={run_lease_id} "
            f"timeout={timeout:.1f}s max_concurrent_resets={max_concurrent}"
        )


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _mark_docker_degraded(reason: str) -> None:
    global _DOCKER_DEGRADED_REASON, _DOCKER_DEGRADED_UNTIL
    cooldown = max(0.0, _env_float("WORKER_DOCKER_DEGRADED_COOLDOWN", 120.0))
    if cooldown <= 0:
        return
    _DOCKER_DEGRADED_REASON = reason
    _DOCKER_DEGRADED_UNTIL = max(_DOCKER_DEGRADED_UNTIL, time.time() + cooldown)


def _record_docker_cli_probe(ok: bool, *, timeout: float) -> None:
    global _DOCKER_CLI_FAIL_STREAK, _DOCKER_DEGRADED_REASON, _DOCKER_DEGRADED_UNTIL
    if ok:
        _DOCKER_CLI_FAIL_STREAK = 0
        if time.time() >= _DOCKER_DEGRADED_UNTIL:
            _DOCKER_DEGRADED_REASON = ""
        return
    _DOCKER_CLI_FAIL_STREAK += 1
    threshold = max(1, _env_int("WORKER_DOCKER_DEGRADED_FAIL_STREAK", 2))
    if _DOCKER_CLI_FAIL_STREAK >= threshold:
        _mark_docker_degraded(
            f"docker CLI probe failed {_DOCKER_CLI_FAIL_STREAK} consecutive "
            f"time(s), timeout={timeout:.1f}s"
        )


def _docker_degraded_details() -> dict[str, Any] | None:
    remaining = _DOCKER_DEGRADED_UNTIL - time.time()
    if remaining <= 0:
        return None
    return {
        "docker_degraded_remaining_sec": round(remaining, 1),
        "docker_degraded_reason": _DOCKER_DEGRADED_REASON,
        "docker_cli_fail_streak": _DOCKER_CLI_FAIL_STREAK,
    }


def _split_env_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


_TASK_ID_PREFIX_RE = re.compile(r"^([0-9]+)(?:[-_.:]|$)")
_FIXED_TASK_SERVICE_RE = re.compile(r"^tb__([0-9]+)__.*")


def _task_id_from_ref(value: str | None) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    fixed = _FIXED_TASK_SERVICE_RE.match(raw)
    if fixed:
        return fixed.group(1)
    prefixed = _TASK_ID_PREFIX_RE.match(raw)
    if prefixed:
        return prefixed.group(1)
    return None


def _docker_name_variants(value: str | None) -> set[str]:
    if not value:
        return set()
    raw = value.strip()
    if not raw:
        return set()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-_.")
    variants = {
        raw,
        cleaned,
        cleaned.replace(".", "-"),
        cleaned.replace("_", "-"),
        cleaned.replace(".", "_"),
    }
    return {v for v in variants if v and "slime-run" in v}


def _task_key_tokens(task_key: str) -> set[str]:
    raw = str(task_key or "").strip()
    tokens = {raw} if raw else set()
    if ":" in raw:
        task_name, task_path = raw.split(":", 1)
        if task_name:
            tokens.add(task_name)
        if task_path:
            tokens.add(task_path)
            tail = Path(task_path).name
            if tail:
                tokens.add(tail)
    task_id = _task_id_from_ref(raw)
    if task_id:
        tokens.add(task_id)
    return {token for token in tokens if token}


def _parse_task_max_runs_overrides(raw: str | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in _split_env_csv(raw):
        if "=" not in item:
            logger.warning(
                "Ignoring malformed WORKER_TASK_MAX_RUNS_OVERRIDES entry %r; "
                "expected task=limit",
                item,
            )
            continue
        key, value_raw = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            value = int(value_raw.strip())
        except ValueError:
            logger.warning(
                "Ignoring invalid WORKER_TASK_MAX_RUNS_OVERRIDES entry %r", item
            )
            continue
        if value <= 0:
            logger.warning(
                "Ignoring non-positive WORKER_TASK_MAX_RUNS_OVERRIDES entry %r", item
            )
            continue
        overrides[key] = value
    return overrides


def docker_data_root_stats() -> dict[str, Any]:
    path = os.getenv("DOCKER_DATA_ROOT") or os.getenv("DOCKER_ROOT") or "/data"
    usage = shutil.disk_usage(path)
    st = os.statvfs(path)
    total_inodes = int(st.f_files)
    free_inodes = int(st.f_ffree)
    used_inodes = max(total_inodes - free_inodes, 0)
    used_pct = (usage.used * 100.0 / usage.total) if usage.total else 0.0
    inode_used_pct = (
        (used_inodes * 100.0 / total_inodes) if total_inodes else 0.0
    )
    return {
        "path": path,
        "total_gb": usage.total / 1024**3,
        "used_gb": usage.used / 1024**3,
        "free_gb": usage.free / 1024**3,
        "used_pct": used_pct,
        "total_inodes": total_inodes,
        "used_inodes": used_inodes,
        "free_inodes": free_inodes,
        "inode_used_pct": inode_used_pct,
    }


_PRESSURE_CACHE: tuple[float, dict[str, Any]] | None = None


def _read_cgroup_pids_stats() -> dict[str, Any] | None:
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None

    search_roots: list[Path] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers = parts[1]
        rel = parts[2].strip("/")
        if controllers == "":
            search_roots.append(Path("/sys/fs/cgroup") / rel)
        elif "pids" in controllers.split(","):
            search_roots.append(Path("/sys/fs/cgroup/pids") / rel)
            search_roots.append(Path("/sys/fs/cgroup") / rel)

    for start in search_roots:
        cur = start
        while True:
            current_file = cur / "pids.current"
            max_file = cur / "pids.max"
            if current_file.is_file() and max_file.is_file():
                try:
                    current = int(current_file.read_text().strip())
                    raw_max = max_file.read_text().strip()
                    if raw_max == "max":
                        return None
                    maximum = int(raw_max)
                except (OSError, ValueError):
                    return None
                if maximum > 0:
                    return {
                        "pids_current": current,
                        "pids_max": maximum,
                        "pids_source": str(cur),
                    }
                return None
            if cur == cur.parent:
                break
            cur = cur.parent
    return None


def _read_proc_pressure_stats() -> dict[str, Any]:
    total_procs = 0
    total_tasks = 0
    zombies = 0
    shim = 0
    runc = 0
    dockerd = 0
    containerd = 0
    docker_cli = 0

    for proc_dir in Path("/proc").glob("[0-9]*"):
        if not proc_dir.is_dir():
            continue
        total_procs += 1
        try:
            name = (proc_dir / "comm").read_text(errors="ignore").strip()
        except OSError:
            name = ""
        try:
            stat = (proc_dir / "stat").read_text(errors="ignore")
            rest = stat.split(") ", 1)[1]
            if rest.split(" ", 1)[0] == "Z":
                zombies += 1
        except (OSError, IndexError):
            pass
        try:
            total_tasks += sum(1 for p in (proc_dir / "task").iterdir() if p.is_dir())
        except OSError:
            pass

        if name == "dockerd":
            dockerd += 1
        elif name == "containerd":
            containerd += 1
        elif name.startswith("containerd-shim"):
            shim += 1
        elif name == "runc":
            runc += 1
        elif name == "docker":
            docker_cli += 1

    pids_max = 0
    try:
        pids_max = int(Path("/proc/sys/kernel/threads-max").read_text().strip())
    except (OSError, ValueError):
        pids_max = 0
    pids_current = total_tasks
    pids_source = "/proc"
    cgroup_pids = _read_cgroup_pids_stats()
    if cgroup_pids is not None:
        cgroup_max = int(cgroup_pids.get("pids_max") or 0)
        if cgroup_max > 0 and (pids_max <= 0 or cgroup_max <= pids_max):
            pids_current = int(cgroup_pids.get("pids_current") or total_tasks)
            pids_max = cgroup_max
            pids_source = str(cgroup_pids.get("pids_source") or "cgroup")

    pids_pct = (pids_current * 100.0 / pids_max) if pids_max > 0 else 0.0
    return {
        "procs": total_procs,
        "tasks": total_tasks,
        "pids_current": pids_current,
        "pids_max": pids_max,
        "pids_pct": pids_pct,
        "pids_source": pids_source,
        "zombies": zombies,
        "dockerd": dockerd,
        "containerd": containerd,
        "shim": shim,
        "runc": runc,
        "docker_cli_procs": docker_cli,
    }


def _docker_cli_ok(timeout_sec: float) -> bool:
    try:
        result = subprocess.run(
            ["docker", "ps", "-q"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_sec,
        )
        return result.returncode == 0
    except Exception:
        return False


def worker_pressure_stats(*, force: bool = False) -> dict[str, Any]:
    global _PRESSURE_CACHE
    ttl = _env_float("WORKER_PRESSURE_CACHE_TTL", 5.0)
    now = time.time()
    if (
        not force
        and _PRESSURE_CACHE is not None
        and now - _PRESSURE_CACHE[0] <= ttl
    ):
        return dict(_PRESSURE_CACHE[1])

    stats = _read_proc_pressure_stats()
    docker_timeout = _env_float("WORKER_DOCKER_CLI_TIMEOUT", 3.0)
    docker_cli_ok = _docker_cli_ok(docker_timeout)
    _record_docker_cli_probe(docker_cli_ok, timeout=docker_timeout)
    stats["docker_cli_ok"] = docker_cli_ok
    stats["docker_cli_timeout_sec"] = docker_timeout
    degraded = _docker_degraded_details()
    if degraded is not None:
        stats.update(degraded)
    _PRESSURE_CACHE = (now, dict(stats))
    return stats


def assert_worker_has_capacity_for_docker(
    *,
    phase: str = "health",
    pending_closes: int = 0,
    pool_status: dict[str, Any] | None = None,
) -> None:
    if os.getenv("WORKER_DISK_GUARD_ENABLED", "1") == "0":
        disk_guard_enabled = False
    else:
        disk_guard_enabled = True

    if disk_guard_enabled:
        min_free_gb = _env_float("WORKER_MIN_DOCKER_FREE_GB", 50.0)
        max_used_pct = _env_float("WORKER_MAX_DOCKER_USED_PCT", 85.0)
        max_inode_pct = _env_float("WORKER_MAX_DOCKER_INODE_PCT", 80.0)

        try:
            stats = docker_data_root_stats()
        except Exception as exc:
            raise ResourcePressureError(
                "WORKER_DISK_STATS_FAILED",
                f"Failed to read Docker data-root stats: {exc}",
                {"error": str(exc), "phase": phase},
            ) from exc

        over_capacity = (
            stats["free_gb"] < min_free_gb
            or stats["used_pct"] > max_used_pct
            or stats["inode_used_pct"] > max_inode_pct
        )
        if over_capacity:
            raise ResourcePressureError(
                "WORKER_DOCKER_DISK_PRESSURE",
                (
                    "Worker Docker data-root is under disk pressure: "
                    f"path={stats['path']} free={stats['free_gb']:.1f}GB "
                    f"used={stats['used_pct']:.1f}% inode={stats['inode_used_pct']:.1f}% "
                    f"thresholds free>={min_free_gb:.1f}GB used<={max_used_pct:.1f}% "
                    f"inode<={max_inode_pct:.1f}%"
                ),
                {
                    **stats,
                    "phase": phase,
                    "min_free_gb": min_free_gb,
                    "max_used_pct": max_used_pct,
                    "max_inode_pct": max_inode_pct,
                },
            )

    if os.getenv("WORKER_PRESSURE_GUARD_ENABLED", "1") == "0":
        return

    degraded = _docker_degraded_details()
    if degraded is not None and phase in {"allocate", "reset"}:
        raise ResourcePressureError(
            "WORKER_DOCKER_DEGRADED",
            "Worker Docker API is in short cooldown after recent CLI failures; "
            "refusing new Docker work.",
            {"phase": phase, "pending_closes": pending_closes, **degraded},
        )

    # CRITICAL FIX: Catch RuntimeError that blocks all reset operations
    # Issue: "cannot reuse already awaited coroutine" causes 100% reset failure
    # This is a defensive measure while investigating root cause
    try:
        pressure = worker_pressure_stats()
    except RuntimeError as e:
        logger.error(
            "RuntimeError in worker_pressure_stats (allowing %s to proceed): %s",
            phase,
            e,
            exc_info=True
        )
        # Degraded mode: skip pressure checks to unblock reset operations
        # This allows containers to be reset/deleted, preventing >1h uptime accumulation
        return
    except Exception as e:
        logger.exception("Unexpected error in worker_pressure_stats for phase=%s: %s", phase, e)
        return

    pids_pause_pct = _env_float("WORKER_PIDS_PAUSE_ALLOCATE_PCT", 60.0)
    pids_reject_reset_pct = _env_float("WORKER_PIDS_REJECT_RESET_PCT", 70.0)
    pids_min_free_allocate = _env_int("WORKER_PIDS_MIN_FREE_ALLOCATE", 6000)
    pids_min_free_reset = _env_int("WORKER_PIDS_MIN_FREE_RESET", 4000)
    shim_pause = _env_int("WORKER_SHIM_PAUSE_ALLOCATE", 256)
    shim_reject_reset = _env_int("WORKER_SHIM_REJECT_RESET", 384)
    pending_pause = _env_int("WORKER_PENDING_CLOSES_PAUSE_ALLOCATE", 50)
    pending_reject_reset = _env_int("WORKER_PENDING_CLOSES_REJECT_RESET", 100)

    pids_current = int(pressure.get("pids_current") or pressure.get("tasks") or 0)
    pids_max = int(pressure.get("pids_max") or 0)
    pids_free = max(pids_max - pids_current, 0) if pids_max > 0 else -1

    details = {
        **pressure,
        "phase": phase,
        "pending_closes": pending_closes,
        "pids_free": pids_free,
        "pids_min_free_allocate": pids_min_free_allocate,
        "pids_min_free_reset": pids_min_free_reset,
    }
    if pool_status is not None:
        phase_counts = pool_status.get("phase_counts", {})
        resetting = int((phase_counts or {}).get("resetting", 0) or 0)
        active_runs = int(pool_status.get("total_active_runs", 0) or 0)
        reset_age = pool_status.get("resetting_age_sec", {}) or {}
        reset_max_age = float(reset_age.get("max", 0.0) or 0.0)
        details.update(
            {
                "pool_total_active_runs": active_runs,
                "pool_resetting_runs": resetting,
                "pool_resetting_max_age_sec": reset_max_age,
            }
        )
        if (
            phase in {"allocate", "reset"}
            and _env_bool("WORKER_RESET_STORM_GUARD", True)
        ):
            block_allocate = _env_bool("WORKER_RESET_STORM_BLOCK_ALLOCATE", True)
            if phase == "reset" or block_allocate:
                min_resetting = _env_int("WORKER_RESET_STORM_MIN_RESETTING", 32)
                min_age = _env_float("WORKER_RESET_STORM_MIN_AGE", 180.0)
                ratio_threshold = _env_float("WORKER_RESET_STORM_RATIO_PCT", 50.0)
                ratio = (
                    resetting * 100.0 / max(1, active_runs)
                    if active_runs > 0
                    else 0.0
                )
                details["pool_resetting_ratio_pct"] = round(ratio, 1)
                if (
                    resetting >= min_resetting
                    and ratio >= ratio_threshold
                    and reset_max_age >= min_age
                ):
                    _mark_docker_degraded(
                        f"reset storm resetting={resetting}/{active_runs} "
                        f"ratio={ratio:.1f}% max_age={reset_max_age:.1f}s"
                    )
                    raise ResourcePressureError(
                        "WORKER_RESET_STORM",
                        "Worker has a reset storm; refusing new reset/allocation "
                        "until existing reset work drains.",
                        {
                            **details,
                            "reset_storm_min_resetting": min_resetting,
                            "reset_storm_min_age": min_age,
                            "reset_storm_ratio_pct": ratio_threshold,
                        },
                    )
    if not bool(pressure.get("docker_cli_ok", False)):
        raise ResourcePressureError(
            "WORKER_DOCKER_CLI_UNHEALTHY",
            "Worker Docker CLI probe failed; refusing new Docker work.",
            details,
        )

    if phase == "reset":
        if pressure["pids_pct"] >= pids_reject_reset_pct:
            raise ResourcePressureError(
                "WORKER_PIDS_PRESSURE",
                (
                    f"Worker pids pressure {pressure['pids_pct']:.1f}% "
                    f">= reset threshold {pids_reject_reset_pct:.1f}%"
                ),
                details,
            )
        if pids_free >= 0 and pids_free < pids_min_free_reset:
            raise ResourcePressureError(
                "WORKER_PIDS_HEADROOM_LOW",
                (
                    f"Worker pids free headroom {pids_free} "
                    f"< reset threshold {pids_min_free_reset}"
                ),
                details,
            )
        if pressure["shim"] >= shim_reject_reset:
            raise ResourcePressureError(
                "WORKER_SHIM_PRESSURE",
                f"Worker shim pressure {pressure['shim']} >= reset threshold {shim_reject_reset}",
                details,
            )
        if pending_closes >= pending_reject_reset:
            raise ResourcePressureError(
                "WORKER_PENDING_CLOSES_PRESSURE",
                f"Worker pending_closes {pending_closes} >= reset threshold {pending_reject_reset}",
                details,
            )
        return

    if phase in {"allocate", "health"}:
        if pressure["pids_pct"] >= pids_pause_pct:
            raise ResourcePressureError(
                "WORKER_PIDS_PRESSURE",
                (
                    f"Worker pids pressure {pressure['pids_pct']:.1f}% "
                    f">= allocate threshold {pids_pause_pct:.1f}%"
                ),
                details,
            )
        if pids_free >= 0 and pids_free < pids_min_free_allocate:
            raise ResourcePressureError(
                "WORKER_PIDS_HEADROOM_LOW",
                (
                    f"Worker pids free headroom {pids_free} "
                    f"< allocate threshold {pids_min_free_allocate}"
                ),
                details,
            )
        if pressure["shim"] >= shim_pause:
            raise ResourcePressureError(
                "WORKER_SHIM_PRESSURE",
                f"Worker shim pressure {pressure['shim']} >= allocate threshold {shim_pause}",
                details,
            )
        if pending_closes >= pending_pause:
            raise ResourcePressureError(
                "WORKER_PENDING_CLOSES_PRESSURE",
                f"Worker pending_closes {pending_closes} >= allocate threshold {pending_pause}",
                details,
            )


@dataclass
class RunSlot:
    run_lease_id: str
    task_key: str
    env: TerminalEnv
    created_ts: float = field(default_factory=time.time)
    last_used_ts: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    phase: str = "allocated"
    in_flight_ops: int = 0
    active_op: str | None = None
    close_requested: bool = False
    close_reason: str | None = None
    close_requested_ts: float | None = None
    reset_started_ts: float | None = None
    reset_completed_ts: float | None = None
    reset_request_id: str | None = None
    reset_future: asyncio.Task | None = None
    reset_result: dict[str, Any] | None = None
    first_step_ts: float | None = None
    evaluate_completed_ts: float | None = None
    drop_scheduled: bool = False  # P0 fix: Flag to prevent double-pop race


@dataclass
class TaskSlot:
    task_key: str
    runs: dict[str, RunSlot] = field(default_factory=dict)
    created_ts: float = field(default_factory=time.time)
    last_used_ts: float = field(default_factory=time.time)


class WorkerPool:
    def __init__(
        self,
        *,
        max_tasks: int,
        max_runs_per_task: int,
        run_idle_ttl: int,
        output_root: str,
        default_timeouts: TaskTimeouts,
        idempotency_ttl: int = 300,
        max_concurrent_closes: int = 8,
    ) -> None:
        self.max_tasks = max_tasks
        self.max_runs_per_task = max_runs_per_task
        self.run_idle_ttl = run_idle_ttl
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.default_timeouts = default_timeouts
        self.idempotency_ttl = idempotency_ttl
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

        self._tasks: dict[str, TaskSlot] = {}
        self._run_to_task: dict[str, str] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, float]] = {}
        self._lock = asyncio.Lock()

        self._close_sem = asyncio.Semaphore(max_concurrent_closes)
        self.max_concurrent_resets = _env_int("WORKER_MAX_CONCURRENT_RESETS", 16)
        self.reset_admission_timeout = _env_float("WORKER_RESET_ADMISSION_TIMEOUT", 30.0)
        self._reset_admission_sem = asyncio.BoundedSemaphore(
            max(1, self.max_concurrent_resets)
        )
        self._reset_admission_waiting = 0
        self._reset_admission_rejected = 0
        self._closing_tasks: set[asyncio.Task] = set()
        self._closing_task_started: dict[asyncio.Task, float] = {}
        self._closing_task_labels: dict[asyncio.Task, str] = {}
        self._force_cleanup_tasks: set[asyncio.Task] = set()
        self._force_cleanup_task_started: dict[asyncio.Task, float] = {}
        self._force_cleanup_task_labels: dict[asyncio.Task, str] = {}
        self._close_requested_release_tasks: dict[str, asyncio.Task] = {}

        # P0 fix: Track reset count for automatic shim cleanup
        self._reset_count: int = 0
        self._last_shim_cleanup_ts: float = time.time()
        self._last_orphan_sweep_ts: float = 0.0
        self._orphan_sweep_fail_streak: int = 0
        self._orphan_sweep_backoff_until: float = 0.0
        self._serial_task_ids = set(
            _split_env_csv(os.getenv("WORKER_SERIAL_TASK_IDS", "892,1133"))
        )
        self._task_max_runs_overrides = _parse_task_max_runs_overrides(
            os.getenv("WORKER_TASK_MAX_RUNS_OVERRIDES", "")
        )
        self._auto_serialize_unsafe_compose = _env_bool(
            "WORKER_AUTO_SERIALIZE_UNSAFE_COMPOSE", False
        )
        self._unsafe_compose_cache: dict[str, bool] = {}

    def _new_env(self) -> TerminalEnv:
        return TerminalEnv()

    @staticmethod
    def _run_slot_container_info(run_slot: RunSlot) -> dict[str, Any]:
        env = run_slot.env
        terminal = getattr(env, "_terminal", None)
        container = getattr(terminal, "container", None) if terminal is not None else None
        container_id = getattr(container, "id", None)
        short_id = container_id[:12] if isinstance(container_id, str) else None
        container_name = (
            getattr(container, "name", None)
            or getattr(env, "_last_client_container_name", None)
        )
        container_status = getattr(container, "status", None)
        trial_name = getattr(env, "_last_trial_name", None)
        return {
            "id": container_id,
            "short_id": short_id,
            "name": container_name,
            "status": container_status,
            "trial_name": trial_name,
        }

    @classmethod
    def _run_slot_container_ref(cls, run_slot: RunSlot) -> str:
        info = cls._run_slot_container_info(run_slot)
        return (
            f"container_name={info.get('name') or '?'} "
            f"container_id={info.get('short_id') or '?'} "
            f"container_status={info.get('status') or '?'} "
            f"trial={info.get('trial_name') or '?'}"
        )

    def _active_container_names_locked(self) -> set[str]:
        names: set[str] = set()
        for task_slot in self._tasks.values():
            for run_slot in task_slot.runs.values():
                info = self._run_slot_container_info(run_slot)
                name = info.get("name")
                if isinstance(name, str) and name:
                    names.add(name)
        return names

    def _task_uses_unsafe_compose(self, task_key: str) -> bool:
        cached = self._unsafe_compose_cache.get(task_key)
        if cached is not None:
            return cached
        unsafe = False
        dataset_dir = os.getenv("DATASET_DIR", "").strip()
        if dataset_dir and ":" in task_key:
            _task_name, task_path = task_key.split(":", 1)
            compose_path = Path(dataset_dir) / task_path / "docker-compose.yaml"
            try:
                text = compose_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
            if text:
                fixed_non_client_name = False
                current_service = ""
                for raw_line in text.splitlines():
                    stripped = raw_line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    indent = len(raw_line) - len(raw_line.lstrip(" "))
                    if indent == 2 and stripped.endswith(":"):
                        current_service = stripped[:-1]
                    if stripped.startswith("container_name:") and current_service != "client":
                        fixed_non_client_name = True
                unsafe = fixed_non_client_name or "ipam:" in text or "subnet:" in text
        self._unsafe_compose_cache[task_key] = unsafe
        if unsafe:
            logger.warning(
                "Task %s detected as non-parallel-safe compose; "
                "effective max_runs_per_task=1",
                task_key,
            )
        return unsafe

    def _effective_max_runs_per_task(self, task_key: str) -> int:
        tokens = _task_key_tokens(task_key)
        for token in tokens:
            override = self._task_max_runs_overrides.get(token)
            if override is not None:
                return max(1, override)
        if tokens.intersection(self._serial_task_ids):
            return 1
        if self._auto_serialize_unsafe_compose and self._task_uses_unsafe_compose(
            task_key
        ):
            return 1
        return max(1, self.max_runs_per_task)

    def _active_docker_refs_locked(self) -> tuple[set[str], set[str], set[str]]:
        container_names: set[str] = set()
        project_names: set[str] = set()
        task_ids: set[str] = set()
        for task_key, task_slot in self._tasks.items():
            task_id = _task_id_from_ref(task_key)
            if task_id:
                task_ids.add(task_id)
            for run_slot in task_slot.runs.values():
                info = self._run_slot_container_info(run_slot)
                for key in ("name", "trial_name"):
                    value = info.get(key)
                    if not isinstance(value, str) or not value:
                        continue
                    if key == "name":
                        container_names.add(value)
                    project_names.update(_docker_name_variants(value))
                    task_id = _task_id_from_ref(value)
                    if task_id:
                        task_ids.add(task_id)
        return container_names, project_names, task_ids

    def _pop_run_slot_locked(
        self, run_lease_id: str
    ) -> tuple[str, RunSlot] | None:
        task_key = self._run_to_task.pop(run_lease_id, None)
        if task_key is None:
            return None
        task_slot = self._tasks.get(task_key)
        run_slot = task_slot.runs.pop(run_lease_id, None) if task_slot else None
        if task_slot is not None and not task_slot.runs:
            self._tasks.pop(task_key, None)
            logger.info("Removed empty task slot: %s", task_key)
        if run_slot is None:
            return None
        return task_key, run_slot

    def _phase_for_op(self, op_name: str) -> str:
        return {
            "reset": "resetting",
            "exec_tool": "stepping",
            "evaluate": "evaluating",
            "heartbeat": "heartbeat",
        }.get(op_name, op_name)

    async def _begin_run_op(self, run_lease_id: str, op_name: str) -> RunSlot:
        async with self._lock:
            run_slot = self._get_run_slot(run_lease_id)
            if run_slot.close_requested:
                raise RuntimeError(
                    f"Run {run_lease_id} is closing; rejecting new {op_name} request"
                )
            now = time.time()
            run_slot.in_flight_ops += 1
            run_slot.active_op = op_name
            run_slot.phase = self._phase_for_op(op_name)
            run_slot.last_used_ts = now
            if op_name == "reset":
                run_slot.reset_started_ts = now
            logger.debug(
                "Run op begin: lease=%s task=%s op=%s phase=%s in_flight=%d %s",
                run_lease_id,
                run_slot.task_key,
                op_name,
                run_slot.phase,
                run_slot.in_flight_ops,
                self._run_slot_container_ref(run_slot),
            )
            return run_slot

    async def _finish_run_op(
        self, run_slot: RunSlot, op_name: str, *, success: bool, is_timeout_drop: bool = False
    ) -> None:
        close_after: tuple[str, str, RunSlot, str] | None = None
        async with self._lock:
            now = time.time()
            run_slot.in_flight_ops = max(0, run_slot.in_flight_ops - 1)
            run_slot.last_used_ts = now
            if success:
                if op_name == "reset":
                    run_slot.reset_completed_ts = now
                    run_slot.phase = "ready"
                    # P0 fix: Track successful resets for shim cleanup trigger
                    self._reset_count += 1
                elif op_name == "exec_tool":
                    if run_slot.first_step_ts is None:
                        run_slot.first_step_ts = now
                    run_slot.phase = "stepped"
                elif op_name == "evaluate":
                    run_slot.evaluate_completed_ts = now
                    run_slot.phase = "evaluated"
                elif run_slot.in_flight_ops == 0:
                    run_slot.phase = "ready"
            else:
                run_slot.phase = "failed"
            if run_slot.in_flight_ops == 0:
                run_slot.active_op = None

            # P0 fix: Check drop_scheduled flag to prevent double-pop race
            if run_slot.close_requested and run_slot.in_flight_ops == 0 and not run_slot.drop_scheduled:
                popped = self._pop_run_slot_locked(run_slot.run_lease_id)
                if popped is not None:
                    task_key, popped_slot = popped
                    close_reason = (
                        "Closing run slot after in-flight "
                        f"{op_name}: {popped_slot.close_reason or 'close_requested'}"
                    )
                    close_after = (
                        task_key,
                        popped_slot.run_lease_id,
                        popped_slot,
                        close_reason,
                    )

        if close_after is not None:
            task_key, run_lease_id, slot_to_close, close_reason = close_after
            self._schedule_close(
                task_key,
                run_lease_id,
                slot_to_close,
                reason=close_reason,
            )

        # FIX-1: Handle timeout drop AFTER _finish_run_op completes to prevent TOCTOU race
        # This ensures in_flight_ops is decremented before lease removal
        if is_timeout_drop:
            logger.info(
                "Timeout drop deferred until after _finish_run_op: lease=%s op=%s",
                run_slot.run_lease_id,
                op_name,
            )
            await self._drop_resetting_run_for_timeout(
                run_slot.run_lease_id, run_slot, timeout=0.0  # timeout already logged earlier
            )

    async def _close_run_slot_under_lock(self, run_slot: RunSlot) -> None:
        async with run_slot.lock:
            run_slot.phase = "closing"
            try:
                await asyncio.wait_for(
                    run_slot.env.close(), timeout=self.close_session_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "env.close() timed out after %.1fs for lease=%s; proceeding to force_cleanup",
                    self.close_session_timeout,
                    run_slot.run_lease_id,
                )
                raise
            run_slot.phase = "closed"

    def _prune_done_closing_tasks(self) -> int:
        done = {task for task in self._closing_tasks if task.done()}
        self._closing_tasks.difference_update(done)
        for task in done:
            self._closing_task_started.pop(task, None)
            self._closing_task_labels.pop(task, None)
        return len(done)

    def _prune_done_force_cleanup_tasks(self) -> int:
        done = {task for task in self._force_cleanup_tasks if task.done()}
        self._force_cleanup_tasks.difference_update(done)
        for task in done:
            self._force_cleanup_task_started.pop(task, None)
            self._force_cleanup_task_labels.pop(task, None)
        return len(done)

    async def _close_run_slot_with_semaphore(self, run_slot: RunSlot) -> None:
        try:
            await asyncio.wait_for(
                self._close_sem.acquire(), timeout=self.close_queue_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting %.1fs for close semaphore lease=%s; "
                "proceeding to force_cleanup",
                self.close_queue_timeout,
                run_slot.run_lease_id,
            )
            raise
        try:
            await self._close_run_slot_under_lock(run_slot)
        finally:
            self._close_sem.release()

    async def _force_cleanup_after_close_failure(
        self, run_slot: RunSlot, run_lease_id: str, *, reason: str
    ) -> None:
        # STABILITY FIX: Increase timeout from 30s to 90s to handle Docker operations under load
        # Analysis shows 93 force cleanup timeouts; Docker container removal can take 60-90s under pressure
        timeout = _env_float("WORKER_FORCE_CLEANUP_TIMEOUT", 90.0)
        try:
            logger.warning(
                "Force cleanup starting for run session %s after %s (timeout=%.1fs)",
                run_lease_id,
                reason,
                timeout,
            )
            # P0 fix: Apply timeout here at the caller level; env.force_cleanup should not use nested timeout
            await asyncio.wait_for(run_slot.env.force_cleanup(reason=reason), timeout=timeout)
            logger.warning(
                "Force cleanup finished for run session %s after %s",
                run_lease_id,
                reason,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Force cleanup timed out for run session %s after %s (timeout=%.1fs)",
                run_lease_id,
                reason,
                timeout,
            )
        except Exception:
            logger.exception(
                "Force cleanup failed after %s for run session %s",
                reason,
                run_lease_id,
            )

    async def _close_run_slot(
        self, task_key: str, run_lease_id: str, run_slot: RunSlot, *, reason: str
    ) -> None:
        logger.warning("%s %s (task=%s)", reason, run_lease_id, task_key)
        try:
            await self._close_run_slot_with_semaphore(run_slot)
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out closing run session %s "
                "(queue_timeout=%.1fs session_timeout=%.1fs); dropping it "
                "from the pool so the close backlog can drain. Watchdog/preflight "
                "cleanup will remove any orphan Docker objects.",
                run_lease_id,
                self.close_queue_timeout,
                self.close_session_timeout,
            )
            await self._force_cleanup_after_close_failure(
                run_slot, run_lease_id, reason="close_timeout"
            )
        except asyncio.CancelledError:
            logger.warning(
                "Close task for run session %s was cancelled; forcing Docker "
                "cleanup before dropping it from the pool.",
                run_lease_id,
            )
            await asyncio.shield(
                self._force_cleanup_after_close_failure(
                    run_slot, run_lease_id, reason="close_cancelled"
                )
            )
            raise
        except Exception:
            logger.exception("Failed to close run session %s", run_lease_id)
            await self._force_cleanup_after_close_failure(
                run_slot, run_lease_id, reason="close_exception"
            )

    def _schedule_close(
        self, task_key: str, run_lease_id: str, run_slot: RunSlot, *, reason: str
    ) -> None:
        task = asyncio.create_task(
            self._close_run_slot(task_key, run_lease_id, run_slot, reason=reason)
        )
        self._closing_tasks.add(task)
        self._closing_task_started[task] = time.time()
        self._closing_task_labels[task] = f"{reason} {run_lease_id} task={task_key}"

        def _on_done(done_task: asyncio.Task) -> None:
            self._closing_tasks.discard(done_task)
            self._closing_task_started.pop(done_task, None)
            self._closing_task_labels.pop(done_task, None)

        task.add_done_callback(_on_done)

    def _schedule_force_cleanup_slots(
        self, slots: list[tuple[str, str, RunSlot]], *, reason: str
    ) -> None:
        if not slots:
            return
        task = asyncio.create_task(self._force_cleanup_slots(slots, reason=reason))
        self._force_cleanup_tasks.add(task)
        self._force_cleanup_task_started[task] = time.time()
        self._force_cleanup_task_labels[task] = (
            f"{reason} leases={','.join(rid for _tk, rid, _slot in slots[:8])}"
        )

        def _on_done(done_task: asyncio.Task) -> None:
            self._force_cleanup_tasks.discard(done_task)
            self._force_cleanup_task_started.pop(done_task, None)
            self._force_cleanup_task_labels.pop(done_task, None)

        task.add_done_callback(_on_done)

    def _schedule_close_requested_force_release(
        self, run_lease_id: str, *, reason: str
    ) -> None:
        if os.getenv("WORKER_CLOSE_REQUESTED_FORCE_RELEASE", "1") != "1":
            return
        existing = self._close_requested_release_tasks.get(run_lease_id)
        if existing is not None and not existing.done():
            return
        delay = max(
            0.0, _env_float("WORKER_CLOSE_REQUESTED_FORCE_RELEASE_AFTER", 30.0)
        )
        task = asyncio.create_task(
            self._force_release_close_requested_after_delay(
                run_lease_id,
                reason=reason,
                delay=delay,
            )
        )
        self._close_requested_release_tasks[run_lease_id] = task

        def _on_done(done_task: asyncio.Task) -> None:
            current = self._close_requested_release_tasks.get(run_lease_id)
            if current is done_task:
                self._close_requested_release_tasks.pop(run_lease_id, None)

        task.add_done_callback(_on_done)

    async def _force_release_close_requested_after_delay(
        self, run_lease_id: str, *, reason: str, delay: float
    ) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
        slot_to_force_cleanup: tuple[str, str, RunSlot] | None = None
        async with self._lock:
            task_key = self._run_to_task.get(run_lease_id)
            if task_key is None:
                return
            task_slot = self._tasks.get(task_key)
            run_slot = task_slot.runs.get(run_lease_id) if task_slot else None
            if run_slot is None or not run_slot.close_requested:
                return
            if run_slot.in_flight_ops <= 0 and not run_slot.lock.locked():
                popped = self._pop_run_slot_locked(run_lease_id)
                if popped is not None:
                    task_key, run_slot = popped
                    logger.warning(
                        "Force-releasing close_requested idle run lease=%s task=%s "
                        "reason=%s phase=%s",
                        run_lease_id,
                        task_key,
                        reason,
                        run_slot.phase,
                    )
                    self._schedule_close(
                        task_key,
                        run_lease_id,
                        run_slot,
                        reason=f"Force-releasing idle close_requested run: {reason}",
                    )
                return
            popped = self._pop_run_slot_locked(run_lease_id)
            if popped is None:
                return
            task_key, run_slot = popped
            logger.warning(
                "Force-releasing close_requested in-flight run lease=%s task=%s "
                "reason=%s phase=%s in_flight=%d active_op=%s %s",
                run_lease_id,
                task_key,
                reason,
                run_slot.phase,
                run_slot.in_flight_ops,
                run_slot.active_op,
                self._run_slot_container_ref(run_slot),
            )
            slot_to_force_cleanup = (task_key, run_lease_id, run_slot)

        if slot_to_force_cleanup is not None:
            self._schedule_force_cleanup_slots(
                [slot_to_force_cleanup],
                reason=f"close_requested_force_release:{reason}",
            )

    def _reap_idle_locked(self) -> list[tuple[str, str, RunSlot]]:
        now = time.time()
        expired_slots: list[tuple[str, str, RunSlot]] = []
        allocated_ttl = _env_float("WORKER_ALLOCATED_TTL", 120.0)

        expired_idem = [
            k
            for k, (_, ts) in self._idempotency.items()
            if now - ts > self.idempotency_ttl
        ]
        for k in expired_idem:
            self._idempotency.pop(k, None)

        for task_key, task_slot in list(self._tasks.items()):
            expired_runs: list[str] = []
            for rid, rslot in task_slot.runs.items():
                if rslot.in_flight_ops > 0 or rslot.lock.locked():
                    continue
                if rslot.close_requested:
                    continue
                if (
                    rslot.phase == "allocated"
                    and allocated_ttl > 0
                    and now - rslot.created_ts > allocated_ttl
                ):
                    expired_runs.append(rid)
                    continue
                if now - rslot.last_used_ts > self.run_idle_ttl:
                    expired_runs.append(rid)

            for rid in expired_runs:
                rslot = task_slot.runs.pop(rid, None)
                self._run_to_task.pop(rid, None)
                if rslot is not None:
                    expired_slots.append((task_key, rid, rslot))

            if task_slot.runs:
                task_slot.last_used_ts = max(
                    r.last_used_ts for r in task_slot.runs.values()
                )
            else:
                logger.info("Reaping empty task slot: %s", task_key)
                self._tasks.pop(task_key, None)

        return expired_slots

    @staticmethod
    def _stale_reason_for_run_slot(run_slot: RunSlot, now: float) -> tuple[str, float]:
        allocated_ttl = _env_float("WORKER_ALLOCATED_TTL", 120.0)
        # Keep this above WORKER_RESET_OPERATION_TIMEOUT so legitimate reset
        # operations are not reaped before their timeout handler runs.
        resetting_ttl = _env_float("WORKER_RESETTING_TTL", 2100.0)
        closing_ttl = _env_float("WORKER_CLOSING_REQUESTED_TTL", 300.0)
        created_age_sec = now - run_slot.created_ts
        reset_age_sec = (
            now - run_slot.reset_started_ts
            if run_slot.reset_started_ts is not None
            else 0.0
        )
        close_age_sec = (
            now - run_slot.close_requested_ts
            if run_slot.close_requested_ts is not None
            else 0.0
        )
        if (
            run_slot.phase == "allocated"
            and allocated_ttl > 0
            and created_age_sec >= allocated_ttl
        ):
            return "allocated_ttl_exceeded", created_age_sec
        if (
            run_slot.phase == "resetting"
            and resetting_ttl > 0
            and reset_age_sec >= resetting_ttl
        ):
            return "resetting_ttl_exceeded", reset_age_sec
        if (
            run_slot.close_requested
            and closing_ttl > 0
            and close_age_sec >= closing_ttl
        ):
            return "closing_requested_ttl_exceeded", close_age_sec
        return "", 0.0

    def _get_run_slot(self, run_lease_id: str) -> RunSlot:
        task_key = self._run_to_task.get(run_lease_id)
        if task_key is None:
            raise KeyError(f"Unknown run_lease_id: {run_lease_id}")
        task_slot = self._tasks.get(task_key)
        if task_slot is None:
            raise KeyError(f"Run {run_lease_id} points to missing task slot")
        run_slot = task_slot.runs.get(run_lease_id)
        if run_slot is None:
            raise KeyError(f"Run {run_lease_id} not found in task slot")
        return run_slot

    async def allocate(
        self, task_key: str, request_id: str | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            expired_slots = self._reap_idle_locked()

            if request_id:
                idem_key = (task_key, request_id)
                cached = self._idempotency.get(idem_key)
                if cached is not None:
                    run_lease_id, _ = cached
                    if run_lease_id in self._run_to_task:
                        logger.info(
                            "allocate_ok lease_id=%s task_key=%s request_id=%s reused=%s",
                            run_lease_id,
                            task_key,
                            request_id,
                            True,
                        )
                        return {"lease_id": run_lease_id, "reused": True}

            task_slot = self._tasks.get(task_key)
            if task_slot is None:
                if len(self._tasks) >= self.max_tasks:
                    raise CapacityError(
                        "TASK_SLOTS_EXHAUSTED",
                        f"Worker at task capacity: {len(self._tasks)}/{self.max_tasks}",
                    )
                task_slot = TaskSlot(task_key=task_key)
                self._tasks[task_key] = task_slot

            effective_max_runs = self._effective_max_runs_per_task(task_key)
            if len(task_slot.runs) >= effective_max_runs:
                raise CapacityError(
                    "RUN_SLOTS_EXHAUSTED",
                    f"Task {task_key} at run capacity: {len(task_slot.runs)}/{effective_max_runs}",
                )

            env = self._new_env()
            run_lease_id = f"run-{uuid.uuid4().hex[:16]}"
            run_slot = RunSlot(run_lease_id=run_lease_id, task_key=task_key, env=env)
            task_slot.runs[run_lease_id] = run_slot
            task_slot.last_used_ts = time.time()
            self._run_to_task[run_lease_id] = task_key

            if request_id:
                self._idempotency[(task_key, request_id)] = (run_lease_id, time.time())

        for tk, rid, rslot in expired_slots:
            self._schedule_close(tk, rid, rslot, reason="Reaping idle run slot")

        logger.info(
            "allocate_ok lease_id=%s task_key=%s request_id=%s reused=%s",
            run_lease_id,
            task_key,
            request_id or "",
            False,
        )
        return {"lease_id": run_lease_id, "reused": False}

    async def heartbeat(self, run_lease_id: str) -> None:
        run_slot = await self._begin_run_op(run_lease_id, "heartbeat")
        success = False
        try:
            async with run_slot.lock:
                success = True
        finally:
            await self._finish_run_op(run_slot, "heartbeat", success=success)

    @staticmethod
    def _reset_operation_timeout(timeouts: TaskTimeouts) -> float:
        configured = _env_float("WORKER_RESET_OPERATION_TIMEOUT", 0.0)
        if configured > 0:
            return configured
        return max(
            30.0,
            float(timeouts.ensure_image) + float(timeouts.reset_session) + 120.0,
        )

    async def _drop_resetting_run_for_timeout(
        self, run_lease_id: str, run_slot: RunSlot, *, timeout: float
    ) -> None:
        popped_slot: tuple[str, str, RunSlot] | None = None
        async with self._lock:
            task_key = self._run_to_task.get(run_lease_id)
            if task_key is None:
                return
            current = self._tasks.get(task_key)
            if current is None or current.runs.get(run_lease_id) is not run_slot:
                return
            # P0 fix: Set drop_scheduled flag to prevent _finish_run_op from also trying to pop
            run_slot.drop_scheduled = True
            popped = self._pop_run_slot_locked(run_lease_id)
            if popped is not None:
                popped_task_key, popped_run_slot = popped
                logger.warning(
                    "Dropping reset-timed-out run lease=%s task=%s timeout=%.1fs "
                    "phase=%s in_flight=%d %s",
                    run_lease_id,
                    popped_task_key,
                    timeout,
                    popped_run_slot.phase,
                    popped_run_slot.in_flight_ops,
                    self._run_slot_container_ref(popped_run_slot),
                )
                popped_slot = (popped_task_key, run_lease_id, popped_run_slot)

        if popped_slot is not None:
            self._schedule_force_cleanup_slots(
                [popped_slot],
                reason=f"reset_timeout:{timeout:.1f}s",
            )

    async def _acquire_reset_admission(self, run_lease_id: str) -> None:
        timeout = max(0.0, self.reset_admission_timeout)
        async with self._lock:
            self._reset_admission_waiting += 1
        try:
            try:
                if timeout > 0:
                    await asyncio.wait_for(
                        self._reset_admission_sem.acquire(), timeout=timeout
                    )
                else:
                    await self._reset_admission_sem.acquire()
            except asyncio.TimeoutError as exc:
                async with self._lock:
                    self._reset_admission_rejected += 1
                raise ResetAdmissionBacklogError(
                    run_lease_id,
                    timeout,
                    self.max_concurrent_resets,
                ) from exc
        finally:
            async with self._lock:
                self._reset_admission_waiting = max(
                    0, self._reset_admission_waiting - 1
                )

    async def _run_reset_once(
        self,
        run_lease_id: str,
        task_meta: dict[str, Any],
        run_ctx_payload: dict[str, Any] | None,
        task_timeouts: dict[str, Any] | None,
    ) -> dict[str, Any]:
        run_ctx = _build_run_ctx(
            run_ctx_payload, default_log_dir=self.output_root / "AgentRunner_Output"
        )
        timeouts = _parse_timeout_overrides(self.default_timeouts, task_timeouts)
        task_spec = _build_task_spec(task_meta)
        reset_timeout = self._reset_operation_timeout(timeouts)

        # Use a Task instead of a bare coroutine. wait_for() cancels its awaitable
        # on timeout, and bare coroutines cannot be awaited again after that.
        warn_after = _env_float("WORKER_RESET_WARN_AFTER", 300.0)
        warn_timeout = max(0.1, min(reset_timeout / 2.0, warn_after))
        remaining_timeout = max(0.1, reset_timeout - warn_timeout)
        is_timeout_drop = False
        success = False
        reset_task: asyncio.Task[tuple[str, list[dict[str, Any]]]] | None = None
        run_slot: RunSlot | None = None
        reset_admission_acquired = False

        def _consume_cancelled_reset_task(task: asyncio.Task[Any]) -> None:
            try:
                task.exception()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        try:
            await self._acquire_reset_admission(run_lease_id)
            reset_admission_acquired = True
            run_slot = await self._begin_run_op(run_lease_id, "reset")
            async with run_slot.lock:
                reset_task = asyncio.create_task(
                    run_slot.env.reset(
                        task_meta=task_meta,
                        task_spec=task_spec,
                        run_ctx=run_ctx,
                        timeouts=timeouts,
                    )
                )
                try:
                    done, _ = await asyncio.wait({reset_task}, timeout=warn_timeout)
                except asyncio.CancelledError:
                    reset_task.cancel()
                    raise

                if reset_task not in done:
                    logger.warning(
                        "Reset exceeds %.1fs (warn threshold), allowing %.1fs more: lease=%s",
                        warn_timeout,
                        remaining_timeout,
                        run_lease_id,
                    )

                try:
                    user_msg, tool_schemas = await asyncio.wait_for(
                        asyncio.shield(reset_task),
                        timeout=remaining_timeout,
                    )
                except asyncio.TimeoutError as exc:
                    if reset_task.done():
                        raise
                    is_timeout_drop = True
                    reset_task.cancel()
                    reset_task.add_done_callback(_consume_cancelled_reset_task)
                    raise TimeoutError(
                        f"WORKER_RESET_TIMEOUT lease_id={run_lease_id} "
                        f"after {reset_timeout:.1f}s"
                    ) from exc
                success = True
                return {"user_msg": user_msg, "tool_schemas": tool_schemas}
        finally:
            if not success and reset_task is not None and not reset_task.done():
                reset_task.cancel()
                reset_task.add_done_callback(_consume_cancelled_reset_task)
            if run_slot is not None:
                await self._finish_run_op(
                    run_slot,
                    "reset",
                    success=success,
                    is_timeout_drop=is_timeout_drop,
                )
            if reset_admission_acquired:
                self._reset_admission_sem.release()

    async def reset(
        self,
        run_lease_id: str,
        task_meta: dict[str, Any],
        run_ctx_payload: dict[str, Any] | None = None,
        task_timeouts: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(task_meta, dict):
            raise ValueError("task_meta must be a dict")

        request_id = str(request_id or "")
        future: asyncio.Task

        def _consume_reset_future_exception(task: asyncio.Task[Any]) -> None:
            try:
                task.exception()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        async with self._lock:
            run_slot = self._get_run_slot(run_lease_id)
            existing = run_slot.reset_future
            if request_id and run_slot.reset_request_id == request_id:
                if run_slot.reset_result is not None:
                    return dict(run_slot.reset_result)
                if existing is not None and not existing.done():
                    future = existing
                elif existing is not None and existing.done():
                    future = existing
                else:
                    future = asyncio.create_task(
                        self._run_reset_once(
                            run_lease_id,
                            task_meta,
                            run_ctx_payload,
                            task_timeouts,
                        )
                    )
                    run_slot.reset_future = future
            else:
                if existing is not None and not existing.done():
                    raise ResetInProgressError(run_lease_id, run_slot.reset_request_id)
                run_slot.reset_request_id = request_id or f"reset-{uuid.uuid4().hex[:16]}"
                run_slot.reset_result = None
                future = asyncio.create_task(
                    self._run_reset_once(
                        run_lease_id,
                        task_meta,
                        run_ctx_payload,
                        task_timeouts,
                    )
                )
                run_slot.reset_future = future

        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError as exc:
            if not future.done():
                future.cancel()
                future.add_done_callback(_consume_reset_future_exception)
            async with self._lock:
                try:
                    run_slot = self._get_run_slot(run_lease_id)
                except KeyError:
                    pass
                else:
                    if run_slot.reset_future is future:
                        run_slot.reset_future = None
                        run_slot.reset_result = None
            raise TimeoutError(
                f"WORKER_RESET_CANCELLED lease_id={run_lease_id} request_id={request_id}"
            ) from exc
        except Exception:
            raise
        else:
            async with self._lock:
                try:
                    run_slot = self._get_run_slot(run_lease_id)
                except KeyError:
                    logger.info(
                        "Reset completed after lease=%s was already removed; "
                        "returning reset result without caching it.",
                        run_lease_id,
                    )
                    return result
                if run_slot.reset_future is future:
                    run_slot.reset_result = dict(result)
            return result

    async def exec_tool(
        self, run_lease_id: str, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        run_slot = await self._begin_run_op(run_lease_id, "exec_tool")
        success = False
        try:
            async with run_slot.lock:
                observation = await run_slot.env.exec_tool(tool_name, arguments or {})
                success = True
                return str(observation)
        finally:
            await self._finish_run_op(run_slot, "exec_tool", success=success)

    async def evaluate(
        self, run_lease_id: str, trajectory: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any] | None]:
        run_slot = await self._begin_run_op(run_lease_id, "evaluate")
        success = False
        try:
            async with run_slot.lock:
                score = await run_slot.env.evaluate(trajectory)
                details = run_slot.env.last_eval_details()
                success = True
                return float(score), details
        finally:
            await self._finish_run_op(run_slot, "evaluate", success=success)

    async def close_run(self, run_lease_id: str, *, reason: str = "external_close") -> bool:
        close_now: tuple[str, str, RunSlot] | None = None
        async with self._lock:
            task_key = self._run_to_task.get(run_lease_id)
            if task_key is None:
                logger.debug(
                    "close_run: lease %s already gone, nothing to do.", run_lease_id
                )
                return False
            task_slot = self._tasks.get(task_key)
            run_slot = task_slot.runs.get(run_lease_id) if task_slot else None
            if run_slot is None:
                return False

            if run_slot.close_requested:
                logger.info(
                    "close_run: duplicate close ignored lease=%s task=%s phase=%s "
                    "in_flight=%d reason=%s %s",
                    run_lease_id,
                    task_key,
                    run_slot.phase,
                    run_slot.in_flight_ops,
                    run_slot.close_reason,
                    self._run_slot_container_ref(run_slot),
                )
                return True

            run_slot.close_requested = True
            run_slot.close_reason = reason
            run_slot.close_requested_ts = time.time()
            stack = "".join(traceback.format_stack(limit=8))
            logger.warning(
                "close_run requested lease=%s task=%s phase=%s in_flight=%d "
                "first_step=%s evaluate_done=%s reason=%s %s\nClose request stack:\n%s",
                run_lease_id,
                task_key,
                run_slot.phase,
                run_slot.in_flight_ops,
                run_slot.first_step_ts is not None,
                run_slot.evaluate_completed_ts is not None,
                reason,
                self._run_slot_container_ref(run_slot),
                stack,
            )
            if run_slot.in_flight_ops > 0 or run_slot.lock.locked():
                run_slot.phase = "closing_requested"
                self._schedule_close_requested_force_release(
                    run_lease_id, reason=reason
                )
                return True

            popped = self._pop_run_slot_locked(run_lease_id)
            if popped is not None:
                task_key, run_slot = popped
                close_now = (task_key, run_lease_id, run_slot)

        if close_now is not None:
            task_key, run_lease_id, run_slot = close_now
            self._schedule_close(task_key, run_lease_id, run_slot, reason="Closing run slot")
        return True

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            self._prune_done_closing_tasks()
            self._prune_done_force_cleanup_tasks()
            now = time.time()
            allocated_ttl = _env_float("WORKER_ALLOCATED_TTL", 120.0)
            resetting_ttl = _env_float("WORKER_RESETTING_TTL", 2100.0)
            closing_ttl = _env_float("WORKER_CLOSING_REQUESTED_TTL", 300.0)
            close_ages = [
                now - started for started in self._closing_task_started.values()
            ]
            force_cleanup_ages = [
                now - started for started in self._force_cleanup_task_started.values()
            ]
            pending_close_age_sec = {
                "min": round(min(close_ages), 1) if close_ages else 0.0,
                "max": round(max(close_ages), 1) if close_ages else 0.0,
                "over_close_timeout": sum(
                    1 for age in close_ages if age >= self.close_task_timeout
                ),
            }
            pending_force_cleanup_age_sec = {
                "min": round(min(force_cleanup_ages), 1) if force_cleanup_ages else 0.0,
                "max": round(max(force_cleanup_ages), 1) if force_cleanup_ages else 0.0,
            }
            tasks_info: dict[str, Any] = {}
            active_container_ids: set[str] = set()
            active_container_names: set[str] = set()
            active_trial_names: set[str] = set()
            active_project_names: set[str] = set()
            active_task_ids: set[str] = set()
            phase_counts: dict[str, int] = {}
            stale_runs: list[dict[str, Any]] = []
            reset_ages = []
            total_runs = 0
            in_flight_runs = 0
            closing_requested_runs = 0
            for tk, ts in self._tasks.items():
                task_id = _task_id_from_ref(tk)
                if task_id:
                    active_task_ids.add(task_id)
                run_details = {}
                for rid, rslot in ts.runs.items():
                    phase_counts[rslot.phase] = phase_counts.get(rslot.phase, 0) + 1
                    if rslot.in_flight_ops > 0:
                        in_flight_runs += 1
                    if rslot.close_requested:
                        closing_requested_runs += 1
                    container_info = self._run_slot_container_info(rslot)
                    for key in ("id", "short_id"):
                        value = container_info.get(key)
                        if isinstance(value, str) and value:
                            active_container_ids.add(value)
                    container_name = container_info.get("name")
                    if isinstance(container_name, str) and container_name:
                        active_container_names.add(container_name)
                        active_project_names.update(
                            _docker_name_variants(container_name)
                        )
                        task_id = _task_id_from_ref(container_name)
                        if task_id:
                            active_task_ids.add(task_id)
                    trial_name = container_info.get("trial_name")
                    if isinstance(trial_name, str) and trial_name:
                        active_trial_names.add(trial_name)
                        active_project_names.update(_docker_name_variants(trial_name))
                        task_id = _task_id_from_ref(trial_name)
                        if task_id:
                            active_task_ids.add(task_id)
                    created_age_sec = now - rslot.created_ts
                    reset_age_sec = (
                        now - rslot.reset_started_ts
                        if rslot.reset_started_ts is not None
                        else 0.0
                    )
                    if rslot.phase == "resetting":
                        reset_ages.append(reset_age_sec)
                    close_age_sec = (
                        now - rslot.close_requested_ts
                        if rslot.close_requested_ts is not None
                        else 0.0
                    )
                    stale_reason, stale_age_sec = self._stale_reason_for_run_slot(
                        rslot, now
                    )
                    if stale_reason:
                        stale_runs.append(
                            {
                                "lease_id": rid,
                                "task_key": tk,
                                "phase": rslot.phase,
                                "reason": stale_reason,
                                "age_sec": round(stale_age_sec, 1),
                                "in_flight_ops": rslot.in_flight_ops,
                                "active_op": rslot.active_op,
                                "close_requested": rslot.close_requested,
                                "container": container_info,
                            }
                        )
                    run_details[rid] = {
                        "phase": rslot.phase,
                        "in_flight_ops": rslot.in_flight_ops,
                        "active_op": rslot.active_op,
                        "close_requested": rslot.close_requested,
                        "age_sec": round(now - rslot.last_used_ts, 1),
                        "created_age_sec": round(created_age_sec, 1),
                        "reset_age_sec": round(reset_age_sec, 1),
                        "close_requested_age_sec": round(close_age_sec, 1),
                        "first_step": rslot.first_step_ts is not None,
                        "evaluate_done": rslot.evaluate_completed_ts is not None,
                        "container": container_info,
                    }
                tasks_info[tk] = {
                    "active_runs": len(ts.runs),
                    "max_runs": self._effective_max_runs_per_task(tk),
                    "runs": run_details,
                }
                total_runs += len(ts.runs)

            return {
                "max_tasks": self.max_tasks,
                "active_tasks": len(self._tasks),
                "max_runs_per_task": self.max_runs_per_task,
                "serial_task_ids": sorted(self._serial_task_ids),
                "task_max_runs_overrides": dict(
                    sorted(self._task_max_runs_overrides.items())
                ),
                "auto_serialize_unsafe_compose": self._auto_serialize_unsafe_compose,
                "total_active_runs": total_runs,
                "in_flight_runs": in_flight_runs,
                "closing_requested_runs": closing_requested_runs,
                "pending_closes": len(self._closing_tasks),
                "pending_force_cleanups": len(self._force_cleanup_tasks),
                "reset_admission": {
                    "max_concurrent": self.max_concurrent_resets,
                    "available": int(getattr(self._reset_admission_sem, "_value", 0)),
                    "waiting": self._reset_admission_waiting,
                    "rejected": self._reset_admission_rejected,
                    "timeout": self.reset_admission_timeout,
                },
                "docker_image_build": docker_image_build_status(),
                "close_queue_timeout": self.close_queue_timeout,
                "close_session_timeout": self.close_session_timeout,
                "close_task_timeout": self.close_task_timeout,
                "pending_close_age_sec": pending_close_age_sec,
                "pending_force_cleanup_age_sec": pending_force_cleanup_age_sec,
                "resetting_age_sec": {
                    "min": round(min(reset_ages), 1) if reset_ages else 0.0,
                    "max": round(max(reset_ages), 1) if reset_ages else 0.0,
                },
                "phase_counts": phase_counts,
                "stale_runs": stale_runs,
                "active_container_ids": sorted(active_container_ids),
                "active_container_names": sorted(active_container_names),
                "active_trial_names": sorted(active_trial_names),
                "active_project_names": sorted(active_project_names),
                "active_task_ids": sorted(active_task_ids),
                "tasks": tasks_info,
            }

    async def repair_pending_closes(
        self,
        *,
        reason: str,
        max_active_runs: int = 0,
        cancel_timeout: float = 5.0,
        min_age: float | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        if min_age is None:
            min_age = max(0.0, self.close_task_timeout + 5.0)
        async with self._lock:
            pruned_done = self._prune_done_closing_tasks()
            active_runs = sum(len(ts.runs) for ts in self._tasks.values())
            pending_before_cancel = len(self._closing_tasks)
            if max_active_runs >= 0 and active_runs > max_active_runs:
                return {
                    "repaired": False,
                    "reason": "active_runs_above_limit",
                    "active_runs": active_runs,
                    "max_active_runs": max_active_runs,
                    "pending_closes": pending_before_cancel,
                    "pruned_done": pruned_done,
                }
            tasks_to_cancel = [
                task
                for task in self._closing_tasks
                if now - self._closing_task_started.get(task, now) >= min_age
            ]
            skipped_young = pending_before_cancel - len(tasks_to_cancel)

        # P0 fix: Cancel tasks and wait for their cancellation handlers using shield
        cancelled = 0
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                cancelled += 1

        if tasks_to_cancel:
            try:
                # P0 fix: Use asyncio.shield to ensure cancellation handlers complete
                await asyncio.shield(
                    asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=max(0.1, cancel_timeout),
                    )
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out waiting for pending close task cancellation: "
                    "reason=%s pending=%d timeout=%.1fs",
                    reason,
                    len(tasks_to_cancel),
                    cancel_timeout,
                )

        async with self._lock:
            pruned_after_cancel = self._prune_done_closing_tasks()
            pending_after = len(self._closing_tasks)

        logger.warning(
            "Repaired pending close tasks: reason=%s active_runs=%d "
            "min_age=%.1fs pruned_done=%d cancelled=%d skipped_young=%d "
            "pruned_after_cancel=%d pending_after=%d",
            reason,
            active_runs,
            min_age,
            pruned_done,
            cancelled,
            skipped_young,
            pruned_after_cancel,
            pending_after,
        )
        return {
            "repaired": True,
            "reason": reason,
            "active_runs": active_runs,
            "max_active_runs": max_active_runs,
            "min_age": min_age,
            "pending_before_cancel": pending_before_cancel,
            "pruned_done": pruned_done,
            "cancelled": cancelled,
            "skipped_young": skipped_young,
            "pruned_after_cancel": pruned_after_cancel,
            "pending_after": pending_after,
        }

    async def repair_stale_runs(
        self,
        *,
        reason: str,
        min_age: float = 0.0,
        max_repairs: int = 20,
        wait_for_cleanup: bool = True,
    ) -> dict[str, Any]:
        now = time.time()
        slots_to_force_cleanup: list[tuple[str, str, RunSlot]] = []
        repaired_runs: list[dict[str, Any]] = []
        async with self._lock:
            self._prune_done_closing_tasks()
            for task_key, task_slot in list(self._tasks.items()):
                for run_lease_id, run_slot in list(task_slot.runs.items()):
                    stale_reason, stale_age_sec = self._stale_reason_for_run_slot(
                        run_slot, now
                    )
                    if not stale_reason or stale_age_sec < min_age:
                        continue
                    popped = self._pop_run_slot_locked(run_lease_id)
                    if popped is None:
                        continue
                    popped_task_key, popped_slot = popped
                    slots_to_force_cleanup.append(
                        (popped_task_key, run_lease_id, popped_slot)
                    )
                    repaired_runs.append(
                        {
                            "lease_id": run_lease_id,
                            "task_key": popped_task_key,
                            "phase": popped_slot.phase,
                            "reason": stale_reason,
                            "age_sec": round(stale_age_sec, 1),
                            "in_flight_ops": popped_slot.in_flight_ops,
                            "active_op": popped_slot.active_op,
                            "close_requested": popped_slot.close_requested,
                            "container": self._run_slot_container_info(popped_slot),
                        }
                    )
                    if max_repairs > 0 and len(repaired_runs) >= max_repairs:
                        break
                if max_repairs > 0 and len(repaired_runs) >= max_repairs:
                    break

        if slots_to_force_cleanup and wait_for_cleanup:
            await self._force_cleanup_slots(
                slots_to_force_cleanup,
                reason=f"repair_stale_runs:{reason}",
            )
        elif slots_to_force_cleanup:
            self._schedule_force_cleanup_slots(
                slots_to_force_cleanup,
                reason=f"repair_stale_runs:{reason}",
            )

        return {
            "repaired": bool(repaired_runs),
            "reason": reason,
            "min_age": min_age,
            "max_repairs": max_repairs,
            "wait_for_cleanup": wait_for_cleanup,
            "repaired_count": len(repaired_runs),
            "repaired_runs": repaired_runs,
        }

    async def repair_close_requested_runs(
        self,
        *,
        reason: str,
        min_age: float = 0.0,
        max_repairs: int = 20,
        wait_for_cleanup: bool = False,
    ) -> dict[str, Any]:
        now = time.time()
        slots_to_force_cleanup: list[tuple[str, str, RunSlot]] = []
        repaired_runs: list[dict[str, Any]] = []
        async with self._lock:
            self._prune_done_closing_tasks()
            self._prune_done_force_cleanup_tasks()
            for task_key, task_slot in list(self._tasks.items()):
                for run_lease_id, run_slot in list(task_slot.runs.items()):
                    if not run_slot.close_requested:
                        continue
                    close_age_sec = (
                        now - run_slot.close_requested_ts
                        if run_slot.close_requested_ts is not None
                        else now - run_slot.last_used_ts
                    )
                    if close_age_sec < min_age:
                        continue
                    popped = self._pop_run_slot_locked(run_lease_id)
                    if popped is None:
                        continue
                    popped_task_key, popped_slot = popped
                    slots_to_force_cleanup.append(
                        (popped_task_key, run_lease_id, popped_slot)
                    )
                    repaired_runs.append(
                        {
                            "lease_id": run_lease_id,
                            "task_key": popped_task_key,
                            "phase": popped_slot.phase,
                            "reason": "close_requested_capacity_pressure",
                            "age_sec": round(close_age_sec, 1),
                            "in_flight_ops": popped_slot.in_flight_ops,
                            "active_op": popped_slot.active_op,
                            "close_requested": popped_slot.close_requested,
                            "container": self._run_slot_container_info(popped_slot),
                        }
                    )
                    if max_repairs > 0 and len(repaired_runs) >= max_repairs:
                        break
                if max_repairs > 0 and len(repaired_runs) >= max_repairs:
                    break

        if slots_to_force_cleanup and wait_for_cleanup:
            await self._force_cleanup_slots(
                slots_to_force_cleanup,
                reason=f"repair_close_requested_runs:{reason}",
            )
        elif slots_to_force_cleanup:
            self._schedule_force_cleanup_slots(
                slots_to_force_cleanup,
                reason=f"repair_close_requested_runs:{reason}",
            )

        return {
            "repaired": bool(repaired_runs),
            "reason": reason,
            "min_age": min_age,
            "max_repairs": max_repairs,
            "wait_for_cleanup": wait_for_cleanup,
            "repaired_count": len(repaired_runs),
            "repaired_runs": repaired_runs,
        }

    async def repair_resetting_runs(
        self,
        *,
        reason: str,
        min_age: float = 0.0,
        max_repairs: int = 20,
        wait_for_cleanup: bool = False,
    ) -> dict[str, Any]:
        now = time.time()
        slots_to_force_cleanup: list[tuple[str, str, RunSlot]] = []
        repaired_runs: list[dict[str, Any]] = []
        async with self._lock:
            self._prune_done_closing_tasks()
            self._prune_done_force_cleanup_tasks()
            for task_key, task_slot in list(self._tasks.items()):
                for run_lease_id, run_slot in list(task_slot.runs.items()):
                    if run_slot.phase != "resetting":
                        continue
                    reset_age_sec = (
                        now - run_slot.reset_started_ts
                        if run_slot.reset_started_ts is not None
                        else now - run_slot.last_used_ts
                    )
                    if reset_age_sec < min_age:
                        continue
                    if (
                        run_slot.reset_future is not None
                        and not run_slot.reset_future.done()
                    ):
                        run_slot.reset_future.cancel()
                    popped = self._pop_run_slot_locked(run_lease_id)
                    if popped is None:
                        continue
                    popped_task_key, popped_slot = popped
                    slots_to_force_cleanup.append(
                        (popped_task_key, run_lease_id, popped_slot)
                    )
                    repaired_runs.append(
                        {
                            "lease_id": run_lease_id,
                            "task_key": popped_task_key,
                            "phase": popped_slot.phase,
                            "reason": "resetting_storm_repair",
                            "age_sec": round(reset_age_sec, 1),
                            "in_flight_ops": popped_slot.in_flight_ops,
                            "active_op": popped_slot.active_op,
                            "close_requested": popped_slot.close_requested,
                            "container": self._run_slot_container_info(popped_slot),
                        }
                    )
                    if max_repairs > 0 and len(repaired_runs) >= max_repairs:
                        break
                if max_repairs > 0 and len(repaired_runs) >= max_repairs:
                    break

        if slots_to_force_cleanup and wait_for_cleanup:
            await self._force_cleanup_slots(
                slots_to_force_cleanup,
                reason=f"repair_resetting_runs:{reason}",
            )
        elif slots_to_force_cleanup:
            self._schedule_force_cleanup_slots(
                slots_to_force_cleanup,
                reason=f"repair_resetting_runs:{reason}",
            )

        return {
            "repaired": bool(repaired_runs),
            "reason": reason,
            "min_age": min_age,
            "max_repairs": max_repairs,
            "wait_for_cleanup": wait_for_cleanup,
            "repaired_count": len(repaired_runs),
            "repaired_runs": repaired_runs,
        }

    async def _force_cleanup_slots(
        self, slots: list[tuple[str, str, RunSlot]], *, reason: str
    ) -> None:
        if not slots:
            return
        per_slot_timeout = _env_float("WORKER_FORCE_CLEANUP_TIMEOUT", 90.0)
        timeout = _env_float(
            "WORKER_SHUTDOWN_FORCE_CLEANUP_TIMEOUT",
            max(30.0, per_slot_timeout + 10.0),
        )
        logger.warning(
            "Batch force cleanup starting for %d run slot(s), reason=%s timeout=%.1fs",
            len(slots),
            reason,
            timeout,
        )
        cleanup_tasks: dict[asyncio.Task[Any], str] = {}
        for task_key, run_lease_id, run_slot in slots:
            task = asyncio.create_task(
                self._force_cleanup_after_close_failure(
                    run_slot,
                    run_lease_id,
                    reason=reason,
                )
            )
            cleanup_tasks[task] = f"{task_key}:{run_lease_id}"

        done, pending = await asyncio.wait(cleanup_tasks, timeout=timeout)
        if pending:
            logger.warning(
                "Batch force cleanup timed out with %d cleanup task(s) still pending: %s",
                len(pending),
                ",".join(cleanup_tasks[task] for task in pending),
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        if done:
            await asyncio.gather(*done, return_exceptions=True)
        logger.warning("Batch force cleanup finished for reason=%s", reason)

    async def periodic_reap(self, interval: float = 60.0) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                async with self._lock:
                    expired_slots = self._reap_idle_locked()
                for tk, rid, rslot in expired_slots:
                    self._schedule_close(
                        tk, rid, rslot, reason="Periodic reaper: idle run slot"
                    )
                if expired_slots:
                    logger.info(
                        "Periodic reaper cleaned up %d idle run slots",
                        len(expired_slots),
                    )
                # P0 fix: Automatic shim cleanup every 50 resets or when pressure detected
                await self._maybe_cleanup_shims()
                await self._maybe_cleanup_orphan_docker_containers()
            except Exception:
                logger.exception("Periodic reaper error")

    async def _maybe_cleanup_orphan_docker_containers(self) -> None:
        if os.getenv("WORKER_ORPHAN_DOCKER_SWEEP", "1") != "1":
            return
        now = time.time()
        if now < self._orphan_sweep_backoff_until:
            return
        interval = max(1.0, _env_float("WORKER_ORPHAN_DOCKER_SWEEP_INTERVAL", 60.0))
        if now - self._last_orphan_sweep_ts < interval:
            return
        self._last_orphan_sweep_ts = now

        async with self._lock:
            (
                active_container_names,
                active_project_names,
                active_task_ids,
            ) = self._active_docker_refs_locked()

        min_age_sec = max(0.0, _env_float("WORKER_ORPHAN_DOCKER_SWEEP_MIN_AGE", 600.0))
        max_remove = _env_int("WORKER_ORPHAN_DOCKER_SWEEP_MAX_REMOVE", 128)
        timeout = max(1.0, _env_float("WORKER_ORPHAN_DOCKER_SWEEP_TIMEOUT", 30.0))
        try:
            removed = await asyncio.wait_for(
                asyncio.to_thread(
                    force_remove_orphan_docker_objects,
                    active_container_names=active_container_names,
                    active_project_names=active_project_names,
                    active_task_ids=active_task_ids,
                    reason="periodic_reap",
                    min_age_sec=min_age_sec,
                    max_remove=max_remove,
                ),
                timeout=timeout,
            )
            if removed < 0:
                self._record_orphan_sweep_failure("docker_ps_failed")
                return
            self._orphan_sweep_fail_streak = 0
            self._orphan_sweep_backoff_until = 0.0
            if removed:
                logger.warning(
                    "Periodic orphan Docker sweep removed %d stale container(s) "
                    "active_containers=%d active_projects=%d active_tasks=%d min_age=%.1fs",
                    removed,
                    len(active_container_names),
                    len(active_project_names),
                    len(active_task_ids),
                    min_age_sec,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "Periodic orphan Docker sweep timed out after %.1fs "
                "active_containers=%d active_projects=%d active_tasks=%d",
                timeout,
                len(active_container_names),
                len(active_project_names),
                len(active_task_ids),
            )
            self._record_orphan_sweep_failure(f"timeout_after_{timeout:.1f}s")
        except Exception:
            self._record_orphan_sweep_failure("exception")
            logger.exception("Periodic orphan Docker sweep failed")

    def _record_orphan_sweep_failure(self, reason: str) -> None:
        self._orphan_sweep_fail_streak += 1
        base = max(1.0, _env_float("WORKER_ORPHAN_DOCKER_SWEEP_BACKOFF_BASE", 120.0))
        max_delay = max(base, _env_float("WORKER_ORPHAN_DOCKER_SWEEP_BACKOFF_MAX", 900.0))
        delay = min(max_delay, base * (2 ** min(self._orphan_sweep_fail_streak - 1, 6)))
        self._orphan_sweep_backoff_until = time.time() + delay
        logger.warning(
            "Periodic orphan Docker sweep failed (%s); backoff %.1fs streak=%d",
            reason,
            delay,
            self._orphan_sweep_fail_streak,
        )

    async def _maybe_cleanup_shims(self) -> None:
        """P0 fix: Proactively clean Docker shims to prevent resource exhaustion."""
        try:
            now = time.time()
            cleanup_interval = _env_float("WORKER_SHIM_CLEANUP_INTERVAL", 600.0)  # 10 min default
            reset_trigger = _env_int("WORKER_SHIM_CLEANUP_RESET_COUNT", 50)  # every 50 resets
            pressure_threshold = _env_int("WORKER_SHIM_CLEANUP_PRESSURE_THRESHOLD", 140)  # cleanup at 140 shims

            should_cleanup = False
            reason = ""

            # Check if enough time has passed since last cleanup
            if now - self._last_shim_cleanup_ts < cleanup_interval:
                return

            # Trigger 1: Reset count threshold
            if reset_trigger > 0 and self._reset_count >= reset_trigger:
                should_cleanup = True
                reason = f"reset_count={self._reset_count}>={reset_trigger}"

            # Trigger 2: Shim pressure threshold
            pressure = worker_pressure_stats()
            shim_count = int(pressure.get("shim", 0))
            if shim_count >= pressure_threshold:
                should_cleanup = True
                reason = f"shim_pressure={shim_count}>={pressure_threshold}"

            if not should_cleanup:
                return

            logger.warning(
                "Triggering automatic Docker shim cleanup: reason=%s shim_count=%d reset_count=%d",
                reason,
                shim_count,
                self._reset_count,
            )

            # Run docker system prune in background with timeout
            # P0 fix: Add fallback if prune hangs; close stderr pipe immediately to prevent fd leak
            cleanup_timeout = _env_float("WORKER_SHIM_CLEANUP_TIMEOUT", 30.0)
            proc = None
            try:
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        "docker",
                        "system",
                        "prune",
                        "-f",
                        "--volumes=false",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,  # P0 fix: Use DEVNULL to avoid fd leak
                    ),
                    timeout=cleanup_timeout,
                )
                await asyncio.wait_for(proc.wait(), timeout=cleanup_timeout)

                # Verify cleanup reduced shim count
                new_pressure = worker_pressure_stats(force=True)
                new_shim_count = int(new_pressure.get("shim", 0))
                logger.warning(
                    "Docker shim cleanup completed: shim_count %d→%d reset_count %d→0",
                    shim_count,
                    new_shim_count,
                    self._reset_count,
                )

                # Reset counters and timestamp
                self._reset_count = 0
                self._last_shim_cleanup_ts = now

            except asyncio.TimeoutError:
                logger.warning(
                    "Docker shim cleanup timed out after %.1fs; skipping and relying on watchdog (non-fatal)",
                    cleanup_timeout,
                )
                # P0 fix: Kill hung subprocess
                if proc is not None:
                    try:
                        proc.kill()
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except Exception:
                        pass
                # P0 fix: Still update timestamp to prevent retry storms
                self._last_shim_cleanup_ts = now
            except Exception as exc:
                logger.warning(
                    "Docker shim cleanup failed: %s (non-fatal, will retry next cycle)",
                    exc,
                )
                # P0 fix: Update timestamp on failure to prevent tight retry loop
                self._last_shim_cleanup_ts = now

        except Exception:
            logger.exception("Error in _maybe_cleanup_shims (non-fatal)")

    async def shutdown(self) -> None:
        async with self._lock:
            slots_to_close: list[tuple[str, str, RunSlot]] = []
            for task_key, task_slot in self._tasks.items():
                for run_lease_id, run_slot in task_slot.runs.items():
                    slots_to_close.append((task_key, run_lease_id, run_slot))
            self._tasks.clear()
            self._run_to_task.clear()
            self._idempotency.clear()

        for task_key, run_lease_id, run_slot in slots_to_close:
            self._schedule_close(
                task_key,
                run_lease_id,
                run_slot,
                reason="Closing run slot during shutdown",
            )

        if self._closing_tasks:
            logger.info(
                "Shutdown: waiting for %d pending close tasks...",
                len(self._closing_tasks),
            )
            shutdown_timeout = _env_float(
                "WORKER_SHUTDOWN_CLOSE_TASKS_TIMEOUT",
                max(5.0, self.close_task_timeout + 5.0),
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._closing_tasks, return_exceptions=True),
                    timeout=shutdown_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Shutdown timed out after %.1fs with %d pending close tasks; "
                    "cancelling them and forcing Docker cleanup.",
                    shutdown_timeout,
                    len(self._closing_tasks),
                )
                for task in list(self._closing_tasks):
                    task.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *self._closing_tasks, return_exceptions=True
                        ),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Shutdown cancellation wait timed out; exiting with %d "
                        "close task(s) still pending.",
                        len(self._closing_tasks),
                    )
                await self._force_cleanup_slots(
                    slots_to_close,
                    reason="shutdown_close_timeout",
                )
            else:
                await self._force_cleanup_slots(
                    slots_to_close,
                    reason="shutdown_final_sweep",
                )
        if self._force_cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *self._force_cleanup_tasks, return_exceptions=True
                    ),
                    timeout=_env_float("WORKER_SHUTDOWN_FORCE_CLEANUP_TASKS_TIMEOUT", 10.0),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Shutdown timed out waiting for %d background force cleanup task(s)",
                    len(self._force_cleanup_tasks),
                )


POOL: WorkerPool | None = None


@app.get("/healthz")
async def healthz() -> JSONResponse:
    try:
        pending_closes = 0
        pool_status: dict[str, Any] | None = None
        if POOL is not None:
            pool_status = await POOL.status()
            pending_closes = int(pool_status.get("pending_closes", 0))
        assert_worker_has_capacity_for_docker(
            phase="health",
            pending_closes=pending_closes,
            pool_status=pool_status if POOL is not None else None,
        )
        return JSONResponse({"ok": True})
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "code": exc.code,
                "error": exc.message,
                "details": exc.details,
            },
            status_code=503,
        )


@app.get("/status")
async def status() -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    disk: dict[str, Any] | None = None
    pressure: dict[str, Any] | None = None
    disk_ok = True
    disk_error: str | None = None
    pool_status = await POOL.status()
    try:
        disk = docker_data_root_stats()
        pressure = worker_pressure_stats()
        assert_worker_has_capacity_for_docker(
            phase="health",
            pending_closes=int(pool_status.get("pending_closes", 0)),
            pool_status=pool_status,
        )
    except ResourcePressureError as exc:
        disk_ok = False
        disk_error = exc.message
        pressure = exc.details
    except Exception as exc:
        disk_ok = False
        disk_error = str(exc)
    return JSONResponse(
        {
            "ok": True,
            "pool": pool_status,
            "docker_data_root": disk,
            "resource_pressure": pressure,
            "admission_ok": disk_ok,
            "admission_error": disk_error,
        }
    )


@app.get("/readyz")
async def readyz() -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {
                "ok": False,
                "code": "POOL_NOT_INITIALIZED",
                "error": "Pool is not initialized",
            },
            status_code=503,
        )

    pool_status = await POOL.status()
    try:
        assert_worker_has_capacity_for_docker(
            phase="health",
            pending_closes=int(pool_status.get("pending_closes", 0)),
            pool_status=pool_status,
        )
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "code": exc.code,
                "error": exc.message,
                "details": exc.details,
                "pool": pool_status,
            },
            status_code=503,
        )

    stale_runs = pool_status.get("stale_runs", [])
    if stale_runs:
        return JSONResponse(
            {
                "ok": False,
                "code": "WORKER_STALE_RUNS",
                "error": f"Worker has {len(stale_runs)} stale run(s)",
                "stale_runs": stale_runs[:20],
                "pool": pool_status,
            },
            status_code=503,
        )

    return JSONResponse({"ok": True, "pool": pool_status})


@app.get("/metrics")
async def metrics() -> Response:
    lines = [
        "# HELP openclaw_worker_up Worker process is serving HTTP.",
        "# TYPE openclaw_worker_up gauge",
        "openclaw_worker_up 1",
    ]
    if POOL is None:
        lines.extend(
            [
                "# HELP openclaw_worker_pool_initialized Worker pool initialization state.",
                "# TYPE openclaw_worker_pool_initialized gauge",
                "openclaw_worker_pool_initialized 0",
            ]
        )
        return Response("\n".join(lines) + "\n", media_type="text/plain")

    pool_status = await POOL.status()
    gauges = {
        "active_tasks": pool_status.get("active_tasks", 0),
        "total_active_runs": pool_status.get("total_active_runs", 0),
        "in_flight_runs": pool_status.get("in_flight_runs", 0),
        "closing_requested_runs": pool_status.get("closing_requested_runs", 0),
        "pending_closes": pool_status.get("pending_closes", 0),
        "stale_runs": len(pool_status.get("stale_runs", []) or []),
    }
    lines.extend(
        [
            "# HELP openclaw_worker_pool_initialized Worker pool initialization state.",
            "# TYPE openclaw_worker_pool_initialized gauge",
            "openclaw_worker_pool_initialized 1",
            "# HELP openclaw_worker_pool_gauge Worker pool gauges.",
            "# TYPE openclaw_worker_pool_gauge gauge",
        ]
    )
    for name, value in gauges.items():
        lines.append(
            f'openclaw_worker_pool_gauge{{name="{name}"}} {int(value or 0)}'
        )

    phase_counts = pool_status.get("phase_counts", {})
    if isinstance(phase_counts, dict):
        lines.extend(
            [
                "# HELP openclaw_worker_run_phase_count Active run count by phase.",
                "# TYPE openclaw_worker_run_phase_count gauge",
            ]
        )
        for phase, count in sorted(phase_counts.items()):
            lines.append(
                f'openclaw_worker_run_phase_count{{phase="{phase}"}} {int(count or 0)}'
            )

    try:
        pressure = worker_pressure_stats()
    except Exception as exc:
        lines.append(f'# worker_pressure_stats_error="{exc}"')
    else:
        lines.extend(
            [
                "# HELP openclaw_worker_pressure Worker pressure gauges.",
                "# TYPE openclaw_worker_pressure gauge",
            ]
        )
        for name in ("tasks", "procs", "zombies", "shim", "runc", "docker_cli_procs"):
            value = pressure.get(name)
            if isinstance(value, (int, float)):
                lines.append(f'openclaw_worker_pressure{{name="{name}"}} {value}')
        pids_pct = pressure.get("pids_pct")
        if isinstance(pids_pct, (int, float)):
            lines.append(f'openclaw_worker_pressure{{name="pids_pct"}} {pids_pct}')
        docker_cli_ok = 1 if pressure.get("docker_cli_ok") else 0
        lines.append(f'openclaw_worker_pressure{{name="docker_cli_ok"}} {docker_cli_ok}')

    return Response("\n".join(lines) + "\n", media_type="text/plain")


@app.post("/probe/rollout")
async def probe_rollout(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {
                "ok": False,
                "code": "POOL_NOT_INITIALIZED",
                "error": "Pool is not initialized",
            },
            status_code=503,
        )

    data = await json_payload(request)
    task_meta = data.get("task_meta")
    if not isinstance(task_meta, dict):
        return JSONResponse(
            {"ok": False, "error": "task_meta dict is required"},
            status_code=400,
        )

    task_key = str(
        data.get("task_key") or f"probe:{task_meta.get('task_name', 'unknown')}"
    )
    run_ctx_payload = (
        data.get("run_ctx") if isinstance(data.get("run_ctx"), dict) else {}
    )
    task_timeouts = data.get("task_timeouts")
    tool_call = data.get("tool_call")
    request_id = str(data.get("request_id") or f"probe-{uuid.uuid4().hex[:16]}")
    lease_id: str | None = None
    started_ts = time.time()
    exec_result: dict[str, Any] | None = None

    try:
        pool_status = await POOL.status()
        assert_worker_has_capacity_for_docker(
            phase="allocate",
            pending_closes=int(pool_status.get("pending_closes", 0)),
            pool_status=pool_status,
        )
        allocated = await POOL.allocate(task_key=task_key, request_id=request_id)
        lease_id = str(allocated["lease_id"])
        await POOL.reset(
            run_lease_id=lease_id,
            task_meta=task_meta,
            run_ctx_payload=run_ctx_payload,
            task_timeouts=task_timeouts if isinstance(task_timeouts, dict) else None,
        )
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments")
            if isinstance(tool_name, str) and tool_name:
                observation = await POOL.exec_tool(
                    lease_id,
                    tool_name,
                    arguments=arguments if isinstance(arguments, dict) else None,
                )
                exec_result = {
                    "tool_name": tool_name,
                    "observation_len": len(observation),
                }
        return JSONResponse(
            {
                "ok": True,
                "lease_id": lease_id,
                "duration_sec": round(time.time() - started_ts, 3),
                "exec": exec_result,
            }
        )
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "code": exc.code,
                "error": exc.message,
                "details": exc.details,
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
        )
    except ResetAdmissionBacklogError as exc:
        logger.warning(
            "Rollout probe deferred by reset admission backlog lease_id=%s "
            "task_key=%s: %s",
            lease_id,
            task_key,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "code": "WORKER_RESET_ADMISSION_BACKLOG",
                "error": str(exc),
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_RESET_BACKLOG_RETRY_AFTER", "10")},
        )
    except DockerImagePreparationBacklogError as exc:
        logger.warning(
            "Rollout probe deferred by Docker image preparation backlog lease_id=%s "
            "task_key=%s: %s",
            lease_id,
            task_key,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "code": "DOCKER_IMAGE_PREP_BACKLOG",
                "error": str(exc),
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
            headers={
                "Retry-After": os.getenv("WORKER_DOCKER_BUILD_BACKLOG_RETRY_AFTER", "15")
            },
        )
    except TaskImageBlacklistedError as exc:
        logger.warning(
            "Rollout probe blocked by task image blacklist lease_id=%s task_key=%s: %s",
            lease_id,
            task_key,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "code": "TASK_IMAGE_BLACKLISTED",
                "error": str(exc),
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_TASK_IMAGE_RETRY_AFTER", "300")},
        )
    except DockerImageBuildError as exc:
        logger.warning(
            "Rollout probe failed on Docker image build lease_id=%s task_key=%s: %s",
            lease_id,
            task_key,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "code": "TASK_IMAGE_BUILD_FAILED",
                "error": str(exc),
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_TASK_IMAGE_RETRY_AFTER", "300")},
        )
    except TimeoutError as exc:
        message = str(exc)
        code = "WORKER_RESET_TIMEOUT"
        if "WORKER_RESET_CANCELLED" in message:
            code = "WORKER_RESET_CANCELLED"
        logger.warning(
            "Rollout probe ended with transient worker timeout lease_id=%s "
            "task_key=%s: %s",
            lease_id,
            task_key,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "code": code,
                "error": message,
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_RESET_TIMEOUT_RETRY_AFTER", "15")},
        )
    except Exception as exc:
        logger.exception(
            "Rollout probe failed lease_id=%s task_key=%s", lease_id, task_key
        )
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "duration_sec": round(time.time() - started_ts, 3),
            },
            status_code=500,
        )
    finally:
        if lease_id:
            try:
                await POOL.close_run(lease_id, reason="rollout_probe_close")
            except Exception:
                logger.exception("Failed to close rollout probe lease_id=%s", lease_id)


@app.post("/repair/pending_closes")
async def repair_pending_closes(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    if os.getenv("WORKER_REPAIR_PENDING_CLOSES", "1") != "1":
        return JSONResponse(
            {
                "ok": False,
                "error": "Pending-close repair endpoint is disabled",
                "code": "REPAIR_DISABLED",
            },
            status_code=403,
        )

    data = await json_payload(request)
    reason = str(data.get("reason") or "manual")
    max_active_runs = _env_int("WORKER_REPAIR_PENDING_CLOSES_MAX_ACTIVE_RUNS", 0)
    cancel_timeout = _env_float("WORKER_REPAIR_PENDING_CLOSES_CANCEL_TIMEOUT", 5.0)
    min_age = _env_float(
        "WORKER_REPAIR_PENDING_CLOSES_MIN_AGE",
        max(0.0, POOL.close_task_timeout + 5.0),
    )
    try:
        if "max_active_runs" in data:
            max_active_runs = int(data["max_active_runs"])
    except (TypeError, ValueError):
        pass
    try:
        if "cancel_timeout" in data:
            cancel_timeout = float(data["cancel_timeout"])
    except (TypeError, ValueError):
        pass
    try:
        if "min_age" in data:
            min_age = float(data["min_age"])
    except (TypeError, ValueError):
        pass

    result = await POOL.repair_pending_closes(
        reason=reason,
        max_active_runs=max_active_runs,
        cancel_timeout=cancel_timeout,
        min_age=min_age,
    )
    return JSONResponse({"ok": True, **result})


@app.post("/repair/stale_runs")
async def repair_stale_runs(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    if os.getenv("WORKER_REPAIR_STALE_RUNS", "1") != "1":
        return JSONResponse(
            {
                "ok": False,
                "error": "Stale-run repair endpoint is disabled",
                "code": "REPAIR_DISABLED",
            },
            status_code=403,
        )

    data = await json_payload(request)
    reason = str(data.get("reason") or "manual")
    min_age = _env_float("WORKER_REPAIR_STALE_RUNS_MIN_AGE", 0.0)
    max_repairs = _env_int("WORKER_REPAIR_STALE_RUNS_MAX_REPAIRS", 20)
    try:
        if "min_age" in data:
            min_age = float(data["min_age"])
    except (TypeError, ValueError):
        pass
    try:
        if "max_repairs" in data:
            max_repairs = int(data["max_repairs"])
    except (TypeError, ValueError):
        pass
    wait_for_cleanup = str(data.get("wait_for_cleanup", "1")).lower() in {
        "1",
        "true",
        "yes",
    }

    result = await POOL.repair_stale_runs(
        reason=reason,
        min_age=max(0.0, min_age),
        max_repairs=max(0, max_repairs),
        wait_for_cleanup=wait_for_cleanup,
    )
    return JSONResponse({"ok": True, **result})


@app.post("/repair/close_requested_runs")
async def repair_close_requested_runs(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    if os.getenv("WORKER_REPAIR_CLOSE_REQUESTED_RUNS", "1") != "1":
        return JSONResponse(
            {
                "ok": False,
                "error": "Close-requested run repair endpoint is disabled",
                "code": "REPAIR_DISABLED",
            },
            status_code=403,
        )

    data = await json_payload(request)
    reason = str(data.get("reason") or "manual")
    min_age = _env_float("WORKER_REPAIR_CLOSE_REQUESTED_MIN_AGE", 0.0)
    max_repairs = _env_int("WORKER_REPAIR_CLOSE_REQUESTED_MAX_REPAIRS", 20)
    try:
        if "min_age" in data:
            min_age = float(data["min_age"])
    except (TypeError, ValueError):
        pass
    try:
        if "max_repairs" in data:
            max_repairs = int(data["max_repairs"])
    except (TypeError, ValueError):
        pass
    wait_for_cleanup = str(data.get("wait_for_cleanup", "0")).lower() in {
        "1",
        "true",
        "yes",
    }

    result = await POOL.repair_close_requested_runs(
        reason=reason,
        min_age=max(0.0, min_age),
        max_repairs=max(0, max_repairs),
        wait_for_cleanup=wait_for_cleanup,
    )
    return JSONResponse({"ok": True, **result})


@app.post("/repair/resetting_runs")
async def repair_resetting_runs(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )
    if os.getenv("WORKER_REPAIR_RESETTING_RUNS", "1") != "1":
        return JSONResponse(
            {
                "ok": False,
                "error": "Resetting-run repair endpoint is disabled",
                "code": "REPAIR_DISABLED",
            },
            status_code=403,
        )

    data = await json_payload(request)
    reason = str(data.get("reason") or "manual")
    min_age = _env_float("WORKER_REPAIR_RESETTING_MIN_AGE", 2100.0)
    max_repairs = _env_int("WORKER_REPAIR_RESETTING_MAX_REPAIRS", 64)
    try:
        if "min_age" in data:
            min_age = float(data["min_age"])
    except (TypeError, ValueError):
        pass
    try:
        if "max_repairs" in data:
            max_repairs = int(data["max_repairs"])
    except (TypeError, ValueError):
        pass
    wait_for_cleanup = str(data.get("wait_for_cleanup", "0")).lower() in {
        "1",
        "true",
        "yes",
    }

    result = await POOL.repair_resetting_runs(
        reason=reason,
        min_age=max(0.0, min_age),
        max_repairs=max(0, max_repairs),
        wait_for_cleanup=wait_for_cleanup,
    )
    return JSONResponse({"ok": True, **result})


@app.post("/allocate")
async def allocate(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    task_key = data.get("task_key", "")
    request_id = data.get("request_id")

    if not task_key:
        return JSONResponse(
            {"ok": False, "error": "task_key is required"}, status_code=400
        )

    auto_repair_result: dict[str, Any] | None = None

    async def _try_allocate_once() -> dict[str, Any]:
        pool_status = await POOL.status()
        assert_worker_has_capacity_for_docker(
            phase="allocate",
            pending_closes=int(pool_status.get("pending_closes", 0)),
            pool_status=pool_status,
        )
        return await POOL.allocate(task_key=str(task_key), request_id=request_id)

    try:
        result = await _try_allocate_once()
        return JSONResponse({"ok": True, **result})
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": exc.message,
                "code": exc.code,
                "details": exc.details,
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_PRESSURE_RETRY_AFTER", "10")},
        )
    except CapacityError as exc:
        if os.getenv("WORKER_AUTO_REPAIR_ON_CAPACITY", "1") == "1":
            max_repairs = _env_int("WORKER_AUTO_REPAIR_MAX_REPAIRS", 20)
            close_min_age = _env_float(
                "WORKER_AUTO_REPAIR_CLOSE_REQUESTED_MIN_AGE", 0.0
            )
            stale_min_age = _env_float("WORKER_AUTO_REPAIR_STALE_MIN_AGE", 0.0)
            close_repair = await POOL.repair_close_requested_runs(
                reason=f"allocate_capacity:{exc.code}",
                min_age=max(0.0, close_min_age),
                max_repairs=max(0, max_repairs),
                wait_for_cleanup=False,
            )
            stale_repair: dict[str, Any] | None = None
            if not close_repair.get("repaired"):
                stale_repair = await POOL.repair_stale_runs(
                    reason=f"allocate_capacity:{exc.code}",
                    min_age=max(0.0, stale_min_age),
                    max_repairs=max(0, max_repairs),
                    wait_for_cleanup=False,
                )
            auto_repair_result = {
                "close_requested": close_repair,
                "stale": stale_repair,
            }
            if close_repair.get("repaired") or (
                stale_repair is not None and stale_repair.get("repaired")
            ):
                try:
                    result = await _try_allocate_once()
                    return JSONResponse(
                        {"ok": True, **result, "auto_repair": auto_repair_result}
                    )
                except CapacityError as retry_exc:
                    exc = retry_exc
        return JSONResponse(
            {
                "ok": False,
                "error": exc.message,
                "code": exc.code,
                "auto_repair": auto_repair_result,
            },
            status_code=429,
            headers={"Retry-After": os.getenv("WORKER_CAPACITY_RETRY_AFTER", "5")},
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/heartbeat")
async def heartbeat(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        await POOL.heartbeat(str(lease_id))
        return JSONResponse({"ok": True})
    except KeyError as exc:
        # FIX-3: Return HTTP 410 Gone for expired lease_id to prevent retry cascades
        logger.warning(
            "Heartbeat lease_id=%s no longer exists (likely timeout cleanup): %s",
            lease_id,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "error": "Lease expired or already cleaned up",
                "code": "LEASE_EXPIRED",
            },
            status_code=410,
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/reset")
async def reset(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    task_meta = data.get("task_meta")
    run_ctx_payload = data.get("run_ctx")
    task_timeouts = data.get("task_timeouts")
    request_id = data.get("request_id")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )
    if not isinstance(task_meta, dict):
        return JSONResponse(
            {"ok": False, "error": "task_meta dict is required"}, status_code=400
        )

    try:
        pool_status = await POOL.status()
        assert_worker_has_capacity_for_docker(
            phase="reset",
            pending_closes=int(pool_status.get("pending_closes", 0)),
            pool_status=pool_status,
        )
        out = await POOL.reset(
            run_lease_id=str(lease_id),
            task_meta=task_meta,
            run_ctx_payload=run_ctx_payload,
            task_timeouts=task_timeouts,
            request_id=str(request_id) if request_id else None,
        )
        return JSONResponse({"ok": True, **out})
    except ResourcePressureError as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": exc.message,
                "code": exc.code,
                "details": exc.details,
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_PRESSURE_RETRY_AFTER", "10")},
        )
    except ResetAdmissionBacklogError as exc:
        logger.warning("Reset deferred by reset admission backlog lease_id=%s: %s", lease_id, exc)
        try:
            await POOL.close_run(str(lease_id), reason="reset_admission_backlog")
        except Exception:
            logger.exception("Failed to schedule cleanup after reset backlog for %s", lease_id)
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "code": "WORKER_RESET_ADMISSION_BACKLOG",
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_RESET_BACKLOG_RETRY_AFTER", "10")},
        )
    except DockerImagePreparationBacklogError as exc:
        logger.warning(
            "Reset deferred by Docker image preparation backlog lease_id=%s: %s",
            lease_id,
            exc,
        )
        try:
            await POOL.close_run(str(lease_id), reason="docker_image_prep_backlog")
        except Exception:
            logger.exception("Failed to schedule cleanup after image prep backlog for %s", lease_id)
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "code": "DOCKER_IMAGE_PREP_BACKLOG",
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
            },
            status_code=503,
            headers={
                "Retry-After": os.getenv("WORKER_DOCKER_BUILD_BACKLOG_RETRY_AFTER", "15")
            },
        )
    except TaskImageBlacklistedError as exc:
        logger.warning("Reset blocked by task image blacklist lease_id=%s: %s", lease_id, exc)
        try:
            await POOL.close_run(str(lease_id), reason="task_image_blacklisted")
        except Exception:
            logger.exception("Failed to schedule cleanup after image blacklist for %s", lease_id)
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "code": "TASK_IMAGE_BLACKLISTED",
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_TASK_IMAGE_RETRY_AFTER", "300")},
        )
    except DockerImageBuildError as exc:
        logger.warning("Reset failed on Docker image build lease_id=%s: %s", lease_id, exc)
        try:
            await POOL.close_run(str(lease_id), reason="task_image_build_failed")
        except Exception:
            logger.exception("Failed to schedule cleanup after image build failure for %s", lease_id)
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "code": "TASK_IMAGE_BUILD_FAILED",
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_TASK_IMAGE_RETRY_AFTER", "300")},
        )
    except ResetInProgressError as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "code": "RESET_IN_PROGRESS",
                "lease_id": exc.run_lease_id,
                "request_id": exc.request_id,
            },
            status_code=429,
            headers={"Retry-After": os.getenv("WORKER_RESET_IN_PROGRESS_RETRY_AFTER", "2")},
        )
    except KeyError as exc:
        # FIX-3: Return HTTP 410 Gone for expired lease_id to prevent retry cascades
        logger.warning(
            "Reset lease_id=%s no longer exists (likely timeout cleanup): %s",
            lease_id,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "error": "Lease expired or already cleaned up",
                "code": "LEASE_EXPIRED",
            },
            status_code=410,
        )
    except TimeoutError as exc:
        message = str(exc)
        code = "WORKER_RESET_TIMEOUT"
        if "WORKER_RESET_CANCELLED" in message:
            code = "WORKER_RESET_CANCELLED"
        logger.warning("Reset ended with transient worker timeout lease_id=%s: %s", lease_id, exc)
        try:
            await POOL.close_run(str(lease_id), reason=code.lower())
        except Exception:
            logger.exception("Failed to schedule cleanup after reset timeout for %s", lease_id)
        return JSONResponse(
            {
                "ok": False,
                "error": message,
                "code": code,
                "task_name": task_meta.get("task_name"),
                "task_path": task_meta.get("task_path"),
            },
            status_code=503,
            headers={"Retry-After": os.getenv("WORKER_RESET_TIMEOUT_RETRY_AFTER", "15")},
        )
    except Exception as exc:
        logger.exception("Reset failed for lease_id=%s", lease_id)
        try:
            await POOL.close_run(str(lease_id), reason="reset_failure")
        except Exception:
            logger.exception("Failed to schedule cleanup after reset failure for %s", lease_id)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/exec_tool")
async def exec_tool(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    tool_call = data.get("tool_call")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )
    if not isinstance(tool_call, dict):
        return JSONResponse(
            {"ok": False, "error": "tool_call dict is required"}, status_code=400
        )

    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments")

    if not isinstance(tool_name, str) or not tool_name:
        return JSONResponse(
            {"ok": False, "error": "tool_call.name is required"}, status_code=400
        )
    if arguments is not None and not isinstance(arguments, dict):
        return JSONResponse(
            {"ok": False, "error": "tool_call.arguments must be a dict"},
            status_code=400,
        )

    try:
        observation = await POOL.exec_tool(
            str(lease_id), tool_name, arguments=arguments
        )
        return JSONResponse({"ok": True, "observation": observation})
    except KeyError as exc:
        # FIX-3: Return HTTP 410 Gone for expired lease_id to prevent retry cascades
        logger.warning(
            "Exec_tool lease_id=%s no longer exists (likely timeout cleanup): %s",
            lease_id,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "error": "Lease expired or already cleaned up",
                "code": "LEASE_EXPIRED",
            },
            status_code=410,
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/evaluate")
async def evaluate(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    trajectory = data.get("trajectory")

    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        score, details = await POOL.evaluate(
            str(lease_id), trajectory if isinstance(trajectory, dict) else None
        )
        payload: dict[str, Any] = {"ok": True, "score": score}
        if details is not None:
            payload["details"] = details
        return JSONResponse(payload)
    except KeyError as exc:
        # FIX-3: Return HTTP 410 Gone for expired lease_id to prevent retry cascades
        logger.warning(
            "Evaluate lease_id=%s no longer exists (likely timeout cleanup): %s",
            lease_id,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "error": "Lease expired or already cleaned up",
                "code": "LEASE_EXPIRED",
            },
            status_code=410,
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/close")
async def close(request: Request) -> JSONResponse:
    if POOL is None:
        return JSONResponse(
            {"ok": False, "error": "Pool is not initialized"}, status_code=500
        )

    data = await json_payload(request)
    lease_id = data.get("lease_id")
    if not lease_id:
        return JSONResponse(
            {"ok": False, "error": "lease_id is required"}, status_code=400
        )

    try:
        found = await POOL.close_run(str(lease_id), reason="http_close")
        return JSONResponse({"ok": True, "found": found})
    except KeyError as exc:
        # FIX-3: Return HTTP 410 Gone for expired lease_id to prevent retry cascades
        logger.warning(
            "Close lease_id=%s no longer exists (likely already cleaned up): %s",
            lease_id,
            exc,
        )
        return JSONResponse(
            {
                "ok": False,
                "error": "Lease expired or already cleaned up",
                "code": "LEASE_EXPIRED",
            },
            status_code=410,
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


_REAPER_TASK: asyncio.Task | None = None


@app.on_event("startup")
async def _on_startup() -> None:
    global _REAPER_TASK
    if POOL is not None:
        _REAPER_TASK = asyncio.create_task(POOL.periodic_reap(interval=60.0))


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    global POOL, _REAPER_TASK
    if _REAPER_TASK is not None:
        _REAPER_TASK.cancel()
        _REAPER_TASK = None
    if POOL is not None:
        await POOL.shutdown()
        POOL = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C-layer: terminal env worker server")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("ENV_SERVER_PORT", "18081"))
    )

    parser.add_argument(
        "--max-tasks", type=int, default=int(os.getenv("WORKER_MAX_TASKS", "16"))
    )
    parser.add_argument(
        "--max-runs-per-task",
        type=int,
        default=int(os.getenv("WORKER_MAX_RUNS_PER_TASK", "8")),
    )
    parser.add_argument(
        "--run-idle-ttl",
        type=int,
        default=int(os.getenv("WORKER_RUN_IDLE_TTL", "600")),
        help="Seconds before an idle RunSlot is reaped",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=os.getenv("TBENCH_OUTPUT_ROOT", "build_outputs"),
    )

    parser.add_argument(
        "--ensure-image-timeout",
        type=float,
        default=float(os.getenv("ENSURE_IMAGE_TIMEOUT", "1200.0")),
    )
    parser.add_argument(
        "--reset-session-timeout",
        type=float,
        default=float(os.getenv("RESET_SESSION_TIMEOUT", "600.0")),
    )
    parser.add_argument(
        "--close-session-timeout",
        type=float,
        default=float(os.getenv("CLOSE_SESSION_TIMEOUT", "60.0")),
    )
    parser.add_argument(
        "--eval-timeout", type=float, default=float(os.getenv("EVAL_TIMEOUT", "600.0"))
    )
    parser.add_argument(
        "--max-concurrent-closes",
        type=int,
        default=int(os.getenv("WORKER_MAX_CONCURRENT_CLOSES", "10")),
        help="Max concurrent Docker stop operations",
    )

    return parser.parse_args()


def main() -> None:
    global POOL
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s"
    )

    POOL = WorkerPool(
        max_tasks=args.max_tasks,
        max_runs_per_task=args.max_runs_per_task,
        run_idle_ttl=args.run_idle_ttl,
        output_root=args.output_root,
        default_timeouts=TaskTimeouts(
            ensure_image=float(args.ensure_image_timeout),
            reset_session=float(args.reset_session_timeout),
            close_session=float(args.close_session_timeout),
            eval=float(args.eval_timeout),
        ),
        max_concurrent_closes=args.max_concurrent_closes,
    )

    logger.info(
        "Starting worker server on %s:%s  max_tasks=%s  max_runs_per_task=%s",
        args.host,
        args.port,
        args.max_tasks,
        args.max_runs_per_task,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
