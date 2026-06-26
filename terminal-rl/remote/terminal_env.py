from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from camel.toolkits import FunctionTool, TerminalToolkit

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import Terminal

from ..custom_types import RunContext, TaskSpec, TaskTimeouts

from .agentharm_env import AgentHarmEnv
from .agent_safetybench_env import AgentSafetyBenchEnv
from .docker_compose_utils import compose_up_no_build, prepare_task_docker_image

logger = logging.getLogger("terminal.env.worker.terminal_env")
logger.setLevel(logging.INFO)

_TASK_CONTAINER_RE = re.compile(r"^[0-9]+-[A-Za-z0-9]{8}-slime-run$")
_TASK_ID_PREFIX_RE = re.compile(r"^([0-9]+)(?:[-_.:]|$)")
_FIXED_TASK_SERVICE_RE = re.compile(r"^tb__([0-9]+)__.*")
_DOCKER_CLEANUP_EXECUTOR: ThreadPoolExecutor | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


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
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def _docker_cleanup_executor() -> ThreadPoolExecutor:
    global _DOCKER_CLEANUP_EXECUTOR
    if _DOCKER_CLEANUP_EXECUTOR is None:
        workers = max(1, _env_int("TERMINAL_ENV_DOCKER_CLEANUP_WORKERS", 8))
        _DOCKER_CLEANUP_EXECUTOR = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="openclaw-docker-cleanup",
        )
    return _DOCKER_CLEANUP_EXECUTOR


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


def _matches_project_name(name: str, project_names: set[str], *, broad: bool) -> bool:
    if not name:
        return False
    for project in project_names:
        if name == project:
            return True
        if broad and (
            name.startswith(f"{project}-")
            or name.startswith(f"{project}_")
            or name.startswith(project)
        ):
            return True
    return False


def _docker_image_prefixes(*values: str | None) -> set[str]:
    prefixes: set[str] = set()
    for value in values:
        if not value:
            continue
        raw = value.strip()
        if raw:
            prefixes.add(raw)
        task_match = re.match(r"^([0-9]+)[-_.]", raw)
        if task_match:
            prefixes.add(f"tb__{task_match.group(1)}__")
    return {prefix for prefix in prefixes if prefix.startswith("tb__")}


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


def _fixed_task_service_id(name: str, image: str = "") -> str | None:
    for value in (name, image):
        match = _FIXED_TASK_SERVICE_RE.match(value or "")
        if match:
            return match.group(1)
    return None


def _compose_project_candidates(
    trial_name: str | None, client_container_name: str | None
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for value in (client_container_name, trial_name):
        variants = sorted(_docker_name_variants(value))
        raw = (value or "").strip()
        if raw:
            variants.insert(0, raw)
        for variant in variants:
            if variant and variant not in seen:
                candidates.append(variant)
                seen.add(variant)
    return candidates[:6]


def _docker_status_age_seconds(status: str) -> float | None:
    text = (status or "").strip().lower()
    if not text:
        return None
    if "less than a second" in text:
        return 0.0
    if "about a minute" in text or "a minute" in text:
        return 60.0
    match = re.search(
        r"(\d+)\s+"
        r"(second|seconds|minute|minutes|hour|hours|day|days|week|weeks|month|months)",
        text,
    )
    if match is None:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("second"):
        return float(value)
    if unit.startswith("minute"):
        return float(value * 60)
    if unit.startswith("hour"):
        return float(value * 3600)
    if unit.startswith("day"):
        return float(value * 86400)
    if unit.startswith("week"):
        return float(value * 7 * 86400)
    if unit.startswith("month"):
        return float(value * 30 * 86400)
    return None


def _is_task_container(name: str, image: str) -> bool:
    if _TASK_CONTAINER_RE.match(name or ""):
        return True
    return bool((name or "").endswith("-slime-run") and (image or "").startswith("tb__"))


def _clean_docker_label(value: str | None) -> str:
    raw = (value or "").strip()
    return "" if raw == "<no value>" else raw


def _docker_compose_down_projects(
    *,
    docker_compose_path: str | None,
    trial_name: str,
    client_container_name: str | None,
    reason: str,
    command_timeout: float,
) -> None:
    if not _env_bool("TERMINAL_ENV_COMPOSE_DOWN_CLEANUP", True):
        return
    if not docker_compose_path:
        return
    compose_path = Path(docker_compose_path)
    if not compose_path.exists():
        logger.warning(
            "Skipping docker compose down for TerminalEnv %s (%s): compose file missing: %s",
            trial_name,
            reason,
            compose_path,
        )
        return

    service_timeout = str(max(1, _env_int("TERMINAL_ENV_COMPOSE_DOWN_SERVICE_TIMEOUT", 5)))
    for project in _compose_project_candidates(trial_name, client_container_name):
        cmd = [
            "docker",
            "compose",
            "-p",
            project,
            "-f",
            str(compose_path),
            "down",
            "--remove-orphans",
            "-v",
            "--timeout",
            service_timeout,
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=command_timeout,
            )
            logger.warning(
                "Docker compose down finished for TerminalEnv %s project=%s "
                "reason=%s rc=%s stdout=%s stderr=%s",
                trial_name,
                project,
                reason,
                completed.returncode,
                completed.stdout.strip()[:300],
                completed.stderr.strip()[:300],
            )
        except Exception as exc:
            logger.warning(
                "Docker compose down failed for TerminalEnv %s project=%s reason=%s: %s",
                trial_name,
                project,
                reason,
                exc,
            )


def _remove_fixed_task_services_without_running_clients(
    *,
    task_ids: set[str],
    reason: str,
    timeout: float,
    max_remove: int = 64,
) -> int:
    if not task_ids:
        return 0

    try:
        listed = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning(
            "Could not list Docker containers for fixed task service cleanup (%s): %s",
            reason,
            exc,
        )
        return 0

    running_client_task_ids: set[str] = set()
    rows: list[tuple[str, str, str, str]] = []
    for line in listed.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        container_id, name = parts[0], parts[1]
        image = parts[2] if len(parts) > 2 else ""
        status = parts[3] if len(parts) > 3 else ""
        rows.append((container_id, name, image, status))
        if status.lower().startswith("up") and _is_task_container(name, image):
            task_id = _task_id_from_ref(name)
            if task_id:
                running_client_task_ids.add(task_id)

    blocked = task_ids.intersection(running_client_task_ids)
    removable_task_ids = task_ids.difference(blocked)
    if blocked:
        logger.warning(
            "Skipping fixed task service cleanup for active task id(s) %s reason=%s",
            ",".join(sorted(blocked)),
            reason,
        )
    if not removable_task_ids:
        return 0

    candidates: list[tuple[str, str, str, str, str]] = []
    for container_id, name, image, status in rows:
        task_id = _fixed_task_service_id(name, image)
        if task_id and task_id in removable_task_ids:
            candidates.append((container_id, name, image, status, task_id))
            if max_remove > 0 and len(candidates) >= max_remove:
                break

    if not candidates:
        return 0

    logger.warning(
        "Removing %d fixed task service container(s) without running clients "
        "reason=%s task_ids=%s samples=%s",
        len(candidates),
        reason,
        ",".join(sorted(removable_task_ids)),
        "; ".join(
            f"{cid[:12]} name={name} image={image} status={status}"
            for cid, name, image, status, _task_id in candidates[:8]
        ),
    )

    removed_count = 0
    for start in range(0, len(candidates), 20):
        chunk = candidates[start : start + 20]
        ids = [item[0] for item in chunk]
        try:
            removed = subprocess.run(
                ["docker", "rm", "-f", *ids],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if removed.returncode == 0:
                removed_count += len(ids)
            logger.warning(
                "Fixed task service docker rm finished ids=%s rc=%s stdout=%s stderr=%s",
                ",".join(cid[:12] for cid in ids),
                removed.returncode,
                removed.stdout.strip()[:300],
                removed.stderr.strip()[:300],
            )
        except Exception as exc:
            logger.warning(
                "Fixed task service docker rm failed ids=%s: %s",
                ",".join(cid[:12] for cid in ids),
                exc,
            )
    return removed_count


def _remove_inactive_compose_resources(
    *,
    resource_kind: str,
    active_project_names: set[str],
    active_task_ids: set[str],
    reason: str,
    timeout: float,
    max_remove: int,
) -> int:
    if resource_kind == "network":
        list_cmd = [
            "docker",
            "network",
            "ls",
            "--format",
            "{{.ID}}\t{{.Name}}\t{{.Label \"com.docker.compose.project\"}}",
        ]
        rm_cmd_prefix = ["docker", "network", "rm"]
        use_id = True
    elif resource_kind == "volume":
        list_cmd = [
            "docker",
            "volume",
            "ls",
            "--format",
            "{{.Name}}\t{{.Label \"com.docker.compose.project\"}}",
        ]
        rm_cmd_prefix = ["docker", "volume", "rm"]
        use_id = False
    else:
        return 0

    try:
        listed = subprocess.run(
            list_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("Could not list Docker %ss for orphan cleanup (%s): %s", resource_kind, reason, exc)
        return 0

    candidates: list[tuple[str, str, str]] = []
    for line in listed.stdout.splitlines():
        parts = line.split("\t")
        if resource_kind == "network":
            if len(parts) < 2:
                continue
            resource_id, name = parts[0], parts[1]
            compose_project = _clean_docker_label(parts[2] if len(parts) > 2 else "")
            ref = resource_id if use_id else name
        else:
            if not parts:
                continue
            name = parts[0]
            compose_project = _clean_docker_label(parts[1] if len(parts) > 1 else "")
            ref = name
        if compose_project and _matches_project_name(
            compose_project, active_project_names, broad=True
        ):
            continue
        task_id = _task_id_from_ref(compose_project) or _task_id_from_ref(name)
        looks_like_task_resource = (
            "slime-run" in name
            or "slime-run" in compose_project
            or (task_id is not None and task_id not in active_task_ids)
        )
        if not looks_like_task_resource:
            continue
        if task_id is not None and task_id in active_task_ids:
            continue
        if not compose_project and "slime-run" not in name:
            continue
        candidates.append((ref, name, compose_project))
        if max_remove > 0 and len(candidates) >= max_remove:
            break

    if not candidates:
        return 0

    removed_count = 0
    logger.warning(
        "Orphan Docker sweep removing %d stale compose %s(s) reason=%s samples=%s",
        len(candidates),
        resource_kind,
        reason,
        "; ".join(
            f"name={name} project={project}" for _ref, name, project in candidates[:8]
        ),
    )
    for ref, name, _project in candidates:
        try:
            removed = subprocess.run(
                [*rm_cmd_prefix, ref],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if removed.returncode == 0:
                removed_count += 1
            logger.warning(
                "Orphan Docker sweep %s rm finished name=%s rc=%s stdout=%s stderr=%s",
                resource_kind,
                name,
                removed.returncode,
                removed.stdout.strip()[:300],
                removed.stderr.strip()[:300],
            )
        except Exception as exc:
            logger.warning(
                "Orphan Docker sweep %s rm failed name=%s: %s",
                resource_kind,
                name,
                exc,
            )
    return removed_count


def _force_remove_docker_objects(
    *,
    trial_name: str,
    client_container_name: str | None,
    docker_image_name_prefix: str | None = None,
    docker_compose_path: str | None = None,
    reason: str,
) -> None:
    if not _env_bool("TERMINAL_ENV_FORCE_DOCKER_CLEANUP", True):
        return

    timeout = float(os.getenv("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT", "20"))
    broad = _env_bool("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_BROAD", True)
    project_names = _docker_name_variants(trial_name)
    project_names.update(_docker_name_variants(client_container_name))
    image_prefixes = _docker_image_prefixes(
        docker_image_name_prefix,
        trial_name,
        client_container_name,
    )
    task_ids = {
        task_id
        for task_id in (
            _task_id_from_ref(docker_image_name_prefix),
            _task_id_from_ref(trial_name),
            _task_id_from_ref(client_container_name),
        )
        if task_id
    }
    for prefix in image_prefixes:
        task_id = _task_id_from_ref(prefix)
        if task_id:
            task_ids.add(task_id)
    if not project_names and not image_prefixes:
        return

    def _run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    _docker_compose_down_projects(
        docker_compose_path=docker_compose_path,
        trial_name=trial_name,
        client_container_name=client_container_name,
        reason=reason,
        command_timeout=timeout,
    )

    direct_removed = False
    direct_name = (client_container_name or "").strip()
    if direct_name:
        try:
            removed = _run(["docker", "rm", "-f", direct_name])
            if removed.returncode == 0:
                direct_removed = True
                logger.warning(
                    "Force docker rm exact finished for TerminalEnv %s name=%s rc=%s stdout=%s stderr=%s",
                    trial_name,
                    direct_name,
                    removed.returncode,
                    removed.stdout.strip()[:300],
                    removed.stderr.strip()[:300],
                )
            else:
                logger.warning(
                    "Force docker rm exact did not remove container for TerminalEnv %s name=%s rc=%s stdout=%s stderr=%s",
                    trial_name,
                    direct_name,
                    removed.returncode,
                    removed.stdout.strip()[:300],
                    removed.stderr.strip()[:300],
                )
        except Exception as exc:
            logger.warning(
                "Force docker rm exact failed for TerminalEnv %s name=%s: %s",
                trial_name,
                direct_name,
                exc,
            )

    try:
        listed = _run(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}",
            ]
        )
    except Exception as exc:
        logger.warning(
            "Force cleanup could not list Docker containers for %s (%s): %s",
            trial_name,
            reason,
            exc,
        )
        return

    container_ids: list[str] = []
    container_samples: list[str] = []
    for line in listed.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        container_id, name = parts[0], parts[1]
        image = parts[2] if len(parts) > 2 else ""
        status = parts[3] if len(parts) > 3 else ""
        name_match = _matches_project_name(name, project_names, broad=broad)
        # Image-prefix matching is intentionally only a fallback when we do not
        # know a project/container name; matching tb__<task>__ can otherwise
        # remove other active samples of the same task.
        image_match = not project_names and any(
            image.startswith(prefix) for prefix in image_prefixes
        )
        if name_match or image_match:
            container_ids.append(container_id)
            if len(container_samples) < 8:
                container_samples.append(
                    f"{container_id[:12]} name={name} image={image} status={status}"
                )

    if container_ids:
        logger.warning(
            "Force removing %d Docker container(s) for TerminalEnv %s (%s): %s",
            len(container_ids),
            trial_name,
            reason,
            "; ".join(container_samples),
        )
        for start in range(0, len(container_ids), 20):
            chunk = container_ids[start : start + 20]
            try:
                removed = _run(["docker", "rm", "-f", *chunk])
                logger.warning(
                    "Force docker rm finished for TerminalEnv %s ids=%s rc=%s stdout=%s stderr=%s",
                    trial_name,
                    ",".join(cid[:12] for cid in chunk),
                    removed.returncode,
                    removed.stdout.strip()[:300],
                    removed.stderr.strip()[:300],
                )
            except Exception as exc:
                logger.warning(
                    "Force docker rm failed for TerminalEnv %s ids=%s: %s",
                    trial_name,
                    ",".join(chunk),
                    exc,
                )
    elif direct_removed:
        logger.warning(
            "Force cleanup removed exact Docker container for TerminalEnv %s (%s) "
            "and matched no additional containers; client_container=%s "
            "image_prefixes=%s projects=%s",
            trial_name,
            reason,
            client_container_name or "",
            ",".join(sorted(image_prefixes)) or "",
            ",".join(sorted(project_names)) or "",
        )
    else:
        logger.warning(
            "Force cleanup matched no Docker containers for TerminalEnv %s (%s); "
            "client_container=%s image_prefixes=%s projects=%s",
            trial_name,
            reason,
            client_container_name or "",
            ",".join(sorted(image_prefixes)) or "",
            ",".join(sorted(project_names)) or "",
        )

    try:
        networks = _run(["docker", "network", "ls", "--format", "{{.ID}}\t{{.Name}}"])
    except Exception:
        return

    network_ids: list[str] = []
    for line in networks.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        net_id, name = parts[0], parts[1]
        if _matches_project_name(name, project_names, broad=broad):
            network_ids.append(net_id)

    for net_id in network_ids:
        try:
            removed_net = _run(["docker", "network", "rm", net_id])
            logger.warning(
                "Force docker network rm finished for TerminalEnv %s id=%s rc=%s",
                trial_name,
                net_id[:12],
                removed_net.returncode,
            )
        except Exception:
            pass

    removed_fixed = _remove_fixed_task_services_without_running_clients(
        task_ids=task_ids,
        reason=f"force_cleanup:{reason}",
        timeout=timeout,
        max_remove=_env_int("TERMINAL_ENV_FIXED_SERVICE_CLEANUP_MAX_REMOVE", 64),
    )
    if removed_fixed:
        logger.warning(
            "Force cleanup removed %d fixed task service container(s) for TerminalEnv %s "
            "(%s) task_ids=%s",
            removed_fixed,
            trial_name,
            reason,
            ",".join(sorted(task_ids)),
        )


def _attach_detached_cleanup_logger(
    fut: asyncio.Future[Any], *, trial_name: str, reason: str
) -> None:
    def _on_done(done: asyncio.Future[Any]) -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            logger.warning(
                "Detached Docker cleanup was cancelled for TerminalEnv %s (%s)",
                trial_name,
                reason,
            )
        except Exception:
            logger.exception(
                "Detached Docker cleanup failed for TerminalEnv %s (%s)",
                trial_name,
                reason,
            )
        else:
            logger.warning(
                "Detached Docker cleanup finished for TerminalEnv %s (%s)",
                trial_name,
                reason,
            )

    fut.add_done_callback(_on_done)


async def _force_remove_docker_objects_async(
    *,
    trial_name: str,
    client_container_name: str | None,
    docker_image_name_prefix: str | None = None,
    docker_compose_path: str | None = None,
    reason: str,
) -> None:
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(
        _docker_cleanup_executor(),
        partial(
            _force_remove_docker_objects,
            trial_name=trial_name,
            client_container_name=client_container_name,
            docker_image_name_prefix=docker_image_name_prefix,
            docker_compose_path=docker_compose_path,
            reason=reason,
        ),
    )
    try:
        await asyncio.shield(fut)
    except asyncio.CancelledError:
        logger.warning(
            "Docker cleanup detached after cancellation for TerminalEnv %s (%s); "
            "cleanup will continue in the executor.",
            trial_name,
            reason,
        )
        _attach_detached_cleanup_logger(fut, trial_name=trial_name, reason=reason)
        raise


def force_remove_orphan_docker_objects(
    *,
    active_container_names: set[str],
    active_project_names: set[str] | None = None,
    active_task_ids: set[str] | None = None,
    reason: str,
    min_age_sec: float,
    max_remove: int,
) -> int:
    if not _env_bool("TERMINAL_ENV_FORCE_DOCKER_CLEANUP", True):
        return 0

    timeout = float(os.getenv("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_TIMEOUT", "20"))
    try:
        listed = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Label \"com.docker.compose.project\"}}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("Orphan Docker sweep could not list containers (%s): %s", reason, exc)
        return -1
    if listed.returncode != 0:
        logger.warning(
            "Orphan Docker sweep docker ps failed (%s): rc=%s stdout=%s stderr=%s",
            reason,
            listed.returncode,
            (listed.stdout or "").strip()[:400],
            (listed.stderr or "").strip()[:400],
        )
        return -1

    active = {name for name in active_container_names if name}
    active_projects = {name for name in (active_project_names or set()) if name}
    active_tasks = {task_id for task_id in (active_task_ids or set()) if task_id}
    for name in active:
        task_id = _task_id_from_ref(name)
        if task_id:
            active_tasks.add(task_id)

    running_client_task_ids: set[str] = set()
    rows: list[tuple[str, str, str, str, str]] = []
    for line in listed.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        container_id, name = parts[0], parts[1]
        image = parts[2] if len(parts) > 2 else ""
        status = parts[3] if len(parts) > 3 else ""
        compose_project = _clean_docker_label(parts[4] if len(parts) > 4 else "")
        rows.append((container_id, name, image, status, compose_project))
        if status.lower().startswith("up") and _is_task_container(name, image):
            task_id = _task_id_from_ref(name)
            if task_id:
                running_client_task_ids.add(task_id)

    candidates: list[tuple[str, str, str, str, float, str]] = []
    for container_id, name, image, status, compose_project in rows:
        if name in active:
            continue
        if compose_project and _matches_project_name(
            compose_project, active_projects, broad=True
        ):
            continue

        fixed_task_id = _fixed_task_service_id(name, image)
        fixed_service_orphan = bool(
            fixed_task_id
            and fixed_task_id not in active_tasks
            and fixed_task_id not in running_client_task_ids
        )
        inactive_project_container = bool(
            compose_project
            and "slime-run" in compose_project
            and not _matches_project_name(compose_project, active_projects, broad=True)
            and ((image or "").startswith("tb__") or _task_id_from_ref(compose_project))
        )
        stale_client = _is_task_container(name, image)
        if not (stale_client or fixed_service_orphan or inactive_project_container):
            continue
        age_sec = _docker_status_age_seconds(status)
        if age_sec is None or age_sec < min_age_sec:
            continue
        if fixed_service_orphan:
            match_reason = f"fixed_service_task={fixed_task_id}"
        elif inactive_project_container:
            match_reason = f"inactive_project={compose_project}"
        else:
            match_reason = "stale_client"
        candidates.append((container_id, name, image, status, age_sec, match_reason))
        if max_remove > 0 and len(candidates) >= max_remove:
            break

    removed_count = 0
    if candidates:
        logger.warning(
            "Orphan Docker sweep removing %d stale task container(s) reason=%s "
            "min_age=%.1fs active=%d samples=%s",
            len(candidates),
            reason,
            min_age_sec,
            len(active),
            "; ".join(
                f"{cid[:12]} name={name} image={image} status={status} reason={why}"
                for cid, name, image, status, _age, why in candidates[:8]
            ),
        )

        for start in range(0, len(candidates), 20):
            chunk = candidates[start : start + 20]
            ids = [item[0] for item in chunk]
            try:
                removed = subprocess.run(
                    ["docker", "rm", "-f", *ids],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if removed.returncode == 0:
                    removed_count += len(ids)
                logger.warning(
                    "Orphan Docker sweep rm finished ids=%s rc=%s stdout=%s stderr=%s",
                    ",".join(cid[:12] for cid in ids),
                    removed.returncode,
                    removed.stdout.strip()[:300],
                    removed.stderr.strip()[:300],
                )
            except Exception as exc:
                logger.warning(
                    "Orphan Docker sweep rm failed ids=%s: %s",
                    ",".join(cid[:12] for cid in ids),
                    exc,
                )
    if _env_bool("WORKER_ORPHAN_DOCKER_SWEEP_RESOURCES", True):
        resource_max_remove = max(0, _env_int("WORKER_ORPHAN_DOCKER_SWEEP_RESOURCE_MAX_REMOVE", 128))
        if resource_max_remove:
            _remove_inactive_compose_resources(
                resource_kind="network",
                active_project_names=active_projects,
                active_task_ids=active_tasks,
                reason=reason,
                timeout=timeout,
                max_remove=resource_max_remove,
            )
            _remove_inactive_compose_resources(
                resource_kind="volume",
                active_project_names=active_projects,
                active_task_ids=active_tasks,
                reason=reason,
                timeout=timeout,
                max_remove=resource_max_remove,
            )
    return removed_count


def _stop_terminal_compat(terminal: Terminal, timeout: float) -> None:
    try:
        supports_timeout = "timeout" in inspect.signature(terminal.stop).parameters
    except (TypeError, ValueError):
        supports_timeout = False

    if supports_timeout:
        terminal.stop(timeout=timeout)
    else:
        if _env_bool("TERMINAL_ENV_FAST_CLOSE", False) or _env_bool(
            "TERMINAL_ENV_SKIP_UNBOUNDED_STOP", False
        ):
            logger.warning(
                "Terminal.stop(timeout=...) is unsupported; skipping unbounded "
                "Terminal.stop() under fast close."
            )
            return
        logger.warning(
            "Terminal.stop(timeout=...) is unsupported; retrying with Terminal.stop()."
        )
        terminal.stop()


def _drain_toolkit_sessions(toolkit: Any) -> None:
    sessions = getattr(toolkit, "shell_sessions", None)
    if not isinstance(sessions, dict):
        return
    lock = getattr(toolkit, "_session_lock", None)
    acquired_lock = False
    try:
        if lock is not None:
            try:
                acquired_lock = bool(lock.acquire(blocking=False))
            except TypeError:
                acquired_lock = bool(lock.acquire(False))
            if not acquired_lock:
                logger.warning(
                    "Skipping TerminalToolkit session drain because session lock "
                    "is currently held."
                )
                return
        for session in sessions.values():
            proc = session.get("process")
            if proc is not None:
                try:
                    if hasattr(proc, "terminate"):
                        proc.terminate()
                    elif hasattr(proc, "close"):
                        proc.close()
                except Exception:
                    pass
            q = session.get("output_stream")
            if q is not None:
                try:
                    while not q.empty():
                        q.get_nowait()
                except Exception:
                    pass
        sessions.clear()
    finally:
        if lock is not None and acquired_lock:
            try:
                lock.release()
            except RuntimeError:
                pass


class TerminalEnv:
    def __init__(self) -> None:
        self._closed = False
        self._task_spec: TaskSpec | None = None
        self._run_ctx: RunContext | None = None
        self._timeouts: TaskTimeouts | None = None

        self._trial_handler: TrialHandler | None = None
        self._terminal: Terminal | None = None
        self._parser = None
        self._terminal_toolkit: TerminalToolkit | None = None
        self._tools: dict[str, Any] = {}
        self._agent_safetybench_env: AgentSafetyBenchEnv | None = None
        self._agentharm_env: AgentHarmEnv | None = None
        self._eval_attempt = 0
        self._last_eval: dict[str, Any] | None = None
        self._last_trial_name: str | None = None
        self._last_client_container_name: str | None = None
        self._last_docker_image_name_prefix: str | None = None
        self._last_docker_compose_path: str | None = None

    async def reset(
        self,
        *,
        task_meta: dict[str, Any],
        task_spec: TaskSpec,
        run_ctx: RunContext,
        timeouts: TaskTimeouts,
    ) -> tuple[str, list[dict[str, Any]]]:
        await self.close()

        self._closed = False
        self._task_spec = task_spec
        self._run_ctx = run_ctx
        self._timeouts = timeouts
        self._eval_attempt = 0
        self._last_eval = None

        if task_meta.get("data_source") == "agent_safetybench":
            self._agent_safetybench_env = AgentSafetyBenchEnv()
            return await self._agent_safetybench_env.reset(
                task_meta=task_meta,
                task_spec=task_spec,
                run_ctx=run_ctx,
            )
        if task_meta.get("data_source") == "agentharm":
            self._agentharm_env = AgentHarmEnv()
            return await self._agentharm_env.reset(
                task_meta=task_meta,
                task_spec=task_spec,
                run_ctx=run_ctx,
            )

        cancel_event = threading.Event()
        reset_started = time.monotonic()
        image_prepare_started = time.monotonic()
        logger.info(
            "TerminalEnv reset image prepare starting task=%s uid=%s timeout=%.1fs",
            self._task_spec.task_name,
            self._run_ctx.uid,
            self._timeouts.ensure_image,
        )
        try:
            image_prep = await asyncio.to_thread(
                prepare_task_docker_image,
                task=task_meta,
                timeout=self._timeouts.ensure_image,
                cancel_event=cancel_event,
            )
        except asyncio.CancelledError:
            cancel_event.set()
            logger.warning(
                "TerminalEnv reset cancelled during image prepare task=%s uid=%s "
                "elapsed=%.1fs total_elapsed=%.1fs",
                self._task_spec.task_name,
                self._run_ctx.uid,
                time.monotonic() - image_prepare_started,
                time.monotonic() - reset_started,
            )
            raise
        except Exception:
            logger.exception(
                "TerminalEnv reset image prepare failed task=%s uid=%s elapsed=%.1fs "
                "total_elapsed=%.1fs",
                self._task_spec.task_name,
                self._run_ctx.uid,
                time.monotonic() - image_prepare_started,
                time.monotonic() - reset_started,
            )
            raise
        logger.info(
            "TerminalEnv reset image prepare finished task=%s uid=%s mode=%s "
            "image=%s elapsed=%.1fs total_elapsed=%.1fs",
            self._task_spec.task_name,
            self._run_ctx.uid,
            getattr(image_prep, "mode", ""),
            getattr(image_prep, "client_image_name", ""),
            time.monotonic() - image_prepare_started,
            time.monotonic() - reset_started,
        )

        dataset_dir = str(os.getenv("DATASET_DIR", "")).strip()
        if not dataset_dir:
            raise ValueError("DATASET_DIR is required")
        task_path = Path(dataset_dir) / self._task_spec.task_path
        output_path = Path(self._run_ctx.log_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        def _raise_if_cancelled(stage: str) -> None:
            if cancel_event.is_set():
                raise RuntimeError(
                    f"TERMINAL_ENV_RESET_CANCELLED task={self._task_spec.task_name} "
                    f"uid={self._run_ctx.uid} stage={stage}"
                )

        def _sync_reset() -> tuple[str, list[dict[str, Any]]]:
            _raise_if_cancelled("before_docker_cleanup")
            # P0 FIX: Force recreate container to avoid Docker daemon API slowdown
            # Root cause: containers.get() HTTP call hangs 360s when container runs >1h
            # Docker daemon state accumulation causes API performance degradation
            # Solution: Delete old container, create fresh one (fast API response)
            container_name = f"{self._task_spec.task_name}.{self._run_ctx.uid}.slime-run"
            try:
                import subprocess
                # Force remove old container (timeout 5s, ignore errors)
                subprocess.run(
                    ['docker', 'rm', '-f', container_name],
                    timeout=5,
                    capture_output=True,
                    check=False
                )
                logger.info(
                    "Forced container recreation for %s to avoid Docker API slowdown",
                    container_name
                )
            except Exception as e:
                logger.debug("Container force-remove failed (may not exist): %s", e)

            self._trial_handler = TrialHandler(
                trial_name=f"{self._task_spec.task_name}.{self._run_ctx.uid}.slime-run",
                input_path=task_path,
                output_path=output_path,
            )
            self._last_trial_name = self._trial_handler.trial_name
            self._last_client_container_name = self._trial_handler.client_container_name
            self._last_docker_image_name_prefix = (
                self._trial_handler.docker_image_name_prefix
            )
            self._last_docker_compose_path = str(
                self._trial_handler.task_paths.docker_compose_path
            )
            _raise_if_cancelled("before_terminal_create")
            task_config = self._trial_handler.task
            self._parser = ParserFactory.get_parser(task_config.parser_name)
            client_image_name = (
                image_prep.client_image_name or self._trial_handler.client_image_name
            )

            self._terminal = Terminal(
                client_container_name=self._trial_handler.client_container_name,
                client_image_name=client_image_name,
                docker_compose_path=self._trial_handler.task_paths.docker_compose_path,
                docker_image_name_prefix=self._trial_handler.docker_image_name_prefix,
                sessions_logs_path=self._trial_handler.trial_paths.sessions_path,
                agent_logs_path=self._trial_handler.trial_paths.agent_logging_dir,
                no_rebuild=True,
                cleanup=False,
            )
            docker_start_started = time.monotonic()
            logger.info(
                "TerminalEnv docker start starting task=%s uid=%s container=%s "
                "mode=%s timeout=%.1fs",
                self._task_spec.task_name,
                self._run_ctx.uid,
                self._trial_handler.client_container_name,
                image_prep.mode,
                self._timeouts.reset_session,
            )
            try:
                _raise_if_cancelled("before_docker_start")
                if image_prep.mode == "pull":
                    compose_up_no_build(
                        self._terminal,
                        timeout=self._timeouts.reset_session,
                        container_name=self._trial_handler.client_container_name,
                        logger=logger,
                    )
                else:
                    import inspect
                    if "timeout" in inspect.signature(self._terminal.start).parameters:
                        self._terminal.start(timeout=self._timeouts.reset_session)
                    else:
                        self._terminal.start()
                    try:
                        from .docker_compose_utils import (
                            _apply_container_runtime_limits,
                        )

                        _apply_container_runtime_limits(
                            self._trial_handler.client_container_name,
                            logger=logger,
                        )
                    except Exception:
                        pass
            except Exception:
                logger.exception(
                    "TerminalEnv docker start failed task=%s uid=%s container=%s "
                    "mode=%s elapsed=%.1fs total_elapsed=%.1fs",
                    self._task_spec.task_name,
                    self._run_ctx.uid,
                    self._trial_handler.client_container_name,
                    image_prep.mode,
                    time.monotonic() - docker_start_started,
                    time.monotonic() - reset_started,
                )
                _force_remove_docker_objects(
                    trial_name=self._trial_handler.trial_name,
                    client_container_name=self._trial_handler.client_container_name,
                    docker_image_name_prefix=self._trial_handler.docker_image_name_prefix,
                    docker_compose_path=str(
                        self._trial_handler.task_paths.docker_compose_path
                    ),
                    reason="reset_start_failed",
                )
                raise
            logger.info(
                "TerminalEnv docker start finished task=%s uid=%s container=%s "
                "mode=%s elapsed=%.1fs total_elapsed=%.1fs",
                self._task_spec.task_name,
                self._run_ctx.uid,
                self._trial_handler.client_container_name,
                image_prep.mode,
                time.monotonic() - docker_start_started,
                time.monotonic() - reset_started,
            )

            try:
                _raise_if_cancelled("after_docker_start")
            except Exception:
                logger.warning(
                    "TerminalEnv reset cancelled after docker start task=%s uid=%s; "
                    "cleaning up container=%s",
                    self._task_spec.task_name,
                    self._run_ctx.uid,
                    self._trial_handler.client_container_name,
                )
                _force_remove_docker_objects(
                    trial_name=self._trial_handler.trial_name,
                    client_container_name=self._trial_handler.client_container_name,
                    docker_image_name_prefix=self._trial_handler.docker_image_name_prefix,
                    docker_compose_path=str(
                        self._trial_handler.task_paths.docker_compose_path
                    ),
                    reason="reset_cancelled_after_start",
                )
                raise
            session_logs_dir = (
                self._trial_handler.trial_paths.sessions_path
                / "terminal_toolkit_session_logs"
            )
            self._terminal_toolkit = TerminalToolkit(
                timeout=20.0,
                working_directory=None,
                use_docker_backend=True,
                docker_container_name=self._trial_handler.client_container_name,
                session_logs_dir=session_logs_dir,
                safe_mode=False,
            )
            self._tools = {
                "shell_exec": self._terminal_toolkit.shell_exec,
                "shell_view": self._terminal_toolkit.shell_view,
                "shell_write_to_process": self._terminal_toolkit.shell_write_to_process,
                "shell_write_content_to_file": self._terminal_toolkit.shell_write_content_to_file,
            }

            user_msg = f"Task name:{self._task_spec.task_name}\nTask instruction: {self._task_spec.instruction}"
            function_tools = [FunctionTool(fn) for fn in self._tools.values()]
            tool_schemas = [
                func_tool.get_openai_tool_schema() for func_tool in function_tools
            ]
            return user_msg, tool_schemas

        # Keep a bounded wrapper around Docker/session startup, but leave enough
        # grace for slow compose starts after image preparation has completed.
        reset_thread_timeout = _env_float(
            "TERMINAL_ENV_RESET_THREAD_TIMEOUT",
            float(self._timeouts.reset_session) + 120.0,
        )
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_sync_reset),
                timeout=reset_thread_timeout,
            )
        except asyncio.CancelledError:
            cancel_event.set()
            logger.warning(
                "TerminalEnv reset cancelled during docker/session startup task=%s "
                "uid=%s total_elapsed=%.1fs",
                self._task_spec.task_name,
                self._run_ctx.uid,
                time.monotonic() - reset_started,
            )
            raise
        except asyncio.TimeoutError:
            cancel_event.set()
            logger.error(
                "CRITICAL: reset operation hung beyond internal timeout "
                f"(timeout={reset_thread_timeout}s, reset_session={self._timeouts.reset_session}s). "
                "Thread may still be running. "
                "This indicates Docker operations are stuck. Manual intervention may be required."
            )
            # Thread will continue running in background - this is a known limitation
            # of asyncio.to_thread. The watchdog should detect this and restart the worker.
            raise TimeoutError(
                f"Reset operation exceeded timeout ({reset_thread_timeout}s). "
                "Docker operations may be hung. Worker may need restart."
            )

    async def exec_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._agent_safetybench_env is not None:
            return await self._agent_safetybench_env.exec_tool(name, arguments)
        if self._agentharm_env is not None:
            return await self._agentharm_env.exec_tool(name, arguments)

        if not self._tools:
            raise RuntimeError("env is not initialized; call reset first")

        if name not in self._tools:
            return f"[TOOL_ERROR] unknown tool: {name}"

        fn = self._tools[name]

        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(**arguments)
            elif hasattr(fn, "async_call") and callable(fn.async_call):
                result = await fn.async_call(**arguments)
            else:
                result = await asyncio.to_thread(partial(fn, **arguments))
        except Exception as exc:
            return f"[TOOL_ERROR] {name}: {type(exc).__name__}: {exc}"

        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    async def evaluate(self, trajectory: dict[str, Any] | None = None) -> float:
        if self._agent_safetybench_env is not None:
            return await self._agent_safetybench_env.evaluate(trajectory)
        if self._agentharm_env is not None:
            return await self._agentharm_env.evaluate(trajectory)

        if (
            self._trial_handler is None
            or self._terminal is None
            or self._parser is None
            or self._timeouts is None
        ):
            raise RuntimeError("env is not initialized; call reset first")

        def _sync_eval() -> float:
            task_name = (
                self._task_spec.task_name if self._task_spec is not None else "unknown"
            )
            paths: list[Path] = [self._trial_handler.task_paths.run_tests_path]
            if self._trial_handler.task_paths.test_dir.exists():
                paths.append(self._trial_handler.task_paths.test_dir)

            self._terminal.copy_to_container(
                paths=paths,
                container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
            )

            self._eval_attempt += 1
            run_uid = self._run_ctx.uid if self._run_ctx is not None else "unknown"
            test_session = self._terminal.create_session(
                f"tests-{run_uid}-{self._eval_attempt}",
                is_active_stream=False,
                as_configured_user=False,
            )
            test_script_path = str(
                DockerComposeManager.CONTAINER_TEST_DIR / "run-tests.sh"
            )
            test_timeout_sec = min(
                self._timeouts.eval,
                4 * self._trial_handler.task.max_test_timeout_sec,
            )
            try:
                test_session.send_keys(
                    [f"bash {test_script_path}", "Enter"],
                    block=True,
                    max_timeout_sec=test_timeout_sec,
                )
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

            test_output = test_session.capture_pane(capture_entire=True)
            try:
                parser_results = self._parser.parse(test_output)
            except Exception as exc:
                tail = test_output[-2000:] if test_output else ""
                logger.warning(
                    "Failed to parse test output for task=%s with parser=%s: %s. Output tail:\n%s",
                    task_name,
                    type(self._parser).__name__,
                    exc,
                    tail,
                )
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
            passed = sum(
                1
                for status in parser_results.values()
                if status == UnitTestStatus.PASSED
            )
            reward = (
                float(passed / len(parser_results)) if len(parser_results) > 0 else 0.0
            )
            self._last_eval = None
            return reward

        return await asyncio.wait_for(
            asyncio.to_thread(_sync_eval),
            timeout=self._timeouts.eval + 30.0,
        )

    def last_eval_details(self) -> dict[str, Any] | None:
        if self._agent_safetybench_env is not None:
            details = getattr(self._agent_safetybench_env, "_last_eval", None)
            return details if isinstance(details, dict) else None
        if self._agentharm_env is not None:
            details = getattr(self._agentharm_env, "_last_eval", None)
            return details if isinstance(details, dict) else None
        return self._last_eval if isinstance(self._last_eval, dict) else None

    async def close(self) -> None:
        trial_name = (
            self._trial_handler.trial_name
            if self._trial_handler is not None
            else self._last_trial_name or "unknown"
        )
        client_container_name = (
            self._trial_handler.client_container_name
            if self._trial_handler is not None
            else self._last_client_container_name
        )
        docker_image_name_prefix = (
            self._trial_handler.docker_image_name_prefix
            if self._trial_handler is not None
            else self._last_docker_image_name_prefix
        )
        docker_compose_path = (
            str(self._trial_handler.task_paths.docker_compose_path)
            if self._trial_handler is not None
            else self._last_docker_compose_path
        )
        if self._closed:
            logger.warning("TerminalEnv %s already closed", trial_name)
            return
        self._closed = True

        terminal = self._terminal
        timeouts = self._timeouts
        toolkit = self._terminal_toolkit
        agent_safetybench_env = self._agent_safetybench_env
        agentharm_env = self._agentharm_env

        self._tools = {}
        self._terminal = None
        self._trial_handler = None
        self._parser = None
        self._terminal_toolkit = None
        self._task_spec = None
        self._run_ctx = None
        self._timeouts = None
        self._agent_safetybench_env = None
        self._agentharm_env = None
        self._last_eval = None

        cleanup_completed = terminal is None
        cleanup_error = False
        fast_close = _env_bool("TERMINAL_ENV_FAST_CLOSE", False)
        force_cleanup_started = False

        async def _run_force_cleanup(reason: str) -> None:
            nonlocal force_cleanup_started
            force_cleanup_started = True
            await _force_remove_docker_objects_async(
                trial_name=trial_name,
                client_container_name=client_container_name,
                docker_image_name_prefix=docker_image_name_prefix,
                docker_compose_path=docker_compose_path,
                reason=reason,
            )

        try:
            if agent_safetybench_env is not None:
                try:
                    await agent_safetybench_env.close()
                except Exception:
                    logger.exception(
                        "Failed to cleanup Agent-SafetyBench env for %s", trial_name
                    )

            if agentharm_env is not None:
                try:
                    await agentharm_env.close()
                except Exception:
                    logger.exception(
                        "Failed to cleanup AgentHarm env for %s", trial_name
                    )

            if fast_close and terminal is not None:
                try:
                    await _run_force_cleanup("fast_close")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    cleanup_error = True
                    logger.exception(
                        "Force Docker cleanup failed for TerminalEnv %s", trial_name
                    )

            if toolkit is not None:
                if fast_close:
                    logger.warning(
                        "Fast close enabled for %s; skipping TerminalToolkit.cleanup "
                        "and session drain; relying on direct Docker cleanup.",
                        trial_name,
                    )
                else:
                    try:
                        await asyncio.to_thread(toolkit.cleanup)
                    except Exception:
                        cleanup_error = True
                        logger.exception(
                            "Failed to cleanup terminal toolkit for %s", trial_name
                        )
                    try:
                        await asyncio.to_thread(_drain_toolkit_sessions, toolkit)
                    except Exception:
                        cleanup_error = True
                        logger.exception(
                            "Failed to drain toolkit sessions for %s", trial_name
                        )

            if terminal is not None and timeouts is not None:
                try:
                    close_timeout = timeouts.close_session
                    if fast_close:
                        close_timeout = min(
                            close_timeout,
                            _env_float("TERMINAL_ENV_FAST_CLOSE_STOP_TIMEOUT", 5.0),
                        )
                    await asyncio.to_thread(
                        _stop_terminal_compat, terminal, close_timeout
                    )
                    cleanup_completed = True
                    logger.info("TerminalEnv %s closed", trial_name)
                except Exception:
                    cleanup_error = True
                    logger.exception("Failed to stop terminal session during close")
        finally:
            force_always = _env_bool("TERMINAL_ENV_FORCE_DOCKER_CLEANUP_ALWAYS", False)
            compose_down_on_close = _env_bool("TERMINAL_ENV_COMPOSE_DOWN_ON_CLOSE", True)
            force_needed = (
                compose_down_on_close
                or force_always
                or fast_close
                or cleanup_error
                or not cleanup_completed
            )
            if terminal is not None and force_needed and not force_cleanup_started:
                if fast_close:
                    reason = "fast_close"
                elif compose_down_on_close and cleanup_completed and not cleanup_error:
                    reason = "close_compose_down"
                elif force_always and cleanup_completed and not cleanup_error:
                    reason = "always"
                else:
                    reason = "close_incomplete"
                try:
                    await _run_force_cleanup(reason)
                except Exception:
                    logger.exception(
                        "Force Docker cleanup failed for TerminalEnv %s", trial_name
                    )

    async def force_cleanup(self, reason: str = "external") -> None:
        await _force_remove_docker_objects_async(
            trial_name=self._last_trial_name or "unknown",
            client_container_name=self._last_client_container_name,
            docker_image_name_prefix=self._last_docker_image_name_prefix,
            docker_compose_path=self._last_docker_compose_path,
            reason=reason,
        )
