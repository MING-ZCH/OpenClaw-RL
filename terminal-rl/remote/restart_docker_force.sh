#!/usr/bin/env bash
# restart_docker_force.sh — 强制清理并重启 docker（systemd 优先，失败 fallback nohup）
# 用法:
#   sudo env DOCKER_DATA_ROOT=/data bash terminal-rl/remote/restart_docker_force.sh
#
# Env:
#   DOCKER_DATA_ROOT  Docker data root. DOCKER_ROOT is accepted as legacy alias.
#   PROXY_ENV_FILE    Proxy env file sourced before starting dockerd. Default: /etc/seta_build_proxy.env
#   DOCKER_START_WAIT_SECONDS  seconds to wait for dockerd API per attempt. Default: 600
#   FORCE_RESTART_CONTAINERD   1=restart containerd and clear runtime state. Default: 1
#   DOCKER_RUNTIME_RETRY       1=retry dockerd once after runtime cleanup. Default: 1
#   USE_SYSTEMD_START          1=start docker via systemd first, fallback to nohup. Default: 1

set -uo pipefail

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "[ERROR] $*"; exit 1; }

if [ "$(id -u)" -ne 0 ]; then
    die "Must run as root. Use: sudo env DOCKER_DATA_ROOT=/data bash terminal-rl/remote/restart_docker_force.sh"
fi

DATA_ROOT="${DOCKER_DATA_ROOT:-${DOCKER_ROOT:-/data}}"
PROXY_ENV_FILE="${PROXY_ENV_FILE:-/etc/seta_build_proxy.env}"
DOCKER_START_WAIT_SECONDS="${DOCKER_START_WAIT_SECONDS:-600}"
FORCE_RESTART_CONTAINERD="${FORCE_RESTART_CONTAINERD:-1}"
DOCKER_RUNTIME_RETRY="${DOCKER_RUNTIME_RETRY:-1}"
USE_SYSTEMD_START="${USE_SYSTEMD_START:-1}"
LOG_FILE=""

log "Force restarting docker on $(hostname)"
log "  DOCKER_DATA_ROOT=${DATA_ROOT}"
log "  PROXY_ENV_FILE=${PROXY_ENV_FILE}"
log "  DOCKER_START_WAIT=${DOCKER_START_WAIT_SECONDS}s"
log "  RESTART_CONTAINERD=${FORCE_RESTART_CONTAINERD}"
log "  RUNTIME_RETRY=${DOCKER_RUNTIME_RETRY}"
log "  USE_SYSTEMD_START=${USE_SYSTEMD_START}"

dump_docker_start_diagnostics() {
    local log_file="${1:-}"
    log "  diagnostics: dockerd/containerd process state"
    pgrep -a -x dockerd 2>/dev/null | sed 's/^/    /' || true
    pgrep -a -x containerd 2>/dev/null | sed 's/^/    /' || true
    pgrep -a -f containerd-shim 2>/dev/null | head -20 | sed 's/^/    /' || true
    if [ -n "${log_file}" ] && [ -f "${log_file}" ]; then
        log "  dockerd log tail (${log_file}):"
        tail -80 "${log_file}" 2>/dev/null | sed 's/^/    /' || true
    fi
    log "  journal docker tail:"
    journalctl -u docker -n 80 --no-pager 2>/dev/null | sed 's/^/    /' || true
    log "  journal containerd tail:"
    journalctl -u containerd -n 80 --no-pager 2>/dev/null | sed 's/^/    /' || true
}

cleanup_docker_runtime_state() {
    log "  cleaning Docker runtime state under /run (not Docker data-root layers/images)"
    rm -f /var/run/docker.pid /var/run/docker.sock
    rm -rf /run/docker 2>/dev/null || true
    rm -rf /run/containerd/io.containerd.runtime.v2.task/moby 2>/dev/null || true
    rm -rf /run/containerd/io.containerd.grpc.v1.introspection 2>/dev/null || true
}

restart_containerd_clean() {
    if [ "${FORCE_RESTART_CONTAINERD}" != "1" ]; then
        if ! pgrep -x containerd >/dev/null 2>&1; then
            log "  containerd not running, starting via systemd..."
            timeout 10 systemctl start containerd 2>/dev/null || log "  WARN: systemctl start containerd timed out"
            sleep 3
        else
            log "  containerd already running: $(pgrep -x containerd | xargs)"
        fi
        return 0
    fi

    log "  restarting containerd cleanly after shim cleanup"
    timeout 10 systemctl stop containerd 2>/dev/null || log "  WARN: systemctl stop containerd timed out"
    pkill -9 -x containerd 2>/dev/null || true
    cleanup_docker_runtime_state
    timeout 20 systemctl start containerd 2>/dev/null || log "  WARN: systemctl start containerd timed out"
    sleep 3
    if pgrep -x containerd >/dev/null 2>&1; then
        log "  [OK] containerd running: $(pgrep -x containerd | xargs)"
    else
        log "  WARN: containerd still not visible after restart"
    fi
}

load_proxy_env() {
    if [ -f "${PROXY_ENV_FILE}" ]; then
        # shellcheck disable=SC1090
        set -a; . "${PROXY_ENV_FILE}"; set +a
        log "  Loaded proxy env from ${PROXY_ENV_FILE}"
    else
        log "  [SKIP] proxy env file not found: ${PROXY_ENV_FILE}"
    fi
}

start_dockerd_direct() {
    load_proxy_env
    LOG_FILE="/tmp/dockerd_start_$(date +%H%M%S).log"
    nohup dockerd --containerd=/run/containerd/containerd.sock > "${LOG_FILE}" 2>&1 &
    DOCKERD_PID=$!
    disown 2>/dev/null || true
    log "  nohup started dockerd PID=${DOCKERD_PID}, log: ${LOG_FILE}"
}

wait_for_dockerd_api() {
    local label="$1"
    local max_attempts=$((DOCKER_START_WAIT_SECONDS / 5))
    local i
    [ "${max_attempts}" -gt 0 ] 2>/dev/null || max_attempts=120
    log "  waiting for dockerd API (${label}, up to ${DOCKER_START_WAIT_SECONDS}s) ..."
    for i in $(seq 1 "${max_attempts}"); do
        if timeout 10 docker info >/dev/null 2>&1; then
            log "  [OK] dockerd API ready after ~$((i*5))s (${label})"
            return 0
        fi
        if ! pgrep -x dockerd >/dev/null 2>&1; then
            log "  [ERROR] dockerd process died during startup (${label})"
            dump_docker_start_diagnostics "${LOG_FILE:-}"
            return 1
        fi
        if [ $((i % 6)) -eq 0 ]; then
            log "    still waiting ... (~$((i*5))s elapsed, ${label})"
            [ -n "${LOG_FILE:-}" ] && tail -3 "${LOG_FILE}" 2>/dev/null | sed 's/^/      /' || true
        fi
        sleep 5
    done
    log "  [ERROR] dockerd not ready after ${DOCKER_START_WAIT_SECONDS}s (${label})"
    dump_docker_start_diagnostics "${LOG_FILE:-}"
    return 1
}

# ── Step 1: 直接 pkill -9，并阻止 docker.socket race ──────────────────
log "Step 1: Kill dockerd and prevent docker.socket races"

# 1a. 找 dockerd PID
DOCKERD_PIDS=$(pgrep -x dockerd 2>/dev/null || true)
if [ -n "${DOCKERD_PIDS}" ]; then
    log "  Found dockerd PIDs: ${DOCKERD_PIDS}"
    for pid in ${DOCKERD_PIDS}; do
        log "    kill -9 ${pid}"
        kill -9 "${pid}" 2>/dev/null || true
    done
    sleep 2
    if pgrep -x dockerd >/dev/null 2>&1; then
        log "  ⚠️ dockerd still alive after SIGKILL (may be in D state, kernel will reap eventually)"
    else
        log "  ✓ dockerd killed"
    fi
else
    log "  No dockerd running"
fi

# 1b. 杀残留 shim（直接 pkill，不要 -f 全匹配）
SHIM_PIDS=$(pgrep containerd-shim 2>/dev/null || true)
if [ -n "${SHIM_PIDS}" ]; then
    SHIM_COUNT=$(echo "${SHIM_PIDS}" | wc -l)
    log "  Killing ${SHIM_COUNT} containerd-shim processes..."
    echo "${SHIM_PIDS}" | xargs -r kill -9 2>/dev/null || true
    sleep 1
fi

# 1c. 阻止 docker.socket 自动拉起 dockerd（避免 race）
log "  disabling docker.socket auto-start..."
timeout 5 systemctl reset-failed docker.service docker.socket 2>/dev/null || log "  (reset-failed timed out, ignoring)"
timeout 5 systemctl stop docker.socket 2>/dev/null || log "  (stop docker.socket timed out, ignoring)"

# ── Step 2: 清理残留文件 ──────────────────────────────────────────────
log "Step 2: Clean stale files"
cleanup_docker_runtime_state
log "  Removed docker.pid, docker.sock and stale /run runtime state"

# ── Step 3: 删除旧容器状态 ────────────────────────────────────────────
log "Step 3: Remove stale container state"
if [ -d "${DATA_ROOT}/containers" ]; then
    N=$(ls "${DATA_ROOT}/containers" 2>/dev/null | wc -l)
    log "  Found ${N} stale containers, removing..."
    rm -rf "${DATA_ROOT}/containers"/* 2>/dev/null || true
fi
rm -f "${DATA_ROOT}/network/files/local-kv.db" 2>/dev/null
log "  Done"

# ── Step 4: 重启 containerd 并清理 runtime state ──────────────────────
log "Step 4: Restart/ensure containerd"
restart_containerd_clean

# ── Step 5: 启动 dockerd（systemd 优先，失败后 direct nohup）───────────
log "Step 5: Start dockerd"
if [ "${USE_SYSTEMD_START}" = "1" ]; then
    timeout 15 systemctl reset-failed docker.service docker.socket 2>/dev/null || true
    if timeout 90 systemctl start docker 2>&1 | sed 's/^/  /'; then
        log "  [OK] systemctl start docker returned"
    else
        log "  WARN: systemctl start docker failed; falling back to direct nohup"
        start_dockerd_direct
    fi
else
    log "  USE_SYSTEMD_START=0; using direct nohup"
    start_dockerd_direct
fi

# ── Step 6: 等待 API ready，必要时清理 runtime 后重试一次 ─────────────
log "Step 6: Waiting for docker API"
if ! wait_for_dockerd_api "initial start"; then
    if [ "${DOCKER_RUNTIME_RETRY}" = "1" ]; then
        log "Step 6 retry: dockerd did not become ready; forcing runtime cleanup and retrying once"
        pkill -9 -x dockerd 2>/dev/null || true
        SHIM_PIDS=$(pgrep containerd-shim 2>/dev/null || true)
        if [ -n "${SHIM_PIDS}" ]; then
            log "  retry cleanup: killing $(echo "${SHIM_PIDS}" | wc -l) remaining containerd-shim"
            echo "${SHIM_PIDS}" | xargs -r kill -9 2>/dev/null || true
        fi
        restart_containerd_clean
        start_dockerd_direct
        wait_for_dockerd_api "runtime cleanup retry" || exit 1
    else
        exit 1
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────
log "✅ Done. Docker is ready."
DOCKERD_PID=$(pgrep -x dockerd | head -1)
if [ -n "${DOCKERD_PID}" ] && grep -aq "HTTP_PROXY=" "/proc/${DOCKERD_PID}/environ" 2>/dev/null; then
    log "  [OK] dockerd PID=${DOCKERD_PID} has HTTP_PROXY in env"
elif [ -n "${DOCKERD_PID}" ]; then
    log "  [WARN] dockerd PID=${DOCKERD_PID} env does not contain HTTP_PROXY"
fi
docker info 2>&1 | grep -E "Containers:|Running:|Stopped:|Images:|Server Version"
echo
df -h "${DATA_ROOT}" 2>&1 | grep -F "${DATA_ROOT}" || true
echo
log "Next: bash terminal-rl/remote/run_pool_server_pu_v2.sh"
