#!/usr/bin/env bash
# fix_dockerd_and_proxy.sh — One-shot CPU worker repair (watchdog-aware).
#
# Solves the chained failure mode observed on prod:
#   1) `setup_new_worker.sh` interrupted at "Restarting docker daemon..."
#   2) `systemctl restart docker` no longer works because docker-watchdog
#      restarts dockerd via direct `nohup`, leaving systemd's view inactive
#   3) Stale state in /var/run/docker.{sock,pid} and /data/containers/
#   4) seta_env builds fail with exit-17 because dockerd has no HTTP_PROXY
#
# Strategy (run as root on CPU worker):
#   Phase 0  pre-flight (proxy reachable?)
#   Phase 1  stop docker-watchdog (avoid race with our restart)
#   Phase 2  force-restart dockerd (pkill-9 + clean state, like restart_docker_force.sh)
#   Phase 3  write ALL proxy configs:
#              - /etc/systemd/system/docker.service.d/http-proxy.conf
#              - /etc/systemd/system/docker-watchdog.service.d/http-proxy.conf  ← critical
#              - per-user ~/.docker/config.json (build-time HTTP_PROXY auto-inject)
#              - /etc/seta_build_proxy.env (for pool_server to source)
#   Phase 4  start dockerd via systemd (with proxy env), fallback to nohup
#   Phase 5  verify dockerd has HTTP_PROXY in its env
#   Phase 6  verify seta_env/0 builds successfully
#   Phase 7  restart docker-watchdog (if it was running)
#
# Usage (CPU worker, as root):
#   sudo bash terminal-rl/remote/fix_dockerd_and_proxy.sh
#
# Env vars (all optional):
#   PROXY_URL         default: http://httpproxy-headless.kubebrain.svc.pjlab.local:3128
#   NO_PROXY_LIST     default: localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,.pjlab.local,.svc
#   DOCKER_DATA_ROOT  default: /data. DOCKER_ROOT is accepted as legacy alias.
#   START_WATCHDOG    1=auto-start watchdog at end (default), 0=leave stopped
#   SKIP_VERIFY       1=skip seta_env/0 build test (default 0)
#   DOCKER_START_WAIT_SECONDS  seconds to wait for dockerd API per attempt (default 300)
#   FORCE_RESTART_CONTAINERD   1=restart containerd and clear runtime state in Phase 2 (default 1)
#   DOCKER_RUNTIME_RETRY       1=retry dockerd once after runtime cleanup if first start hangs (default 1)
#   SETA              path to seta_env/0/ (default: in this repo)

set -uo pipefail

PROXY_URL="${PROXY_URL:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128}"
NO_PROXY_LIST="${NO_PROXY_LIST:-localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn,.pjlab.local,.svc}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-${DOCKER_ROOT:-/data}}"
START_WATCHDOG="${START_WATCHDOG:-1}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
# DOCKER_START_WAIT_SECONDS="${DOCKER_START_WAIT_SECONDS:-300}"
DOCKER_START_WAIT_SECONDS="${DOCKER_START_WAIT_SECONDS:-600}"
FORCE_RESTART_CONTAINERD="${FORCE_RESTART_CONTAINERD:-1}"
DOCKER_RUNTIME_RETRY="${DOCKER_RUNTIME_RETRY:-1}"
SETA="${SETA:-/mnt/shared-storage-user/puyuan/code/OpenClaw-RL/terminal-rl/dataset/seta_env/0}"

USERS_TO_CONFIGURE=("root")
id puyuan >/dev/null 2>&1 && USERS_TO_CONFIGURE+=("puyuan")

if [ "$(id -u)" -ne 0 ]; then
  echo "[ERROR] Must run as root (systemctl / writes to /etc)"; exit 1
fi

log() { echo "[$(date '+%F %T')] $*"; }
hr()  { echo "------------------------------------------------------------"; }

hr
log "fix_dockerd_and_proxy.sh"
log "  PROXY_URL         = ${PROXY_URL}"
log "  NO_PROXY_LIST     = ${NO_PROXY_LIST}"
log "  DOCKER_DATA_ROOT  = ${DOCKER_DATA_ROOT}"
log "  START_WATCHDOG    = ${START_WATCHDOG}"
log "  SKIP_VERIFY       = ${SKIP_VERIFY}"
log "  DOCKER_START_WAIT = ${DOCKER_START_WAIT_SECONDS}s"
log "  RESTART_CONTAINERD= ${FORCE_RESTART_CONTAINERD}"
log "  RUNTIME_RETRY     = ${DOCKER_RUNTIME_RETRY}"
log "  USERS             = ${USERS_TO_CONFIGURE[*]}"
hr

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
      log "  containerd not running, starting via systemd ..."
      timeout 10 systemctl start containerd 2>/dev/null || \
        log "  WARN: systemctl start containerd timed out"
      sleep 3
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

start_dockerd_direct() {
  LOG_FILE="/tmp/dockerd_fix_$(date +%H%M%S).log"
  export HTTP_PROXY="${PROXY_URL}" HTTPS_PROXY="${PROXY_URL}" NO_PROXY="${NO_PROXY_LIST}"
  export http_proxy="${PROXY_URL}" https_proxy="${PROXY_URL}" no_proxy="${NO_PROXY_LIST}"
  nohup dockerd --containerd=/run/containerd/containerd.sock > "${LOG_FILE}" 2>&1 &
  DOCKERD_PID=$!
  disown 2>/dev/null || true
  log "  nohup started dockerd PID=${DOCKERD_PID}, log: ${LOG_FILE}"
}

wait_for_dockerd_api() {
  local label="$1"
  local max_attempts=$((DOCKER_START_WAIT_SECONDS / 5))
  local i
  [ "${max_attempts}" -gt 0 ] 2>/dev/null || max_attempts=60
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

# ─── Phase 0: pre-flight ─────────────────────────────────────────────
log "Phase 0: pre-flight (proxy reachability)"
if ! timeout 5 curl -fsS -x "${PROXY_URL}" http://example.com >/dev/null 2>&1; then
  log "[ERROR] Proxy ${PROXY_URL} not reachable. Check DNS / network egress. Aborting."
  exit 2
fi
log "  [OK] proxy reachable"

# ─── Phase 1: stop docker-watchdog ───────────────────────────────────
log "Phase 1: stop docker-watchdog"
WATCHDOG_WAS_RUNNING=0
WATCHDOG_PRESENT=0
if systemctl list-unit-files docker-watchdog.service >/dev/null 2>&1; then
  WATCHDOG_PRESENT=1
  if systemctl is-active --quiet docker-watchdog; then
    WATCHDOG_WAS_RUNNING=1
    timeout 15 systemctl stop docker-watchdog 2>&1 | sed 's/^/  /' || \
      log "  WARN: stop docker-watchdog timed out"
    log "  [OK] docker-watchdog stopped (was running)"
  else
    log "  [SKIP] docker-watchdog unit present but inactive"
  fi
else
  log "  [SKIP] no docker-watchdog.service installed"
fi

# ─── Phase 2: force-restart dockerd ──────────────────────────────────
log "Phase 2: force-restart dockerd (skip systemctl, clean stale state)"
if [ ! -d "${DOCKER_DATA_ROOT}" ]; then
  log "  creating DOCKER_DATA_ROOT=${DOCKER_DATA_ROOT}"
  mkdir -p "${DOCKER_DATA_ROOT}"
fi

# 2a. Tell systemd not to fight us
timeout 5 systemctl reset-failed docker.service docker.socket 2>/dev/null || true
timeout 5 systemctl stop docker.socket 2>/dev/null || true

# 2b. Kill all dockerd / containerd-shim (whether systemd or nohup started)
if pgrep -x dockerd >/dev/null 2>&1; then
  log "  killing existing dockerd PIDs: $(pgrep -x dockerd | xargs)"
  pkill -9 -x dockerd 2>/dev/null || true
fi
SHIM_PIDS=$(pgrep containerd-shim 2>/dev/null || true)
if [ -n "${SHIM_PIDS}" ]; then
  log "  killing $(echo "${SHIM_PIDS}" | wc -l) containerd-shim"
  echo "${SHIM_PIDS}" | xargs -r kill -9 2>/dev/null || true
fi
sleep 2

# 2c. Clean stale state
log "  cleaning /var/run/docker.{pid,sock} and ${DOCKER_DATA_ROOT}/containers/*"
rm -f /var/run/docker.pid /var/run/docker.sock
if [ -d "${DOCKER_DATA_ROOT}/containers" ]; then
  N=$(ls "${DOCKER_DATA_ROOT}/containers" 2>/dev/null | wc -l)
  [ "$N" -gt 0 ] && rm -rf "${DOCKER_DATA_ROOT}/containers"/* 2>/dev/null || true
  log "    removed ${N} stale container dirs"
fi
rm -f "${DOCKER_DATA_ROOT}/network/files/local-kv.db" 2>/dev/null || true

# 2d. Restart containerd and clear stale runtime task state.
restart_containerd_clean
log "  [OK] phase 2 complete"

# ─── Phase 3: write proxy configs ────────────────────────────────────
log "Phase 3: write proxy configs"

# 3a. systemd dropin for dockerd
mkdir -p /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=${PROXY_URL}"
Environment="HTTPS_PROXY=${PROXY_URL}"
Environment="NO_PROXY=${NO_PROXY_LIST}"
EOF
log "  [OK] /etc/systemd/system/docker.service.d/http-proxy.conf"

# 3b. systemd dropin for docker-watchdog (critical: watchdog nohup-spawns dockerd)
if [ "${WATCHDOG_PRESENT}" = "1" ]; then
  mkdir -p /etc/systemd/system/docker-watchdog.service.d
  cat > /etc/systemd/system/docker-watchdog.service.d/http-proxy.conf <<EOF
[Service]
# Override docker-watchdog.service's empty proxy env so that the dockerd
# it spawns via nohup (see restart_docker() in docker_watchdog_v2.sh)
# inherits HTTP_PROXY. Watchdog itself talks only to loopback, so its
# own HTTP calls are still bypassed by NO_PROXY=* in the base unit.
Environment="HTTP_PROXY=${PROXY_URL}"
Environment="HTTPS_PROXY=${PROXY_URL}"
Environment="NO_PROXY=${NO_PROXY_LIST}"
EOF
  log "  [OK] /etc/systemd/system/docker-watchdog.service.d/http-proxy.conf"
fi

# 3c. per-user ~/.docker/config.json (build-time HTTP_PROXY auto-injection)
for u in "${USERS_TO_CONFIGURE[@]}"; do
  home_dir=$(getent passwd "$u" | cut -d: -f6)
  [ -z "${home_dir}" ] && continue
  cfg_dir="${home_dir}/.docker"
  cfg_file="${cfg_dir}/config.json"
  mkdir -p "${cfg_dir}"
  if [ -f "${cfg_file}" ]; then
    python3 - "${cfg_file}" "${PROXY_URL}" "${NO_PROXY_LIST}" <<'PY'
import json, sys
path, proxy, no_proxy = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path) as f: cfg = json.load(f)
except Exception:
    cfg = {}
cfg["proxies"] = {"default": {"httpProxy": proxy, "httpsProxy": proxy, "noProxy": no_proxy}}
with open(path, "w") as f: json.dump(cfg, f, indent=2)
PY
  else
    cat > "${cfg_file}" <<EOF
{
  "proxies": {
    "default": {
      "httpProxy":  "${PROXY_URL}",
      "httpsProxy": "${PROXY_URL}",
      "noProxy":    "${NO_PROXY_LIST}"
    }
  }
}
EOF
  fi
  chown -R "${u}:" "${cfg_dir}" 2>/dev/null || true
  log "  [OK] ${cfg_file}"
done

# 3d. /etc/seta_build_proxy.env for pool_server to source
cat > /etc/seta_build_proxy.env <<EOF
HTTP_PROXY=${PROXY_URL}
HTTPS_PROXY=${PROXY_URL}
http_proxy=${PROXY_URL}
https_proxy=${PROXY_URL}
NO_PROXY=${NO_PROXY_LIST}
no_proxy=${NO_PROXY_LIST}
EOF
log "  [OK] /etc/seta_build_proxy.env"

systemctl daemon-reload
log "  [OK] systemctl daemon-reload done"

# ─── Phase 4: start dockerd via systemd (with proxy) ─────────────────
log "Phase 4: start dockerd via systemd"
timeout 15 systemctl reset-failed docker.service docker.socket 2>/dev/null || true
SYSTEMD_OK=0
if timeout 90 systemctl start docker 2>&1 | sed 's/^/  /'; then
  SYSTEMD_OK=1
  log "  [OK] systemctl start docker returned"
else
  log "  WARN: systemctl start docker failed; falling back to direct nohup"
  start_dockerd_direct
fi

if ! wait_for_dockerd_api "initial start"; then
  if [ "${DOCKER_RUNTIME_RETRY}" = "1" ]; then
    log "Phase 4 retry: dockerd did not become ready; forcing runtime cleanup and retrying once"
    pkill -9 -x dockerd 2>/dev/null || true
    SHIM_PIDS=$(pgrep containerd-shim 2>/dev/null || true)
    if [ -n "${SHIM_PIDS}" ]; then
      log "  retry cleanup: killing $(echo "${SHIM_PIDS}" | wc -l) remaining containerd-shim"
      echo "${SHIM_PIDS}" | xargs -r kill -9 2>/dev/null || true
    fi
    restart_containerd_clean
    start_dockerd_direct
    wait_for_dockerd_api "runtime cleanup retry" || exit 3
  else
    exit 3
  fi
fi

# ─── Phase 5: verify dockerd has proxy env ───────────────────────────
log "Phase 5: verify dockerd has HTTP_PROXY in env"
DOCKERD_PID=$(pgrep -x dockerd | head -1)
if [ -n "${DOCKERD_PID}" ] && \
   grep -aq "HTTP_PROXY=" "/proc/${DOCKERD_PID}/environ" 2>/dev/null; then
  log "  [OK] dockerd PID=${DOCKERD_PID} has HTTP_PROXY in env"
else
  log "  [WARN] dockerd env doesn't contain HTTP_PROXY"
  log "    This is OK if you only use per-user ~/.docker/config.json proxies"
  log "    (it auto-injects HTTP_PROXY into build containers at build-time)."
  log "    But base-image pulls in 'FROM ...' will NOT go through proxy."
fi

# ─── Phase 5.5: pre-build proxied base images ────────────────────────
# CRITICAL: apt on Ubuntu 24.04 does NOT honor HTTP_PROXY env var.
# It only reads /etc/apt/apt.conf.d/*. Even with all proxy ENV/dropin
# correctly set, `apt-get update` inside the build STILL hits archive.
# ubuntu.com directly and times out.
#
# We fix this by pre-building shadow base images that bake in
# /etc/apt/apt.conf.d/95proxies, using the SAME tag as upstream so all
# 1377 seta_env Dockerfiles inherit it via `FROM <tag>`.
#
# This mirrors exactly what seta_env/0_my/Dockerfile does manually,
# but for ALL Dockerfiles at once and without modifying any of them.
log "Phase 5.5: pre-build shadow base images with apt proxy"
PREBUILD="$(dirname "$0")/prebuild_proxied_base_images.sh"
if [ -x "${PREBUILD}" ]; then
  PROXY_URL="${PROXY_URL}" NO_PROXY_LIST="${NO_PROXY_LIST}" \
    bash "${PREBUILD}" 2>&1 | sed 's/^/  /' || \
    log "  [WARN] prebuild step had failures; verify in Phase 6"
else
  log "  [SKIP] ${PREBUILD} not found (apt proxy injection won't work for unmodified Dockerfiles)"
fi

# ─── Phase 6: verify seta_env/0 build ────────────────────────────────
if [ "${SKIP_VERIFY}" != "1" ]; then
  log "Phase 6: verify seta_env/0 build"
  if [ -f "${SETA}/docker-compose.yaml" ]; then
    export T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME=seta_proxy_test:latest
    export T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME=seta_proxy_test
    export T_BENCH_TEST_DIR=/tmp/_t_test
    export T_BENCH_TASK_LOGS_PATH=/tmp/_t_logs
    export T_BENCH_TASK_AGENT_LOGS_PATH=/tmp/_t_agent_logs
    export T_BENCH_CONTAINER_LOGS_PATH=/var/log/_t
    export T_BENCH_CONTAINER_AGENT_LOGS_PATH=/var/log/_t_agent
    mkdir -p "$T_BENCH_TASK_LOGS_PATH" "$T_BENCH_TASK_AGENT_LOGS_PATH"

    set -a; . /etc/seta_build_proxy.env; set +a

    BUILD_TMP=$(mktemp)
    if DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 \
       timeout 600 docker compose -p seta_proxy_test \
         -f "${SETA}/docker-compose.yaml" build > "${BUILD_TMP}" 2>&1; then
      log "  [OK] seta_env/0 build succeeded"
      tail -3 "${BUILD_TMP}" | sed 's/^/    /'
      docker compose -p seta_proxy_test -f "${SETA}/docker-compose.yaml" \
        down 2>/dev/null || true
      docker rmi seta_proxy_test:latest 2>/dev/null || true
    else
      log "  [FAIL] seta_env/0 build failed; last 20 lines:"
      tail -20 "${BUILD_TMP}" | sed 's/^/    /'
      log "  Diagnostic checklist:"
      log "    1) systemctl show docker -p Environment | grep -i proxy"
      log "    2) cat /root/.docker/config.json"
      log "    3) docker pull ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624"
      log "    4) curl -x ${PROXY_URL} -I https://ghcr.io"
    fi
    rm -f "${BUILD_TMP}"
  else
    log "  [SKIP] ${SETA}/docker-compose.yaml not found"
  fi
else
  log "Phase 6: SKIP_VERIFY=1, skipping build test"
fi

# ─── Phase 7: restart docker-watchdog ────────────────────────────────
log "Phase 7: restart docker-watchdog"
if [ "${WATCHDOG_PRESENT}" = "1" ] && [ "${START_WATCHDOG}" = "1" ]; then
  if [ "${WATCHDOG_WAS_RUNNING}" = "1" ]; then
    if systemctl start docker-watchdog; then
      sleep 2
      if systemctl is-active --quiet docker-watchdog; then
        log "  [OK] docker-watchdog active"
      else
        log "  [WARN] docker-watchdog started but not active; check: journalctl -u docker-watchdog -n 50"
      fi
    else
      log "  [WARN] failed to start docker-watchdog; check unit file syntax"
    fi
  else
    log "  [SKIP] watchdog was not running before; not auto-starting"
    log "         to enable: sudo systemctl enable --now docker-watchdog"
  fi
elif [ "${START_WATCHDOG}" = "0" ]; then
  log "  [SKIP] START_WATCHDOG=0; start manually with: sudo systemctl start docker-watchdog"
elif [ "${WATCHDOG_PRESENT}" != "1" ]; then
  log "  [SKIP] docker-watchdog.service is not installed"
  log "         install with: sudo cp $(dirname "$0")/docker-watchdog.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now docker-watchdog"
fi

hr
log "=== Done ==="
hr
cat <<EOF

Next steps on this CPU worker:

  1) Source proxy env then start pool_server:
       set -a; . /etc/seta_build_proxy.env; set +a
       cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL
       bash terminal-rl/remote/run_pool_server_pu_v2.sh

  2) On GPU worker, set WORKER_URLS and start training:
       export WORKER_URLS="http://$(hostname -I | awk '{print $1}'):18081"
       bash terminal-rl/terminal-rl_qwen3-8b_pu.sh

  3) Watchdog status:
       sudo systemctl status docker-watchdog
       sudo journalctl -u docker-watchdog -f

  4) Revert (if needed):
       sudo systemctl stop docker-watchdog
       sudo rm /etc/systemd/system/docker.service.d/http-proxy.conf
       sudo rm /etc/systemd/system/docker-watchdog.service.d/http-proxy.conf
       sudo rm /etc/seta_build_proxy.env
       sudo rm /root/.docker/config.json  # and other users
       sudo systemctl daemon-reload && sudo systemctl restart docker

EOF
