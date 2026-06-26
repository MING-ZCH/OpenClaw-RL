#!/usr/bin/env bash
# Docker worker doctor for terminal-rl CPU workers.
#
# Modes:
#   diagnose     Read-only diagnosis. Safe during training.
#   soft-repair  Conservative repair: restart wedged Docker if needed and prune
#                stopped/dangling Docker state. Does not rewrite proxy config.
#   full-repair  Full recovery path: calls fix_dockerd_and_proxy.sh, then
#                runs diagnose again.
#
# Typical usage on a CPU worker:
#   bash terminal-rl/remote/docker_worker_doctor.sh diagnose \
#     --train-log /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/runs/<run>/logs/train.log
#
#   sudo env DOCKER_DATA_ROOT=/data \
#     PROXY_URL=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128 \
#     bash terminal-rl/remote/docker_worker_doctor.sh full-repair \
#     --train-log /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/runs/<run>/logs/train.log

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

MODE="diagnose"
if [ "${1:-}" != "" ] && [[ "${1:-}" != --* ]]; then
  MODE="$1"
  shift
fi

TRAIN_LOG="${TRAIN_LOG:-}"
OUT_DIR="${OUT_DIR:-}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
POOL_HOST="${POOL_HOST:-127.0.0.1}"
POOL_PORT="${POOL_PORT:-18081}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-${DOCKER_ROOT:-/data}}"
DOCKER_TIMEOUT="${DOCKER_TIMEOUT:-10}"
RUN_HEAVY="${RUN_HEAVY:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

usage() {
  cat <<EOF
Usage:
  bash $0 [diagnose|soft-repair|full-repair] [options]

Options:
  --train-log PATH     GPU train.log to parse for /reset, /evaluate, exit codes.
  --out-dir PATH       Output directory for reports.
  --pool-host HOST     Pool server host for local probes. Default: ${POOL_HOST}
  --pool-port PORT     Pool server port. Default: ${POOL_PORT}
  --docker-root PATH   Docker data root. Default: ${DOCKER_DATA_ROOT}
  --run-heavy          Enable heavier Docker stats such as docker system df.
  --skip-verify        In full-repair, skip seta_env build verification.
  -h, --help           Show this help.

Environment:
  DOCKER_DATA_ROOT, DOCKER_ROOT, PROXY_URL, NO_PROXY_LIST, SKIP_VERIFY,
  RUN_HEAVY, POOL_HOST, POOL_PORT, OUT_DIR, TRAIN_LOG.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --train-log)
      TRAIN_LOG="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --pool-host)
      POOL_HOST="${2:-}"
      shift 2
      ;;
    --pool-port)
      POOL_PORT="${2:-}"
      shift 2
      ;;
    --docker-root)
      DOCKER_DATA_ROOT="${2:-}"
      shift 2
      ;;
    --run-heavy)
      RUN_HEAVY=1
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${MODE}" in
  diagnose|soft-repair|full-repair) ;;
  *)
    echo "[ERROR] Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

TS="$(date '+%Y%m%d_%H%M%S')"
if [ -z "${OUT_DIR}" ]; then
  if [ -n "${RUN_DIR:-}" ]; then
    OUT_DIR="${RUN_DIR}/diagnostics/docker_worker_doctor_${TS}"
  elif [ -n "${RUN_ID:-}" ]; then
    OUT_DIR="${RUNS_ROOT}/${RUN_ID}/diagnostics/docker_worker_doctor_${TS}"
  elif [[ -n "${TRAIN_LOG}" && "${TRAIN_LOG}" == "${RUNS_ROOT}/"*"/logs/train.log" ]]; then
    TRAIN_RUN_DIR="$(cd -- "$(dirname -- "${TRAIN_LOG}")/.." 2>/dev/null && pwd -P || true)"
    if [ -n "${TRAIN_RUN_DIR}" ]; then
      OUT_DIR="${TRAIN_RUN_DIR}/diagnostics/docker_worker_doctor_${TS}"
    else
      OUT_DIR="${RUNS_ROOT}/diagnostics/docker_worker_doctor_${TS}"
    fi
  else
    OUT_DIR="${RUNS_ROOT}/diagnostics/docker_worker_doctor_${TS}"
  fi
fi
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/SUMMARY.md"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${SUMMARY}"
}

section() {
  {
    echo
    echo "## $*"
    echo
  } >> "${SUMMARY}"
}

require_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] ${MODE} must run as root. Re-run with sudo." >&2
    exit 1
  fi
}

docker_ready() {
  timeout "${DOCKER_TIMEOUT}" docker info >/dev/null 2>&1
}

count_pattern() {
  local pattern="$1"
  local file="$2"
  if [ -f "${file}" ]; then
    grep -c -E "${pattern}" "${file}" 2>/dev/null || true
  else
    echo 0
  fi
}

write_cmd() {
  local title="$1"
  local file="$2"
  shift 2
  {
    echo "### ${title}"
    echo "\$ $*"
    echo
    "$@"
  } > "${file}" 2>&1
}

write_shell() {
  local title="$1"
  local file="$2"
  local script="$3"
  {
    echo "### ${title}"
    echo "\$ ${script}"
    echo
    bash -lc "${script}"
  } > "${file}" 2>&1
}

detect_docker_root() {
  local root=""
  if docker_ready; then
    root="$(timeout "${DOCKER_TIMEOUT}" docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
  fi
  if [ -n "${root}" ]; then
    DOCKER_DATA_ROOT="${root}"
  fi
}

diagnose_train_log() {
  section "GPU Train Log Signals"
  if [ -z "${TRAIN_LOG}" ] || [ ! -f "${TRAIN_LOG}" ]; then
    log "train_log: not provided or not found; skipping GPU log parsing."
    return
  fi

  local out="${OUT_DIR}/train_log_summary.txt"
  local reset_500 evaluate_500 heartbeat_500 errno11 gen_failed eval_failed
  local exit17 exit125 exit2 exit1 no_space eval_timeout dropped_groups grad_zero
  reset_500="$(count_pattern "500 Internal Server Error.*(/reset|url=.*reset)" "${TRAIN_LOG}")"
  evaluate_500="$(count_pattern "500 Internal Server Error.*(/evaluate|url=.*evaluate)" "${TRAIN_LOG}")"
  heartbeat_500="$(count_pattern "500 Internal Server Error.*(/heartbeat|url=.*heartbeat)" "${TRAIN_LOG}")"
  errno11="$(count_pattern "Resource temporarily unavailable|Errno 11" "${TRAIN_LOG}")"
  gen_failed="$(count_pattern "Generate failed" "${TRAIN_LOG}")"
  eval_failed="$(count_pattern "Evaluation failed, marking FAILED" "${TRAIN_LOG}")"
  exit17="$(count_pattern "exit status 17" "${TRAIN_LOG}")"
  exit125="$(count_pattern "exit status 125" "${TRAIN_LOG}")"
  exit2="$(count_pattern "exit status 2" "${TRAIN_LOG}")"
  exit1="$(count_pattern "exit status 1" "${TRAIN_LOG}")"
  no_space="$(count_pattern "no space left on device|No space left on device" "${TRAIN_LOG}")"
  eval_timeout="$(count_pattern "Evaluation tests timed out|timed out for task" "${TRAIN_LOG}")"
  dropped_groups="$(count_pattern "Dropped constant-reward groups" "${TRAIN_LOG}")"
  grad_zero="$(count_pattern "train/grad_norm': 0\\.0|train/grad_norm: 0\\.0" "${TRAIN_LOG}")"

  {
    echo "train_log: ${TRAIN_LOG}"
    echo "lines: $(wc -l < "${TRAIN_LOG}" 2>/dev/null || echo unknown)"
    echo
    printf "%-34s %s\n" "reset_500" "${reset_500}"
    printf "%-34s %s\n" "evaluate_500" "${evaluate_500}"
    printf "%-34s %s\n" "heartbeat_500" "${heartbeat_500}"
    printf "%-34s %s\n" "errno11_resource_unavailable" "${errno11}"
    printf "%-34s %s\n" "generate_failed" "${gen_failed}"
    printf "%-34s %s\n" "evaluation_failed" "${eval_failed}"
    printf "%-34s %s\n" "docker_exit_17" "${exit17}"
    printf "%-34s %s\n" "docker_exit_125" "${exit125}"
    printf "%-34s %s\n" "docker_exit_2" "${exit2}"
    printf "%-34s %s\n" "docker_exit_1" "${exit1}"
    printf "%-34s %s\n" "no_space_left" "${no_space}"
    printf "%-34s %s\n" "evaluation_timeout" "${eval_timeout}"
    printf "%-34s %s\n" "constant_reward_group_drops" "${dropped_groups}"
    printf "%-34s %s\n" "zero_grad_steps" "${grad_zero}"
    echo
    echo "=== exit-code task ranking ==="
    for code in 17 125 2 1; do
      echo
      echo "--- exit status ${code} ---"
      grep "exit status ${code}" "${TRAIN_LOG}" 2>/dev/null \
        | grep -oE 'seta_env/[0-9]+' \
        | sort 2>/dev/null | uniq -c | sort -rn 2>/dev/null | head -30 \
        || true
    done
    echo
    echo "=== first 40 docker/server failures ==="
    grep -n -E "500 Internal Server Error|exit status|Resource temporarily unavailable|Generate failed|Evaluation failed" "${TRAIN_LOG}" 2>/dev/null \
      | head -40 || true
    echo
    echo "=== last 80 docker/server failures ==="
    grep -n -E "500 Internal Server Error|exit status|Resource temporarily unavailable|Generate failed|Evaluation failed" "${TRAIN_LOG}" 2>/dev/null \
      | tail -80 || true
  } > "${out}"

  {
    echo "- Parsed train log: \`${TRAIN_LOG}\`"
    echo "- \`/reset\` 500: ${reset_500}"
    echo "- \`/evaluate\` 500: ${evaluate_500}"
    echo "- \`/heartbeat\` 500: ${heartbeat_500}"
    echo "- \`Errno 11 / Resource temporarily unavailable\`: ${errno11}"
    echo "- Docker exit statuses: 17=${exit17}, 125=${exit125}, 2=${exit2}, 1=${exit1}"
    echo "- Generate failed: ${gen_failed}; evaluation failed: ${eval_failed}"
    echo "- Constant reward group drops: ${dropped_groups}; zero-grad steps: ${grad_zero}"
    echo "- Details: \`${out}\`"
  } >> "${SUMMARY}"

  TRAIN_RESET_500="${reset_500}"
  TRAIN_EVAL_500="${evaluate_500}"
  TRAIN_HEARTBEAT_500="${heartbeat_500}"
  TRAIN_ERRNO11="${errno11}"
  TRAIN_EXIT17="${exit17}"
  TRAIN_EXIT125="${exit125}"
  TRAIN_EXIT2="${exit2}"
  TRAIN_EXIT1="${exit1}"
  TRAIN_NOSPACE="${no_space}"
  TRAIN_EVAL_TIMEOUT="${eval_timeout}"
}

diagnose_host() {
  section "Host And Resource Snapshot"
  write_shell "host snapshot" "${OUT_DIR}/host.txt" '
set -u
echo "date: $(date --iso-8601=seconds 2>/dev/null || date)"
echo "hostname: $(hostname -f 2>/dev/null || hostname)"
echo "user: $(id)"
echo
echo "=== ip ==="
hostname -I 2>/dev/null || true
echo
echo "=== uname ==="
uname -a
echo
echo "=== uptime ==="
uptime
echo
echo "=== load/mem ==="
free -h 2>/dev/null || true
echo
echo "=== process/thread pressure ==="
echo -n "processes: "; ps -e --no-headers 2>/dev/null | wc -l
echo -n "threads: "; ps -eLf --no-headers 2>/dev/null | wc -l
echo -n "threads-max: "; cat /proc/sys/kernel/threads-max 2>/dev/null || true
echo -n "pid_max: "; cat /proc/sys/kernel/pid_max 2>/dev/null || true
echo -n "file-nr: "; cat /proc/sys/fs/file-nr 2>/dev/null || true
echo
echo "=== ulimit ==="
ulimit -a
'
  log "host snapshot -> ${OUT_DIR}/host.txt"
}

diagnose_docker() {
  section "Docker State"
  detect_docker_root
  local ready="no"
  if docker_ready; then
    ready="yes"
  fi
  echo "- Docker API responsive: ${ready}" >> "${SUMMARY}"
  echo "- Docker data root: \`${DOCKER_DATA_ROOT}\`" >> "${SUMMARY}"

  {
    echo "Docker API responsive: ${ready}"
    echo "Docker data root: ${DOCKER_DATA_ROOT}"
    echo
    echo "=== docker process ==="
    pgrep -a -x dockerd 2>/dev/null || true
    pgrep -a -x containerd 2>/dev/null || true
    echo
    echo "=== docker fd/thread count ==="
    pid="$(pgrep -x dockerd 2>/dev/null | head -1 || true)"
    if [ -n "${pid}" ]; then
      echo "dockerd_pid=${pid}"
      echo -n "dockerd_fds="
      ls "/proc/${pid}/fd" 2>/dev/null | wc -l || true
      echo -n "dockerd_threads="
      grep '^Threads:' "/proc/${pid}/status" 2>/dev/null || true
    else
      echo "dockerd_pid=<none>"
    fi
    echo
    echo "=== docker version ==="
    timeout "${DOCKER_TIMEOUT}" docker version 2>&1 || true
    echo
    echo "=== docker compose version ==="
    timeout "${DOCKER_TIMEOUT}" docker compose version 2>&1 || true
    echo
    echo "=== docker info summary ==="
    timeout "${DOCKER_TIMEOUT}" docker info 2>&1 \
      | grep -E "Containers:|Running:|Paused:|Stopped:|Images:|Server Version|Docker Root Dir|Storage Driver|Cgroup|Default Runtime|Logging Driver|Live Restore|HTTP Proxy|HTTPS Proxy|No Proxy|Registry Mirrors|Insecure Registries" \
      || true
    echo
    echo "=== data-root disk ==="
    df -h "${DOCKER_DATA_ROOT}" 2>&1 || true
    df -ih "${DOCKER_DATA_ROOT}" 2>&1 || true
    findmnt -T "${DOCKER_DATA_ROOT}" 2>&1 || true
    echo
    echo "=== docker object counts ==="
    if [ "${ready}" = "yes" ]; then
      echo -n "running_containers="; timeout "${DOCKER_TIMEOUT}" docker ps -q 2>/dev/null | wc -l
      echo -n "all_containers="; timeout "${DOCKER_TIMEOUT}" docker ps -aq 2>/dev/null | wc -l
      echo -n "images="; timeout "${DOCKER_TIMEOUT}" docker images -q 2>/dev/null | sort -u | wc -l
      echo -n "networks="; timeout "${DOCKER_TIMEOUT}" docker network ls -q 2>/dev/null | wc -l
      echo -n "volumes="; timeout "${DOCKER_TIMEOUT}" docker volume ls -q 2>/dev/null | wc -l
      echo
      echo "=== top running containers ==="
      timeout "${DOCKER_TIMEOUT}" docker ps --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}" 2>&1 | head -40 || true
      echo
      if [ "${RUN_HEAVY}" = "1" ]; then
        echo "=== docker system df ==="
        timeout 60 docker system df 2>&1 || true
      else
        echo "Skipping docker system df. Set RUN_HEAVY=1 or pass --run-heavy to enable."
      fi
    else
      echo "Docker not responsive; object counts skipped."
    fi
  } > "${OUT_DIR}/docker_state.txt"
  log "docker state -> ${OUT_DIR}/docker_state.txt"
}

diagnose_systemd() {
  section "Systemd And Journals"
  write_shell "systemd and journal snapshot" "${OUT_DIR}/systemd_journal.txt" '
set -u
for unit in docker docker.socket containerd docker-watchdog; do
  echo
  echo "=== systemctl status ${unit} ==="
  timeout 10 systemctl status "${unit}" --no-pager 2>&1 || true
done
echo
echo "=== docker journal tail ==="
timeout 20 journalctl -u docker -n 160 --no-pager 2>&1 || true
echo
echo "=== containerd journal tail ==="
timeout 20 journalctl -u containerd -n 80 --no-pager 2>&1 || true
echo
echo "=== docker-watchdog journal tail ==="
timeout 20 journalctl -u docker-watchdog -n 160 --no-pager 2>&1 || true
'
  log "systemd/journal -> ${OUT_DIR}/systemd_journal.txt"
}

diagnose_pool() {
  section "Pool Server State"
  {
    echo "pool: http://${POOL_HOST}:${POOL_PORT}"
    echo
    echo "=== pool processes ==="
    pgrep -a -f "remote.pool_server|pool_server.py|run_pool_server" 2>&1 || true
    echo
    echo "=== listening ports ==="
    ss -tlnp 2>/dev/null | grep -E "(:${POOL_PORT} |dockerd|containerd)" || true
    echo
    echo "=== /healthz ==="
    curl --noproxy "*" -fsS --max-time 5 "http://${POOL_HOST}:${POOL_PORT}/healthz" 2>&1 || true
    echo
    echo "=== /status ==="
    status_tmp="$(mktemp)"
    if curl --noproxy "*" -fsS --max-time 5 "http://${POOL_HOST}:${POOL_PORT}/status" > "${status_tmp}" 2>&1; then
      python3 -m json.tool "${status_tmp}" 2>/dev/null | head -120 || cat "${status_tmp}"
    else
      cat "${status_tmp}"
    fi
    rm -f "${status_tmp}"
  } > "${OUT_DIR}/pool_server.txt"
  log "pool server -> ${OUT_DIR}/pool_server.txt"
}

write_recommendations() {
  section "Diagnosis And Next Actions"
  local rec="${OUT_DIR}/recommendations.txt"
  {
    echo "Recommended interpretation:"
    echo
    if ! docker_ready; then
      echo "- Docker API is not responsive. This matches wedged dockerd/stale socket symptoms."
      echo "  Run: sudo env DOCKER_DATA_ROOT=${DOCKER_DATA_ROOT} bash terminal-rl/remote/docker_worker_doctor.sh full-repair"
    fi
    if [ "${TRAIN_ERRNO11:-0}" != "0" ]; then
      echo "- Errno 11 / Resource temporarily unavailable appears in the train log."
      echo "  This usually means CPU worker resource pressure: too many subprocesses,"
      echo "  containerd-shim processes, file descriptors, threads, or concurrent builds."
      echo "  Immediate action: full-repair, then restart pool_server with lower Docker concurrency."
    fi
    if [ "${TRAIN_EXIT125:-0}" != "0" ]; then
      echo "- Docker exit status 125 appears. This is usually daemon/container creation failure,"
      echo "  not model behavior. Restart Docker/containerd and clear stale state."
    fi
    if [ "${TRAIN_EXIT17:-0}" != "0" ] || [ "${TRAIN_EXIT2:-0}" != "0" ]; then
      echo "- Docker compose build exit 17/2 appears. Likely build/proxy/cache/resource pressure."
      echo "  Run full-repair to rewrite proxy config and prebuild proxied base images."
    fi
    if [ "${TRAIN_NOSPACE:-0}" != "0" ]; then
      echo "- no space left on device appears. Run:"
      echo "  sudo env DOCKER_DATA_ROOT=${DOCKER_DATA_ROOT} AGGRESSIVE=1 bash terminal-rl/remote/fix_docker_overlay2_no_space.sh"
    fi
    if [ "${TRAIN_EVAL_TIMEOUT:-0}" != "0" ]; then
      echo "- Evaluation timeouts appear. After Docker is stable, inspect slow tasks and consider"
      echo "  lowering concurrent runs or increasing task evaluation timeout."
    fi
    if [ "${TRAIN_RESET_500:-0}" != "0" ] || [ "${TRAIN_EVAL_500:-0}" != "0" ]; then
      echo "- Server 500s on /reset or /evaluate are environment-service failures."
      echo "  They directly produce failed trajectories, constant reward groups, and zero-gradient steps."
    fi
    echo
    echo "Standard recovery sequence on CPU worker:"
    echo "  cd ${REPO_ROOT}"
    echo "  sudo env DOCKER_DATA_ROOT=${DOCKER_DATA_ROOT} \\"
    echo "    PROXY_URL=\${PROXY_URL:-http://httpproxy-headless.kubebrain.svc.pjlab.local:3128} \\"
    echo "    bash terminal-rl/remote/docker_worker_doctor.sh full-repair \\"
    if [ -n "${TRAIN_LOG}" ]; then
      echo "    --train-log ${TRAIN_LOG}"
    else
      echo "    --train-log /path/to/gpu/train.log"
    fi
    echo
    echo "Then restart pool_server:"
    echo "  cd ${REPO_ROOT}"
    echo "  nohup bash terminal-rl/remote/run_pool_server_pu_v2.sh > /tmp/cpu_pool.log 2>&1 &"
    echo "  curl --noproxy '*' http://127.0.0.1:${POOL_PORT}/healthz"
  } > "${rec}"

  cat "${rec}" >> "${SUMMARY}"
  log "recommendations -> ${rec}"
}

run_diagnose() {
  : > "${SUMMARY}"
  echo "# Terminal-RL Docker Worker Doctor" >> "${SUMMARY}"
  echo >> "${SUMMARY}"
  echo "- generated_at: $(date '+%F %T %Z')" >> "${SUMMARY}"
  echo "- mode: ${MODE}" >> "${SUMMARY}"
  echo "- host: $(hostname -f 2>/dev/null || hostname)" >> "${SUMMARY}"
  echo "- out_dir: ${OUT_DIR}" >> "${SUMMARY}"
  echo "- docker_data_root: ${DOCKER_DATA_ROOT}" >> "${SUMMARY}"
  echo "- train_log: ${TRAIN_LOG:-<none>}" >> "${SUMMARY}"

  diagnose_train_log
  diagnose_host
  diagnose_docker
  diagnose_systemd
  diagnose_pool
  write_recommendations

  tar -C "$(dirname "${OUT_DIR}")" -czf "${OUT_DIR}.tar.gz" "$(basename "${OUT_DIR}")" 2>/dev/null || true
  echo
  echo "DONE"
  echo "Summary: ${SUMMARY}"
  echo "Bundle:  ${OUT_DIR}.tar.gz"
}

soft_repair() {
  require_root
  log "soft-repair: starting"
  if docker_ready; then
    log "Docker API is responsive; running conservative cleanup."
    DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" bash "${SCRIPT_DIR}/cleanup_docker_cache.sh" \
      > "${OUT_DIR}/soft_repair_cleanup.log" 2>&1 || true
  else
    log "Docker API is not responsive; running force restart."
    DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" bash "${SCRIPT_DIR}/restart_docker_force.sh" \
      > "${OUT_DIR}/soft_repair_restart.log" 2>&1 || true
  fi

  if systemctl list-unit-files docker-watchdog.service >/dev/null 2>&1; then
    timeout 20 systemctl restart docker-watchdog >/dev/null 2>&1 || true
  fi
  MODE="diagnose"
  run_diagnose
}

full_repair() {
  require_root
  log "full-repair: calling fix_dockerd_and_proxy.sh"
  local repair_status=0
  SKIP_VERIFY="${SKIP_VERIFY}" DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}" \
    bash "${SCRIPT_DIR}/fix_dockerd_and_proxy.sh" \
    > "${OUT_DIR}/full_repair.log" 2>&1 || repair_status=$?
  MODE="diagnose"
  run_diagnose
  if [ "${repair_status}" != "0" ]; then
    echo "[ERROR] full-repair failed with status ${repair_status}; see ${OUT_DIR}/full_repair.log" >&2
    exit "${repair_status}"
  fi
}

case "${MODE}" in
  diagnose)
    run_diagnose
    ;;
  soft-repair)
    soft_repair
    ;;
  full-repair)
    full_repair
    ;;
esac
