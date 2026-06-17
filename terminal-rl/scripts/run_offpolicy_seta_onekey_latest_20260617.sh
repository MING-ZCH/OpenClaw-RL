#!/usr/bin/env bash
# One-key launcher for SETA + terminal-rl off-policy experiments.
#
# Usage:
#   bash terminal-rl/scripts/run_offpolicy_seta_onekey_latest_20260617.sh
#   bash terminal-rl/scripts/run_offpolicy_seta_onekey_latest_20260617.sh dapo
#   NUM_GPUS=2 bash terminal-rl/scripts/run_offpolicy_seta_onekey_latest_20260617.sh spear
#
# Modes:
#   spear     SPEAR/SIL replay on top of off-policy DAPO. Default.
#   dapo      Off-policy DAPO replay-buffer baseline.
#   per       Prioritized replay with reward_dev priority.
#   topr      TOPR sequence-level importance weighting.
#   all3      DAPO admission gate + PER + TOPR combined ablation.
#   baseline  No off-policy extension.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

MODE="${1:-${OFFPOLICY_MODE:-spear}}"
case "${MODE}" in
  baseline) MODE="none" ;;
  none|dapo|per|topr|spear|all3) ;;
  *)
    echo "[ERROR] Unknown mode='${MODE}'. Use: spear|dapo|per|topr|all3|baseline" >&2
    exit 1
    ;;
esac

DEFAULT_WORKER_URLS="${OFFPOLICY_DEFAULT_WORKER_URLS:-}"
export WORKER_URLS="${WORKER_URLS:-${DEFAULT_WORKER_URLS}}"
export OFFPOLICY_MODE="${MODE}"
export OFFPOLICY_USE_INTEGRATED_SLIME="${OFFPOLICY_USE_INTEGRATED_SLIME:-1}"
export DATASET="${DATASET:-seta}"
export ALGO="${ALGO:-dapo}"
export SETA_SAFETY="${SETA_SAFETY:-clawsentry}"
export SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0.3}"
export MAX_TURN="${MAX_TURN:-10}"
export MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${REPO_ROOT}/terminal-rl/configs/rollout_qwen3_think.yaml}"

LIGHTRFT_PY312_BIN="${LIGHTRFT_PY312_BIN:-}"
if [[ -n "${LIGHTRFT_PY312_BIN}" && -d "${LIGHTRFT_PY312_BIN}" ]]; then
  export PATH="${LIGHTRFT_PY312_BIN}:${PATH}"
fi

gpu_query_available() {
  command -v nvidia-smi >/dev/null 2>&1
}

print_gpu_processes() {
  if ! gpu_query_available; then
    echo "[gpu] nvidia-smi not found; skip GPU process listing"
    return 0
  fi
  echo "[gpu] current GPU memory:"
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits || true
  echo "[gpu] current GPU compute processes:"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
}

kill_gpu_processes_if_requested() {
  if [[ "${DRY_RUN:-0}" == "1" || "${ONEKEY_KILL_GPU_PROCS:-0}" != "1" ]]; then
    return 0
  fi
  if ! gpu_query_available; then
    echo "[WARN] ONEKEY_KILL_GPU_PROCS=1 but nvidia-smi is unavailable; skip GPU cleanup" >&2
    return 0
  fi

  echo "[gpu-clean] stopping local Ray/SGLang first..."
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 sglang >/dev/null 2>&1 || true
  pkill -9 ray >/dev/null 2>&1 || true
  pkill -9 -f "terminal-rl.router_server" >/dev/null 2>&1 || true
  sleep 2

  mapfile -t gpu_pids < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | awk 'NF {print $1}' \
      | sort -u
  )
  if [[ "${#gpu_pids[@]}" -eq 0 ]]; then
    echo "[gpu-clean] no GPU compute process remains"
    return 0
  fi

  echo "[gpu-clean] killing GPU compute PIDs: ${gpu_pids[*]}"
  for pid in "${gpu_pids[@]}"; do
    kill -TERM "${pid}" >/dev/null 2>&1 || true
  done
  sleep 5
  for pid in "${gpu_pids[@]}"; do
    kill -KILL "${pid}" >/dev/null 2>&1 || true
  done
  sleep 3
  print_gpu_processes
}

require_gpu_memory_if_available() {
  if [[ "${DRY_RUN:-0}" == "1" || "${ONEKEY_GPU_PREFLIGHT:-1}" != "1" ]]; then
    return 0
  fi
  if ! gpu_query_available; then
    echo "[WARN] nvidia-smi not found; skip GPU memory preflight" >&2
    return 0
  fi

  local min_free_gb min_free_mib check_gpus free_mib idx bad
  min_free_gb="${ONEKEY_MIN_GPU_FREE_GB:-80}"
  min_free_mib=$(( min_free_gb * 1024 ))
  check_gpus="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)}"
  if [[ "${check_gpus}" -le 0 ]]; then
    return 0
  fi

  bad=0
  for ((idx = 0; idx < check_gpus; idx++)); do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${idx}" 2>/dev/null | awk 'NR==1 {print int($1)}')"
    if [[ -z "${free_mib}" ]]; then
      continue
    fi
    if (( free_mib < min_free_mib )); then
      echo "[ERROR] GPU ${idx} free memory ${free_mib} MiB < required ${min_free_mib} MiB" >&2
      bad=1
    fi
  done

  if [[ "${bad}" == "1" ]]; then
    print_gpu_processes >&2
    cat >&2 <<EOF
[ERROR] GPU memory preflight failed.
        This run needs a mostly idle training pod for Qwen3-8B actor/rollout.
        If this pod is dedicated to this experiment, rerun with:
          ONEKEY_KILL_GPU_PROCS=1 bash terminal-rl/scripts/run_offpolicy_seta_onekey_latest_20260617.sh ${MODE}
        Or lower ONEKEY_MIN_GPU_FREE_GB / choose a clean pod.
EOF
    exit 1
  fi
}

check_worker() {
  local base_url="$1"
  local health_code ready_code
  health_code="$(curl -sS --noproxy '*' --max-time 5 -o /tmp/openclaw_worker_health.$$ -w '%{http_code}' "${base_url}/healthz" || true)"
  ready_code="$(curl -sS --noproxy '*' --max-time 10 -o /tmp/openclaw_worker_ready.$$ -w '%{http_code}' "${base_url}/readyz" || true)"
  rm -f /tmp/openclaw_worker_health.$$ /tmp/openclaw_worker_ready.$$
  if [[ "${health_code}" =~ ^2[0-9][0-9]$ && "${ready_code}" =~ ^2[0-9][0-9]$ ]]; then
    echo "[worker] ${base_url} health=${health_code} ready=${ready_code}"
    return 0
  fi
  echo "[WARN] worker ${base_url} not fully ready: health=${health_code:-000} ready=${ready_code:-000}" >&2
  return 1
}

echo "[onekey] repo=${REPO_ROOT}"
echo "[onekey] mode=${MODE} dataset=${DATASET} algo=${ALGO}"
echo "[onekey] workers=${WORKER_URLS}"

kill_gpu_processes_if_requested
require_gpu_memory_if_available

worker_ok=0
IFS=',' read -r -a workers <<< "${WORKER_URLS}"
for worker in "${workers[@]}"; do
  [[ -z "${worker}" ]] && continue
  if check_worker "${worker}"; then
    worker_ok=$((worker_ok + 1))
  fi
done

if [[ "${REQUIRE_WORKER_READY:-1}" == "1" && -n "${WORKER_URLS}" && "${worker_ok}" -eq 0 ]]; then
  echo "[ERROR] no ready Docker worker found. Set REQUIRE_WORKER_READY=0 to bypass preflight." >&2
  exit 1
fi

exec bash "${SCRIPT_DIR}/run_offpolicy_seta_latest_20260617.sh" "${MODE}"
