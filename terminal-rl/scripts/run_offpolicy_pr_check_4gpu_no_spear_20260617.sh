#!/usr/bin/env bash
# Local 4-GPU PR validation runner for off-policy replay-buffer modes.
#
# Scope: baseline, dapo, per, topr, all3. SPEAR is intentionally skipped here
# because it has already been validated separately in the current workflow.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

DEFAULT_MODES=(baseline dapo per topr all3)
if [[ "$#" -gt 0 ]]; then
  MODES=("$@")
elif [[ -n "${CHECK_MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES=(${CHECK_MODES})
else
  MODES=("${DEFAULT_MODES[@]}")
fi

for mode in "${MODES[@]}"; do
  if [[ "${mode}" == "spear" ]]; then
    echo "[ERROR] spear is intentionally skipped by this runner. Run the spear launcher separately if needed." >&2
    exit 2
  fi
done

CHECK_ROOT="${CHECK_ROOT:-${REPO_ROOT}/runs/offpolicy_pr_check_4gpu_no_spear_${TIMESTAMP}}"
CHECK_RUNS_ROOT="${CHECK_ROOT}/runs"
CHECK_LOG_ROOT="${CHECK_ROOT}/launcher_logs"
SUMMARY_LOG="${CHECK_ROOT}/summary.log"
mkdir -p "${CHECK_RUNS_ROOT}" "${CHECK_LOG_ROOT}"

DEFAULT_WORKER_URLS="${OFFPOLICY_DEFAULT_WORKER_URLS:-}"

export WORKER_URLS="${WORKER_URLS:-${DEFAULT_WORKER_URLS}}"
export OFFPOLICY_USE_INTEGRATED_SLIME="${OFFPOLICY_USE_INTEGRATED_SLIME:-1}"
export ALGO="${ALGO:-dapo}"
export DATASET="${DATASET:-seta}"
export SETA_SAFETY="${SETA_SAFETY:-clawsentry}"
export SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0.3}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${REPO_ROOT}/terminal-rl/configs/rollout_qwen3_think.yaml}"
export MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-0}"

# Use the full 4-GPU pod: 2 GPUs for actor training and 2 GPUs for rollout.
export NUM_GPUS="${NUM_GPUS:-4}"
export ACTOR_GPUS="${ACTOR_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
export TP_SIZE="${TP_SIZE:-2}"

# Avoid DEBUG_MODE's tiny workload. These defaults are still smoke-sized but
# large enough to keep the 4-GPU pod busy during rollout/train phases.
export DEBUG_MODE="${DEBUG_MODE:-0}"
export MAX_TURN="${MAX_TURN:-5}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-16}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
export N_SAMPLES="${N_SAMPLES:-4}"
export DAPO_OVER_SAMPLING_BATCH_SIZE="${DAPO_OVER_SAMPLING_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
export TRAIN_ITERS_PER_ROLLOUT="${TRAIN_ITERS_PER_ROLLOUT:-2}"
export UPDATE_POLICY_VERSION_EVERY_TRAIN_ITER="${UPDATE_POLICY_VERSION_EVERY_TRAIN_ITER:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-16384}"

export RUNS_ROOT="${CHECK_RUNS_ROOT}"

# This runner is intended for a dedicated 4-GPU validation pod.  Sequential
# Qwen3-8B runs can leave short-lived Ray/SGLang GPU processes behind after a
# successful job exits; clean between modes so the next mode is not rejected by
# GPU-memory preflight or started against a stale Ray cluster.
export PR_CHECK_CLEAN_BETWEEN_MODES="${PR_CHECK_CLEAN_BETWEEN_MODES:-1}"
export PR_CHECK_INTER_MODE_SLEEP="${PR_CHECK_INTER_MODE_SLEEP:-10}"
export PR_CHECK_ONEKEY_KILL_GPU_PROCS="${PR_CHECK_ONEKEY_KILL_GPU_PROCS:-1}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${SUMMARY_LOG}"
}

check_worker() {
  local base_url="$1"
  local health_code ready_code
  health_code="$(curl -sS --noproxy '*' --max-time 5 -o /tmp/openclaw_prcheck_health.$$ -w '%{http_code}' "${base_url}/healthz" || true)"
  ready_code="$(curl -sS --noproxy '*' --max-time 10 -o /tmp/openclaw_prcheck_ready.$$ -w '%{http_code}' "${base_url}/readyz" || true)"
  rm -f /tmp/openclaw_prcheck_health.$$ /tmp/openclaw_prcheck_ready.$$
  log "worker ${base_url} health=${health_code:-000} ready=${ready_code:-000}"
}

start_gpu_monitor() {
  local mode="$1"
  local monitor_log="${CHECK_LOG_ROOT}/${mode}.gpu_util.log"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return 0
  fi
  (
    while true; do
      date '+%F %T'
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
      sleep "${GPU_MONITOR_INTERVAL:-30}"
    done
  ) > "${monitor_log}" 2>&1 &
  echo "$!"
}

stop_gpu_monitor() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
}

cleanup_train_stack() {
  if [[ "${PR_CHECK_CLEAN_BETWEEN_MODES}" != "1" || "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  log "cleanup local Ray/SGLang/router before next mode"
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 sglang >/dev/null 2>&1 || true
  pkill -9 ray >/dev/null 2>&1 || true
  pkill -9 -f "terminal-rl.router_server" >/dev/null 2>&1 || true
  sleep "${PR_CHECK_INTER_MODE_SLEEP}"
}

summarize_mode() {
  local mode="$1"
  local run_id="$2"
  local run_dir="${CHECK_RUNS_ROOT}/${run_id}"
  local gpu_log="${run_dir}/logs/mirror/gpu_run.log"
  log "summary mode=${mode} run_dir=${run_dir}"
  if [[ -f "${gpu_log}" ]]; then
    grep -E "Replay buffer enabled|SPEAR SIL buffer enabled|actor_train_end|train_steps|Ray job succeeded|Run failed|Traceback|RuntimeError|ERROR|importance_weight|staleness|per_is_weight|topr" "${gpu_log}" \
      | tail -120 \
      | tee -a "${SUMMARY_LOG}" || true
  else
    log "missing gpu log: ${gpu_log}"
  fi
}

cd "${REPO_ROOT}"

log "check_root=${CHECK_ROOT}"
log "modes=${MODES[*]}"
log "workers=${WORKER_URLS}"
if [[ -z "${WORKER_URLS}" ]]; then
  log "WARN no WORKER_URLS set; export WORKER_URLS before non-DRY_RUN terminal-rl validation"
fi
log "gpu_config num=${NUM_GPUS} actor=${ACTOR_GPUS} rollout=${ROLLOUT_GPUS} tp=${TP_SIZE} engine_tp=${ROLLOUT_NUM_GPUS_PER_ENGINE}"
log "workload max_turn=${MAX_TURN} num_rollout=${NUM_ROLLOUT} rollout_batch=${ROLLOUT_BATCH_SIZE} n_samples=${N_SAMPLES}"

IFS=',' read -r -a workers <<< "${WORKER_URLS}"
for worker in "${workers[@]}"; do
  [[ -z "${worker}" ]] && continue
  check_worker "${worker}"
done

overall_status=0
for mode in "${MODES[@]}"; do
  run_id="offpolicy_pr_check_4gpu_${mode}_${TIMESTAMP}"
  launcher_log="${CHECK_LOG_ROOT}/${mode}.launcher.log"
  gpu_monitor_pid=""

  cleanup_train_stack
  log "START mode=${mode} run_id=${run_id}"
  gpu_monitor_pid="$(start_gpu_monitor "${mode}")"

  set +e
  RUN_ID="${run_id}" RUN_NAME="${run_id}" \
    ONEKEY_KILL_GPU_PROCS="${PR_CHECK_ONEKEY_KILL_GPU_PROCS}" \
    bash "${SCRIPT_DIR}/run_offpolicy_seta_onekey_latest_20260617.sh" "${mode}" \
      > >(tee -a "${launcher_log}") \
      2> >(tee -a "${launcher_log}" >&2)
  status=$?
  set -e

  stop_gpu_monitor "${gpu_monitor_pid}"
  cleanup_train_stack
  log "END mode=${mode} status=${status}"
  summarize_mode "${mode}" "${run_id}"

  if [[ "${status}" -ne 0 ]]; then
    overall_status="${status}"
    if [[ "${CONTINUE_ON_FAIL:-1}" != "1" ]]; then
      break
    fi
  fi
done

log "DONE overall_status=${overall_status}"
log "logs=${CHECK_ROOT}"
exit "${overall_status}"
