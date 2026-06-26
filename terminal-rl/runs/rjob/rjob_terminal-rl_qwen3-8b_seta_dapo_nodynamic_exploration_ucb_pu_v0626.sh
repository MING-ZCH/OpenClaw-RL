#!/bin/bash

################################################################################
# Submit OpenClaw terminal-rl Qwen3-8B SETA exploration v0626 as a detached
# 1-node 8-GPU rjob task.
#
# The paired run script executes inside the rjob worker and launches the
# SETA-only DAPO nodynamic Agent57-lite exploration training job.
#
# v0626 changes relative to v0623:
#   - uses the risk-calibrated v0626 paired run script;
#   - charges narmodel_gpu as requested;
#   - uses a sudoku-related rjob name with an -expl suffix for queue filtering.
################################################################################

set -euo pipefail

WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCLAW_HOME="${OPENCLAW_HOME:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUN_SCRIPT="${RUN_SCRIPT:-${OPENCLAW_HOME}/terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_exploration_ucb_pu_v0626.sh}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y-%m-%d_%H%M%S')}"
NUM_GPUS="${NUM_GPUS:-8}"
RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${NUM_GPUS}gpu_seta_dapo_nodynamic_exploration_simhash_life_fp_ucb_v0626_riskcal_turninv_truncpenalty_dualadv_think_${RUN_TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${OPENCLAW_HOME}/runs}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_DIR="${WANDB_DIR:-${RUNS_ROOT}/${RUN_ID}/metrics/wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-terminal_rl}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-8b_seta_exploration_v0626}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [[ -n "${WANDB_KEY}" && -z "${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY="${WANDB_KEY}"
fi
RJOB_LOG_ROOT="${RJOB_LOG_ROOT:-${OPENCLAW_HOME}/runs/rjob_logs/terminal_rl_v0626}"

# rjob generates a Kubernetes label by concatenating job name and task name.
# Keep the default short enough that even duplicated as the task name it stays
# within the 63-char label limit.
RJOB_NAME="${RJOB_NAME:-sudoku-a57-v0626-expl}"
RJOB_MEMORY="${RJOB_MEMORY:-1500000}"
RJOB_CPU="${RJOB_CPU:-150}"
RJOB_GPU="${RJOB_GPU:-8}"
# Required for this run request.
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-narmodel_gpu}"

RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE:-group}"
RJOB_IMAGE="${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/rft:20260408}"
RJOB_REPLICA="${RJOB_REPLICA:-1}"
RJOB_CUSTOM_RESOURCES="${RJOB_CUSTOM_RESOURCES:-brainpp.cn/fuse=1}"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "Run script not found: ${RUN_SCRIPT}" >&2
    exit 1
fi

case "${RJOB_PRIVATE_MACHINE,,}" in
    yes|true|1)
        RJOB_PRIVATE_MACHINE="group"
        ;;
    false|0)
        RJOB_PRIVATE_MACHINE="no"
        ;;
    group|no|project|tenant)
        RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE,,}"
        ;;
    *)
        echo "Invalid RJOB_PRIVATE_MACHINE=${RJOB_PRIVATE_MACHINE}; expected group, no, project, tenant, or legacy yes/true/1/false/0." >&2
        exit 1
        ;;
esac

if [[ ! "${RJOB_NAME}" =~ ^[a-zA-Z0-9]([-a-zA-Z0-9]*[a-zA-Z0-9])?$ ]]; then
    echo "Invalid RJOB_NAME=${RJOB_NAME}; use only letters, numbers, and '-' and start/end with alnum." >&2
    exit 1
fi
if (( ${#RJOB_NAME} > 31 )); then
    echo "Invalid RJOB_NAME=${RJOB_NAME}; keep it <=31 chars because rjob builds task labels as <job>-<task> with a 63-char limit." >&2
    exit 1
fi

WANDB_KEY_FORWARDED=0
wandb_key_env_args=()
if [[ -n "${WANDB_KEY}" && ( "${WANDB_MODE}" != "offline" || "${WANDB_FORWARD_KEY:-0}" == "1" ) ]]; then
    wandb_key_env_args=(
        -e WANDB_KEY="${WANDB_KEY}"
        -e WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY}}"
    )
    WANDB_KEY_FORWARDED=1
fi

submit_cmd=(
    rjob submit
    --name="${RJOB_NAME}"
    --gpu="${RJOB_GPU}"
    --memory="${RJOB_MEMORY}"
    --cpu="${RJOB_CPU}"
    --charged-group="${RJOB_CHARGED_GROUP}"
    --private-machine="${RJOB_PRIVATE_MACHINE}"
    -P "${RJOB_REPLICA}"
    --image="${RJOB_IMAGE}"
    --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan
    --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong
    --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public
    --mount=gpfs://gpfs2/narmodel:/mnt/shared-storage-user/narmodel
    -e INSIDE_RJOB=1
    -e SUBMIT_RJOB=0
    -e OPENCLAW_HOME="${OPENCLAW_HOME}"
    -e RUN_SCRIPT="${RUN_SCRIPT}"
    -e RUN_TIMESTAMP="${RUN_TIMESTAMP}"
    -e RUN_ID="${RUN_ID}"
    -e RUN_NAME="${RUN_NAME}"
    -e NUM_GPUS="${NUM_GPUS}"
    -e RUNS_ROOT="${RUNS_ROOT}"
    -e RJOB_LOG_ROOT="${RJOB_LOG_ROOT}"
    -e RJOB_MEMORY="${RJOB_MEMORY}"
    -e RJOB_CPU="${RJOB_CPU}"
    -e RJOB_GPU="${RJOB_GPU}"
    -e RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP}"
    -e RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE}"
    -e RJOB_CUSTOM_RESOURCES="${RJOB_CUSTOM_RESOURCES}"
    -e RJOB_IMAGE="${RJOB_IMAGE}"
    -e WANDB_ENABLE="${WANDB_ENABLE}"
    -e WANDB_MODE="${WANDB_MODE}"
    -e WANDB_DIR="${WANDB_DIR}"
    -e WANDB_PROJECT="${WANDB_PROJECT}"
    -e WANDB_GROUP="${WANDB_GROUP}"
    "${wandb_key_env_args[@]}"
    --custom-resources "${RJOB_CUSTOM_RESOURCES}"
    -- bash -exc "cd \"${OPENCLAW_HOME}\" && exec bash \"${RUN_SCRIPT}\""
)

echo "Submitting OpenClaw terminal-rl v0626 rjob:"
echo "  name:       ${RJOB_NAME}"
echo "  run_id:     ${RUN_ID}"
echo "  group:      ${RJOB_CHARGED_GROUP}"
echo "  gpu:        ${RJOB_GPU}"
echo "  script:     ${RUN_SCRIPT}"
echo "  logs:       ${RJOB_LOG_ROOT}/${RUN_ID}"
echo "  wandb:      ${WANDB_MODE} -> ${WANDB_DIR}"
echo "  wandb_key:  forwarded=${WANDB_KEY_FORWARDED}"

if [[ "${RJOB_DRY_RUN:-0}" == "1" ]]; then
    printf 'Dry-run command:'
    printf ' %q' "${submit_cmd[@]}"
    printf '\n'
    exit 0
fi

exec "${submit_cmd[@]}"
