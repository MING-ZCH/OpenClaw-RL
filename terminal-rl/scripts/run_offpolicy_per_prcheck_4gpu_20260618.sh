#!/usr/bin/env bash
# Focused 4-GPU smoke runner for SETA + off-policy PER.
#
# This is the fastest path to validate PER after DAPO/TOPR/SPEAR have already
# passed.  It keeps logs isolated from full PR-check runs and aggressively
# cleans local Ray/SGLang state before launch to avoid stale GPU memory from a
# previous mode.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export CHECK_ROOT="${CHECK_ROOT:-${REPO_ROOT}/runs/offpolicy_per_prcheck_4gpu_${TIMESTAMP}}"
export CHECK_MODES="per"
export PR_CHECK_CLEAN_BETWEEN_MODES="${PR_CHECK_CLEAN_BETWEEN_MODES:-1}"
export PR_CHECK_ONEKEY_KILL_GPU_PROCS="${PR_CHECK_ONEKEY_KILL_GPU_PROCS:-1}"
export PR_CHECK_INTER_MODE_SLEEP="${PR_CHECK_INTER_MODE_SLEEP:-15}"

export MAX_TURN="${MAX_TURN:-3}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
export N_SAMPLES="${N_SAMPLES:-4}"

exec bash "${SCRIPT_DIR}/run_offpolicy_pr_check_4gpu_no_spear_20260617.sh"
