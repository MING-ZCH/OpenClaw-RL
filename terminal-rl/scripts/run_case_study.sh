#!/usr/bin/env bash
# Run case-study analysis for one terminal-rl run.
#
# Usage:
#   bash terminal-rl/scripts/run_case_study.sh runs/<run_id>
#   CASE_STUDY_CONFIG=/path/to/case_study_samples.yaml \
#     bash terminal-rl/scripts/run_case_study.sh runs/<run_id>
#
# Outputs are written to:
#   <run_dir>/case_study/case_study_report.md
#   <run_dir>/case_study/case_study_summary.json
#   <run_dir>/case_study/case_study_records.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<EOF
Usage: bash terminal-rl/scripts/run_case_study.sh RUN_DIR [CONFIG]

Environment:
  CASE_STUDY_CONFIG                 Config path if CONFIG is not passed.
  CASE_STUDY_MAX_TRAJ_PER_SAMPLE    Default: 3.
  CASE_STUDY_MAX_TEXT_CHARS         Default: 1600.
  CASE_STUDY_MAX_TOOL_RESULT_CHARS  Default: 1200.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  usage >&2
  exit 1
fi
shift || true

CONFIG="${1:-${CASE_STUDY_CONFIG:-${SCRIPT_DIR}/case_study_samples.yaml}}"
MAX_TRAJ="${CASE_STUDY_MAX_TRAJ_PER_SAMPLE:-3}"
MAX_TEXT="${CASE_STUDY_MAX_TEXT_CHARS:-1600}"
MAX_TOOL="${CASE_STUDY_MAX_TOOL_RESULT_CHARS:-1200}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[ERROR] run dir not found: ${RUN_DIR}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[WARN] config not found: ${CONFIG}" >&2
  echo "[WARN] generating a run-local default config under ${RUN_DIR}/case_study/" >&2
  mkdir -p "${RUN_DIR}/case_study"
  CONFIG="${RUN_DIR}/case_study/case_study_samples.yaml"
  python3 "${SCRIPT_DIR}/select_case_study_samples.py" \
    --repo-root "${REPO_ROOT}" \
    --output "${CONFIG}"
fi

python3 "${SCRIPT_DIR}/analyze_case_study.py" \
  --run-dir "${RUN_DIR}" \
  --config "${CONFIG}" \
  --max-trajectories-per-sample "${MAX_TRAJ}" \
  --max-text-chars "${MAX_TEXT}" \
  --max-tool-result-chars "${MAX_TOOL}"
