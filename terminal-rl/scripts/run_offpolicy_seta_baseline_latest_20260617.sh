#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "${SCRIPT_DIR}/run_offpolicy_seta_latest_20260617.sh" none
