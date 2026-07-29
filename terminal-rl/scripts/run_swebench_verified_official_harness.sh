#!/usr/bin/env bash
# Score terminal-rl predictions with a pinned official SWE-bench harness.
set -euo pipefail

OFFICIAL_COMMIT="f7bbbb2ccdf479001d6467c9e34af59e44a840f9"
OFFICIAL_VERSION="4.1.0"
DATASET_NAME="princeton-nlp/SWE-bench_Verified"
OFFICIAL_DATASET_SHA256="f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c"

RUN_DIR="${RUN_DIR:-${1:-}}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: RUN_DIR=/path/to/eval-run bash $0" >&2
  exit 2
fi
RUN_DIR="$(realpath -m -- "${RUN_DIR}")"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-${RUN_DIR}/swebench_official/predictions.jsonl}"
DATASET_PATH="${DATASET_PATH:-${RUN_DIR}/config/swebench_verified_official_test.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_DIR}/swebench_official/harness}"
OFFICIAL_RUN_ID="${OFFICIAL_RUN_ID:-$(basename "${RUN_DIR}")-official}"
MAX_WORKERS="${MAX_WORKERS:-4}"
OPEN_FILE_LIMIT="${OPEN_FILE_LIMIT:-65536}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"
CACHE_LEVEL="${CACHE_LEVEL:-env}"
CLEAN="${CLEAN:-false}"
OVERWRITE_OFFICIAL_RESULTS="${OVERWRITE_OFFICIAL_RESULTS:-0}"

VENV_DIR="${SWEBENCH_VENV_DIR:-${HOME}/.cache/openclaw/swebench-${OFFICIAL_COMMIT:0:12}}"
SOURCE_DIR="${SWEBENCH_SOURCE_DIR:-${HOME}/.cache/openclaw/SWE-bench-${OFFICIAL_COMMIT:0:12}}"
SWEBENCH_PYTHON="${SWEBENCH_PYTHON:-${VENV_DIR}/bin/python}"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  mkdir -p "$(dirname -- "${SOURCE_DIR}")"
  git clone https://github.com/SWE-bench/SWE-bench.git "${SOURCE_DIR}"
  git -C "${SOURCE_DIR}" checkout --detach "${OFFICIAL_COMMIT}"
fi
ACTUAL_SOURCE_COMMIT="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_SOURCE_COMMIT}" != "${OFFICIAL_COMMIT}" ]]; then
  echo "[ERROR] SWE-bench source commit mismatch: ${SOURCE_DIR}" >&2
  echo "        actual=${ACTUAL_SOURCE_COMMIT} expected=${OFFICIAL_COMMIT}" >&2
  exit 2
fi
if ! git -C "${SOURCE_DIR}" diff --quiet ||
   ! git -C "${SOURCE_DIR}" diff --cached --quiet; then
  echo "[ERROR] SWE-bench source checkout has tracked modifications: ${SOURCE_DIR}" >&2
  exit 2
fi

if [[ ! -x "${SWEBENCH_PYTHON}" ]]; then
  python3 -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
fi
INSTALL_MARKER="${VENV_DIR}/.openclaw-swebench-source-commit"
if [[ ! -f "${INSTALL_MARKER}" ]] ||
   [[ "$(<"${INSTALL_MARKER}")" != "${OFFICIAL_COMMIT}" ]]; then
  "${SWEBENCH_PYTHON}" -m pip install --editable "${SOURCE_DIR}"
  printf '%s\n' "${OFFICIAL_COMMIT}" > "${INSTALL_MARKER}"
fi

"${SWEBENCH_PYTHON}" - \
  "${PREDICTIONS_PATH}" \
  "${OFFICIAL_COMMIT}" \
  "${OFFICIAL_VERSION}" \
  "${SOURCE_DIR}" \
  "${DATASET_PATH}" \
  "${OFFICIAL_DATASET_SHA256}" <<'PY'
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import sys
from pathlib import Path

predictions_path = Path(sys.argv[1])
expected_commit = sys.argv[2]
expected_version = sys.argv[3]
source_dir = Path(sys.argv[4]).resolve()
dataset_path = Path(sys.argv[5]).resolve()
expected_dataset_sha256 = sys.argv[6]
if not predictions_path.is_file():
    raise SystemExit(f"[ERROR] predictions file does not exist: {predictions_path}")
if not dataset_path.is_file():
    raise SystemExit(f"[ERROR] pinned dataset file does not exist: {dataset_path}")
actual_dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
if actual_dataset_sha256 != expected_dataset_sha256:
    raise SystemExit(
        "[ERROR] pinned dataset fingerprint mismatch: "
        f"actual={actual_dataset_sha256} expected={expected_dataset_sha256}"
    )

distribution = metadata.distribution("swebench")
actual_version = distribution.version
spec = importlib.util.find_spec("swebench")
module_path = Path(spec.origin).resolve() if spec and spec.origin else None
if actual_version != expected_version:
    raise SystemExit(
        "[ERROR] official harness provenance mismatch: "
        f"version={actual_version!r} expected={expected_version!r}"
    )
if module_path is None or source_dir not in module_path.parents:
    raise SystemExit(
        "[ERROR] swebench is not imported from the pinned source checkout: "
        f"module={module_path} source={source_dir}"
    )

rows = []
ids = set()
for line_no, line in enumerate(predictions_path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    row = json.loads(line)
    if set(row) != {"instance_id", "model_name_or_path", "model_patch"}:
        raise SystemExit(f"[ERROR] prediction row {line_no} has an invalid schema")
    instance_id = str(row["instance_id"])
    if instance_id in ids:
        raise SystemExit(f"[ERROR] duplicate prediction ID: {instance_id}")
    ids.add(instance_id)
    rows.append(row)
if len(rows) != 500:
    raise SystemExit(f"[ERROR] official SWE-Verified requires 500 predictions; found {len(rows)}")
dataset_rows = [
    json.loads(line)
    for line in dataset_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
dataset_ids = [str(row.get("instance_id", "")) for row in dataset_rows]
if len(dataset_rows) != 500 or len(set(dataset_ids)) != 500:
    raise SystemExit(
        "[ERROR] pinned official SWE-Verified dataset must contain 500 unique rows"
    )
if set(dataset_ids) != ids:
    raise SystemExit(
        "[ERROR] prediction IDs do not match the pinned official dataset"
    )
print(
    f"[official-harness] preflight=ok predictions={len(rows)} "
    f"dataset={dataset_path} swebench={actual_version}@{expected_commit} "
    f"source={source_dir}"
)
PY

# Import the complete harness before launching Docker. This catches missing
# package data such as constants/fixtures that metadata-only checks cannot see.
"${SWEBENCH_PYTHON}" -c \
  'from swebench.harness.run_evaluation import main; assert callable(main)'

if [[ "${HARNESS_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[official-harness] preflight-only complete"
  exit 0
fi
if [[ -d "${RESULTS_DIR}" ]] &&
   find "${RESULTS_DIR}" -mindepth 1 -print -quit | grep -q .; then
  if [[ "${OVERWRITE_OFFICIAL_RESULTS}" != "1" ]]; then
    echo "[ERROR] official harness result directory is not empty: ${RESULTS_DIR}" >&2
    echo "        Use a new RUN_DIR/OFFICIAL_RUN_ID, or set OVERWRITE_OFFICIAL_RESULTS=1." >&2
    exit 2
  fi
  rm -rf -- "${RESULTS_DIR}"
fi
mkdir -p "${RESULTS_DIR}"
PREDICTIONS_SHA256="$(sha256sum "${PREDICTIONS_PATH}" | awk '{print $1}')"
DATASET_SHA256="$(sha256sum "${DATASET_PATH}" | awk '{print $1}')"
cat > "${RESULTS_DIR}/evaluation_manifest.json" <<EOF
{
  "swebench_commit": "${OFFICIAL_COMMIT}",
  "swebench_version": "${OFFICIAL_VERSION}",
  "dataset_name": "${DATASET_NAME}",
  "dataset_path": "${DATASET_PATH}",
  "dataset_sha256": "${DATASET_SHA256}",
  "predictions_path": "${PREDICTIONS_PATH}",
  "predictions_sha256": "${PREDICTIONS_SHA256}",
  "official_run_id": "${OFFICIAL_RUN_ID}",
  "image_namespace": "swebench",
  "instance_image_tag": "latest"
}
EOF
cd "${RESULTS_DIR}"
exec "${SWEBENCH_PYTHON}" -m swebench.harness.run_evaluation \
  --dataset_name "${DATASET_PATH}" \
  --split test \
  --predictions_path "${PREDICTIONS_PATH}" \
  --max_workers "${MAX_WORKERS}" \
  --open_file_limit "${OPEN_FILE_LIMIT}" \
  --timeout "${EVAL_TIMEOUT}" \
  --cache_level "${CACHE_LEVEL}" \
  --clean "${CLEAN}" \
  --run_id "${OFFICIAL_RUN_ID}" \
  --namespace swebench \
  --instance_image_tag latest \
  --report_dir "${RESULTS_DIR}"
