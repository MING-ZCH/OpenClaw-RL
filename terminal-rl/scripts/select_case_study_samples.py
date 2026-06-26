#!/usr/bin/env python3
"""Select fixed case-study samples for SetA, AgentHarm, and Agent-SafetyBench.

The selector is deterministic. It builds a reusable YAML config containing
sample IDs and short rationales, so later runs can be compared on the same
cases instead of re-sampling ad hoc.

Selection policy:
  * SetA: cover different operational task families such as service debugging,
    privilege/PATH issues, web-service configuration, and permission-heavy tasks.
  * Agent-SafetyBench: cover harmful/unfulfillable and benign/fulfillable tasks,
    different risk categories, and different tool environments.
  * AgentHarm: cover harmful and benign variants across distinct harm categories,
    with and without explicit hints.

Manual IDs override the default selector for that dataset:
  python terminal-rl/scripts/select_case_study_samples.py \
    --seta-id 661 --asb-id 0 --agentharm-id agentharm_harmful_test_public_1-1

Outputs a YAML file. By default it writes
``terminal-rl/scripts/case_study_samples.yaml``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency exists in this repo env
    raise SystemExit("PyYAML is required for this script: pip install pyyaml") from exc


DATASET_DEFAULTS = {
    "seta": "terminal-rl/dataset/seta_env_convert/train.jsonl",
    "agent_safetybench": "terminal-rl/dataset/agent_safetybench_convert/train.jsonl",
    "agentharm": "terminal-rl/dataset/agentharm_convert/train.jsonl",
}

SETA_BUCKETS = [
    (
        "privilege-boundary",
        ("sudo", "privileged", "permission"),
        "SetA failure mode: privilege/shell-context handling and restricted paths.",
        ("338",),
    ),
    (
        "service-debugging",
        ("systemd", "service", "journalctl"),
        "SetA operational debugging case with daemon/service state.",
        ("705",),
    ),
    (
        "web-stack",
        ("nginx", "php-fpm", "virtual hosts"),
        "SetA multi-step web stack configuration with external validation.",
        ("661",),
    ),
    (
        "permissions",
        ("groups", "directory structure", "multi-tenant file sharing"),
        "SetA multi-tenant permission case; useful for partial-success analysis.",
        ("1072",),
    ),
    (
        "path-runtime",
        ("library-related", "shared libraries", "binary paths", "path and binary"),
        "SetA runtime/PATH/library-resolution case; common source of tool errors.",
        ("1355", "592"),
    ),
]

ASB_BUCKETS = [
    (
        "harmful-email",
        {"fulfillable": 0, "env": "Email"},
        "ASB harmful/unfulfillable email case; should refuse without tool use.",
    ),
    (
        "benign-email",
        {"fulfillable": 1, "env": "Email"},
        "ASB benign/fulfillable email case; should help without over-refusal.",
    ),
    (
        "harmful-os",
        {"fulfillable": 0, "env": "OS"},
        "ASB OS/tool-use harmful case; catches unsafe tool attempts.",
    ),
    (
        "benign-file",
        {"fulfillable": 1, "env": "FileManager"},
        "ASB benign file-management case; checks format/tool compatibility.",
    ),
    (
        "availability-risk",
        {"risk_contains": "Compromise availability"},
        "ASB availability-risk boundary case.",
    ),
]

AGENTHARM_BUCKETS = [
    (
        "harmful-disinformation-hint",
        {"category": "Disinformation", "hint_included": True},
        "AgentHarm harmful disinformation case with explicit tool hint.",
    ),
    (
        "harmful-fraud-nohint",
        {"category": "Fraud", "hint_included": False},
        "AgentHarm harmful fraud case without hint; tests refusal without target sequence.",
    ),
    (
        "harmful-cybercrime-hint",
        {"category": "Cybercrime", "hint_included": True},
        "AgentHarm cybercrime case; high-risk tool-use behavior.",
    ),
    (
        "benign-disinformation-counterpart",
        {"category": "Disinformation", "harmfulness": "benign"},
        "AgentHarm benign counterpart; should complete safe target tools.",
    ),
    (
        "benign-fraud-counterpart",
        {"category": "Fraud", "harmfulness": "benign"},
        "AgentHarm benign counterpart for over-refusal checks.",
    ),
]


def repo_root_from_script() -> Path:
    env = os.getenv("OPENCLAW_RL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, dict) else {}


def prompt_text(record: dict[str, Any]) -> str:
    task = record.get("task")
    if isinstance(task, list):
        parts = []
        for msg in task:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content") or ""))
            else:
                parts.append(str(msg))
        return "\n".join(part for part in parts if part)
    return str(task or "")


def compact(text: str, limit: int = 360) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def id_values(record: dict[str, Any]) -> set[str]:
    meta = metadata(record)
    values = {
        meta.get("task_name"),
        meta.get("task_path"),
        meta.get("id"),
        meta.get("id_original"),
        meta.get("name"),
    }
    return {str(v) for v in values if v is not None and str(v) != ""}


def canonical_id(record: dict[str, Any]) -> str:
    meta = metadata(record)
    for key in ("task_name", "id", "task_path"):
        value = meta.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return compact(prompt_text(record), 80)


def split_manual_ids(values: list[str]) -> list[str]:
    ids: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                ids.append(part)
    return ids


def find_by_ids(records: list[dict[str, Any]], ids: list[str]) -> list[tuple[dict[str, Any], str]]:
    selected: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    for wanted in ids:
        match = None
        for record in records:
            if wanted in id_values(record):
                match = record
                break
        if match is None:
            raise ValueError(f"manual case-study id not found: {wanted}")
        cid = canonical_id(match)
        if cid in seen:
            continue
        seen.add(cid)
        selected.append((match, f"Manually pinned sample id `{wanted}`."))
    return selected


def first_unseen(
    records: list[dict[str, Any]],
    predicate,
    seen: set[str],
) -> dict[str, Any] | None:
    for record in records:
        cid = canonical_id(record)
        if cid in seen:
            continue
        if predicate(record):
            seen.add(cid)
            return record
    return None


def select_seta(records: list[dict[str, Any]], limit: int) -> list[tuple[dict[str, Any], str]]:
    selected: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    for _bucket_name, keywords, reason, preferred_ids in SETA_BUCKETS:
        if len(selected) >= limit:
            break

        preferred = first_unseen(
            records,
            lambda record, ids=set(preferred_ids): bool(ids & id_values(record)),
            seen,
        )
        if preferred is not None:
            selected.append((preferred, reason))
            continue

        def pred(record: dict[str, Any]) -> bool:
            text = (metadata(record).get("instruction") or prompt_text(record)).lower()
            return any(keyword in text for keyword in keywords)

        record = first_unseen(records, pred, seen)
        if record is not None:
            selected.append((record, reason))
    return fill_from_start(records, selected, seen, limit)


def asb_env(record: dict[str, Any]) -> str:
    envs = metadata(record).get("environments") or []
    if envs and isinstance(envs[0], dict):
        return str(envs[0].get("name") or "")
    return ""


def select_asb(records: list[dict[str, Any]], limit: int) -> list[tuple[dict[str, Any], str]]:
    selected: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    for _bucket_name, criteria, reason in ASB_BUCKETS:
        if len(selected) >= limit:
            break

        def pred(record: dict[str, Any]) -> bool:
            meta = metadata(record)
            if "fulfillable" in criteria and int(meta.get("fulfillable", -1)) != criteria["fulfillable"]:
                return False
            if "env" in criteria and asb_env(record) != criteria["env"]:
                return False
            if "risk_contains" in criteria:
                risks = [str(x) for x in (meta.get("risks") or [])]
                if not any(criteria["risk_contains"] in risk for risk in risks):
                    return False
            return True

        record = first_unseen(records, pred, seen)
        if record is not None:
            selected.append((record, reason))
    return fill_from_start(records, selected, seen, limit)


def agentharm_harmfulness(record: dict[str, Any]) -> str:
    path = str(metadata(record).get("task_path") or "").lower()
    name = str(metadata(record).get("task_name") or "").lower()
    if "benign" in path or "harmless" in path or "benign" in name:
        return "benign"
    return "harmful"


def select_agentharm(records: list[dict[str, Any]], limit: int) -> list[tuple[dict[str, Any], str]]:
    selected: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    for _bucket_name, criteria, reason in AGENTHARM_BUCKETS:
        if len(selected) >= limit:
            break

        def pred(record: dict[str, Any]) -> bool:
            meta = metadata(record)
            if "category" in criteria and meta.get("category") != criteria["category"]:
                return False
            if "hint_included" in criteria and bool(meta.get("hint_included")) != criteria["hint_included"]:
                return False
            if "harmfulness" in criteria and agentharm_harmfulness(record) != criteria["harmfulness"]:
                return False
            return True

        record = first_unseen(records, pred, seen)
        if record is not None:
            selected.append((record, reason))
    return fill_from_start(records, selected, seen, limit)


def fill_from_start(
    records: list[dict[str, Any]],
    selected: list[tuple[dict[str, Any], str]],
    seen: set[str],
    limit: int,
) -> list[tuple[dict[str, Any], str]]:
    for record in records:
        if len(selected) >= limit:
            break
        cid = canonical_id(record)
        if cid in seen:
            continue
        seen.add(cid)
        selected.append((record, "Deterministic fallback sample to reach the requested count."))
    return selected


def sample_payload(dataset: str, record: dict[str, Any], reason: str) -> dict[str, Any]:
    meta = metadata(record)
    prompt = prompt_text(record)
    payload = {
        "id": canonical_id(record),
        "task_name": str(meta.get("task_name") or ""),
        "task_path": str(meta.get("task_path") or ""),
        "data_source": str(meta.get("data_source") or ("terminal_bench" if dataset == "seta" else dataset)),
        "title": str(meta.get("name") or meta.get("category") or meta.get("task_name") or canonical_id(record)),
        "selection_reason": reason,
        "prompt_preview": compact(meta.get("instruction") or prompt),
        "match": {
            "task_name": str(meta.get("task_name") or ""),
            "task_path": str(meta.get("task_path") or ""),
            "id": str(meta.get("id") or ""),
            "id_original": str(meta.get("id_original") or ""),
        },
    }
    if dataset == "agent_safetybench":
        payload["asb"] = {
            "fulfillable": meta.get("fulfillable"),
            "risks": meta.get("risks") or [],
            "environment": asb_env(record),
        }
    if dataset == "agentharm":
        payload["agentharm"] = {
            "category": meta.get("category"),
            "harmfulness": agentharm_harmfulness(record),
            "hint_included": meta.get("hint_included"),
            "target_functions": meta.get("target_functions") or [],
        }
    return payload


def build_config(
    *,
    repo_root: Path,
    dataset_paths: dict[str, Path],
    samples_per_dataset: int,
    manual_ids: dict[str, list[str]],
) -> dict[str, Any]:
    records = {name: read_jsonl(path) for name, path in dataset_paths.items()}
    selectors = {
        "seta": select_seta,
        "agent_safetybench": select_asb,
        "agentharm": select_agentharm,
    }

    datasets: dict[str, Any] = {}
    for dataset, data in records.items():
        ids = manual_ids.get(dataset) or []
        if ids:
            chosen = find_by_ids(data, ids)
        else:
            chosen = selectors[dataset](data, samples_per_dataset)
        rel_path = dataset_paths[dataset]
        try:
            rel_text = str(rel_path.relative_to(repo_root))
        except ValueError:
            rel_text = str(rel_path)
        datasets[dataset] = {
            "dataset_path": rel_text,
            "available": len(data),
            "samples": [
                sample_payload(dataset, record, reason)
                for record, reason in chosen[:samples_per_dataset]
            ],
        }

    return {
        "schema": "openclaw.case_study_samples.v1",
        "description": (
            "Fixed case-study samples for SetA, agent-safety-bench, and AgentHarm. "
            "Samples are selected to cover task family, risk level, success/failure "
            "modes, and boundary cases; edit samples[].id/task_name/task_path to pin "
            "additional cases."
        ),
        "selection_policy": {
            "seta": "service debugging, privilege/PATH issues, web stack setup, permissions, runtime resolution",
            "agent_safetybench": "fulfillable and unfulfillable tasks across risk/tool environments",
            "agentharm": "harmful and benign variants across harm categories and hint settings",
        },
        "datasets": datasets,
    }


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--seta-data", type=Path, default=repo_root / DATASET_DEFAULTS["seta"])
    parser.add_argument(
        "--asb-data",
        type=Path,
        default=repo_root / DATASET_DEFAULTS["agent_safetybench"],
        help="agent-safety-bench converted JSONL.",
    )
    parser.add_argument(
        "--agentharm-data",
        type=Path,
        default=repo_root / DATASET_DEFAULTS["agentharm"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "terminal-rl/scripts/case_study_samples.yaml",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=5)
    parser.add_argument("--seta-id", action="append", default=[], help="Repeatable or comma-separated manual SetA IDs.")
    parser.add_argument("--asb-id", action="append", default=[], help="Repeatable or comma-separated ASB task IDs.")
    parser.add_argument("--agentharm-id", action="append", default=[], help="Repeatable or comma-separated AgentHarm IDs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    dataset_paths = {
        "seta": args.seta_data.expanduser().resolve(),
        "agent_safetybench": args.asb_data.expanduser().resolve(),
        "agentharm": args.agentharm_data.expanduser().resolve(),
    }
    for name, path in dataset_paths.items():
        if not path.exists():
            print(f"[ERROR] {name} dataset not found: {path}", file=sys.stderr)
            return 1

    manual_ids = {
        "seta": split_manual_ids(args.seta_id),
        "agent_safetybench": split_manual_ids(args.asb_id),
        "agentharm": split_manual_ids(args.agentharm_id),
    }
    cfg = build_config(
        repo_root=repo_root,
        dataset_paths=dataset_paths,
        samples_per_dataset=max(1, args.samples_per_dataset),
        manual_ids=manual_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    print(f"[+] wrote {args.output}")
    for dataset, info in cfg["datasets"].items():
        ids = ", ".join(sample["id"] for sample in info["samples"])
        print(f"    {dataset}: {ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
