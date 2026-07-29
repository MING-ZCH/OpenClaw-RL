"""Write SWE-bench-compatible prediction, instance, and run reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _task_meta(sample: Any) -> dict[str, Any]:
    metadata = _as_dict(getattr(sample, "metadata", None))
    prompt = getattr(sample, "prompt", None)
    nested = metadata.get("task_meta")
    if isinstance(nested, dict):
        return nested
    if isinstance(prompt, dict):
        nested = prompt.get("metadata")
        if isinstance(nested, dict):
            return nested
        if any(key in prompt for key in ("swe_instance_id", "task_name", "task_path")):
            return prompt
    return metadata


def _sample_status(sample: Any) -> str:
    status = getattr(sample, "status", None)
    value = getattr(status, "value", status)
    if value is None:
        return "unknown"
    return str(value).rsplit(".", 1)[-1].strip().lower() or "unknown"


def _load_expected_ids(path: Path) -> list[str]:
    ids: list[str] = []
    if not path.is_file():
        return ids
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        metadata = _as_dict(row.get("metadata"))
        instance_id = metadata.get("swe_instance_id") or metadata.get("task_name")
        if instance_id:
            ids.append(str(instance_id))
    return ids


def build_prediction_coverage(
    *,
    expected_ids: Iterable[str],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Describe generation coverage without impersonating an official score."""

    expected = set(expected_ids)
    submitted = set(predictions)
    incomplete = expected - submitted
    unexpected = submitted - expected
    empty_patch = {
        instance_id
        for instance_id in expected & submitted
        if predictions[instance_id].get("model_patch") in {None, ""}
    }

    def ordered(values: set[str]) -> list[str]:
        return sorted(values)

    return {
        "schema": "terminal_rl.swebench_prediction_coverage.v1",
        "expected": len(expected),
        "submitted": len(predictions),
        "submitted_expected": len(expected & submitted),
        "empty_patch": len(empty_patch),
        "incomplete": len(incomplete),
        "unexpected": len(unexpected),
        "incomplete_ids": ordered(incomplete),
        "unexpected_ids": ordered(unexpected),
        "empty_patch_ids": ordered(empty_patch),
        "submitted_ids": ordered(submitted),
    }


def write_official_artifacts(samples: Iterable[Any]) -> dict[str, Any] | None:
    """Persist official-format artifacts once for a SWE-bench eval rollout."""

    results_dir_raw = os.getenv("SWEBENCH_RESULTS_DIR", "").strip()
    dataset_path_raw = os.getenv("SWEBENCH_EVAL_DATA_PATH", "").strip()
    if not results_dir_raw or not dataset_path_raw:
        return None

    model_name = os.getenv("SWEBENCH_MODEL_NAME_OR_PATH", "Qwen/Qwen3-8B")
    run_id = os.getenv("RUN_ID", "terminal_rl_sweverified")
    results_dir = Path(results_dir_raw)
    expected_ids = _load_expected_ids(Path(dataset_path_raw))
    if not expected_ids:
        raise RuntimeError(f"No SWE-bench instance IDs found in {dataset_path_raw}")

    predictions: dict[str, dict[str, Any]] = {}
    audit_details: dict[str, dict[str, Any]] = {}
    generation_status_counts: dict[str, int] = {}
    technical_failure_ids: set[str] = set()
    deferred_grading_ids: set[str] = set()
    for sample in samples:
        metadata = _as_dict(getattr(sample, "metadata", None))
        details = _as_dict(metadata.get("reward_details"))
        task_meta = _task_meta(sample)
        instance_id = (
            details.get("instance_id")
            or task_meta.get("swe_instance_id")
            or task_meta.get("task_name")
        )
        if not instance_id:
            continue
        instance_id = str(instance_id)

        status = _sample_status(sample)
        technical_failure = (
            status == "aborted"
            or bool(getattr(sample, "remove_sample", False))
            or metadata.get("evaluation_failed") is True
        )
        if technical_failure:
            technical_failure_ids.add(instance_id)
        # Multi-turn terminal trajectories contain one training Sample per turn.
        # Only the final turn carries the prediction-export details.
        if not details:
            continue
        if instance_id in predictions:
            raise RuntimeError(
                f"Duplicate SWE-bench prediction generated for {instance_id}"
            )

        generation_status_counts[status] = generation_status_counts.get(status, 0) + 1
        if details.get("grading_deferred") is True:
            deferred_grading_ids.add(instance_id)

        model_patch = details.get("model_patch")
        predictions[instance_id] = {
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": model_patch,
        }
        if details:
            audit_details[instance_id] = details
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = results_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fout:
        for instance_id in sorted(predictions):
            fout.write(json.dumps(predictions[instance_id], ensure_ascii=False) + "\n")

    (results_dir / "instance_audit.json").write_text(
        json.dumps(audit_details, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    coverage = build_prediction_coverage(
        expected_ids=expected_ids,
        predictions=predictions,
    )
    coverage_name = "prediction_coverage.json"
    (results_dir / coverage_name).write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "benchmark": "SWE-bench Verified",
        "model_name_or_path": model_name,
        "run_id": run_id,
        "submitted": coverage["submitted"],
        "empty_patch": coverage["empty_patch"],
        "incomplete": coverage["incomplete"],
        "unexpected": coverage["unexpected"],
        "unexpected_ids": coverage["unexpected_ids"],
        "technical_failures": len(technical_failure_ids),
        "technical_failure_ids": sorted(technical_failure_ids),
        "grading_deferred": bool(deferred_grading_ids),
        "pending_official_grading": len(deferred_grading_ids),
        "pending_official_grading_ids": sorted(deferred_grading_ids),
        "generation_status_counts": dict(sorted(generation_status_counts.items())),
        "total": coverage["expected"],
        "authoritative_score": None,
        "official_grading_required": True,
        "prediction_coverage": coverage_name,
        "predictions": predictions_path.name,
    }
    (results_dir / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
