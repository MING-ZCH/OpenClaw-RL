#!/usr/bin/env python3
"""Diagnose whether a terminal-rl run ended in rollout generation stalls.

This complements plot_training_metrics.py: it focuses on the tail of
logs/train.log and on post-last-rollout infra errors such as env /reset 500
and Unknown run_lease_id loops.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
ROLLOUT_RE = re.compile(r"data\.py:\d+ - rollout (\d+): (\{.+\})")
TRAIN_RE = re.compile(r"model\.py:\d+ - step (\d+): (\{.+\})")
RESET_500_RE = re.compile(r"Server error '500 .*?/reset")
HEARTBEAT_500_RE = re.compile(r"Server error '500 .*?/heartbeat")
EVALUATE_500_RE = re.compile(r"Server error '500 .*?/evaluate")
EXEC_TOOL_500_RE = re.compile(r"Server error '500 .*?/exec_tool")
GENERATE_FAILED_RE = re.compile(r"Generate failed \(([^)]+)\).*?url '([^']+)'")
FLAG_VALUE_RE = re.compile(r"(?<!\S)--([a-zA-Z0-9-]+)(?:[=\s]+([^\s]+))?")


def _timestamp(line: str) -> str | None:
    match = TIMESTAMP_RE.search(line)
    return match.group(1) if match else None


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _limited_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rollout/rewards",
        "rollout/raw_reward",
        "rollout/response_lengths",
        "rollout/truncated",
        "train/rollout_id",
        "train/step",
        "train/loss",
        "train/pg_loss",
        "train/entropy_loss",
        "train/grad_norm",
    ]
    return {key: payload.get(key) for key in keys if key in payload}


def _bump(counter: Counter[str], line: str) -> None:
    if RESET_500_RE.search(line):
        counter["reset_500"] += 1
    if HEARTBEAT_500_RE.search(line):
        counter["heartbeat_500"] += 1
    if EVALUATE_500_RE.search(line):
        counter["evaluate_500"] += 1
    if EXEC_TOOL_500_RE.search(line):
        counter["exec_tool_500"] += 1
    if "Unknown run_lease_id" in line:
        counter["unknown_run_lease_id"] += 1
    if "docker compose" in line.lower() or "compose failed" in line.lower():
        counter["docker_compose"] += 1
    if "Generate failed" in line:
        counter["generate_failed"] += 1
    if "ValidationError" in line:
        counter["validation_error"] += 1
    if "No final response produced" in line:
        counter["no_final_response"] += 1
    if "Start terminal rollout" in line:
        counter["start_terminal_rollout"] += 1
    if "Rollout finished: status=" in line:
        counter["terminal_rollout_finished"] += 1
    if "GET /health HTTP/1.1" in line:
        counter["health_check"] += 1
    if ROLLOUT_RE.search(line):
        counter["data_rollout"] += 1
    if TRAIN_RE.search(line):
        counter["train_step"] += 1


def _extract_flags(line: str, flags: dict[str, str | bool]) -> None:
    if "--dynamic-sampling-filter-path" not in line and "--rollout-batch-size" not in line:
        return
    for match in FLAG_VALUE_RE.finditer(line):
        key = match.group(1)
        value = match.group(2)
        if key in {
            "dynamic-sampling-filter-path",
            "rollout-batch-size",
            "n-samples-per-prompt",
            "over-sampling-batch-size",
            "num-rollout",
            "num-steps-per-rollout",
        }:
            flags[key] = value if value is not None else True


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _read_run_config_flags(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    path = run_dir / "config" / "run_config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    result: dict[str, Any] = {}
    for key in ("dapo_dynamic_sampling", "dapo_dynamic_filter_path"):
        if key in data:
            result[key] = data.get(key)
    return result


def _classify_tail(line: str) -> str:
    if "GET /health HTTP/1.1" in line:
        return "health_check"
    if RESET_500_RE.search(line) or "Unknown run_lease_id" in line:
        return "env_reset_or_lease_error"
    if "Generate failed" in line:
        return "generate_failed"
    if "Start terminal rollout" in line:
        return "start_terminal_rollout"
    if "Rollout finished: status=" in line:
        return "terminal_rollout_finished"
    if ROLLOUT_RE.search(line):
        return "data_rollout"
    if TRAIN_RE.search(line):
        return "train_step"
    return "other"


def analyze(log_path: Path, tail_lines: int, run_dir: Path | None = None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    post_counts: Counter[str] = Counter()
    tail = deque(maxlen=tail_lines)
    flags: dict[str, str | bool] = {}
    rollouts: list[dict[str, Any]] = []
    train_steps: list[dict[str, Any]] = []
    generate_failed_by_url: Counter[str] = Counter()
    line_count = 0
    last_timestamp: str | None = None

    with log_path.open(errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line_count = line_no
            tail.append(line.rstrip("\n"))
            ts = _timestamp(line)
            if ts:
                last_timestamp = ts
            _extract_flags(line, flags)
            _bump(counts, line)

            m = GENERATE_FAILED_RE.search(line)
            if m:
                generate_failed_by_url[m.group(2)] += 1

            m = ROLLOUT_RE.search(line)
            if m:
                payload: dict[str, Any] = {}
                try:
                    payload = ast.literal_eval(m.group(2))
                except Exception:
                    pass
                rollouts.append(
                    {
                        "id": int(m.group(1)),
                        "line": line_no,
                        "timestamp": ts,
                        "metrics": _limited_payload(payload),
                    }
                )
                continue

            m = TRAIN_RE.search(line)
            if m:
                payload = {}
                try:
                    payload = ast.literal_eval(m.group(2))
                except Exception:
                    pass
                train_steps.append(
                    {
                        "step_label": int(m.group(1)),
                        "line": line_no,
                        "timestamp": ts,
                        "metrics": _limited_payload(payload),
                    }
                )

    last_rollout = rollouts[-1] if rollouts else None
    last_train = train_steps[-1] if train_steps else None

    if last_rollout is not None:
        with log_path.open(errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                if line_no <= int(last_rollout["line"]):
                    continue
                _bump(post_counts, line)

    tail_list = list(tail)
    tail_counts = Counter(_classify_tail(line) for line in tail_list)
    tail_last_non_health = None
    for line in reversed(tail_list):
        if "GET /health HTTP/1.1" not in line:
            tail_last_non_health = {
                "timestamp": _timestamp(line),
                "kind": _classify_tail(line),
                "line": line[-500:],
            }
            break

    tail_health_ratio = (
        tail_counts["health_check"] / len(tail_list) if tail_list else 0.0
    )
    post_reset_or_lease = post_counts["reset_500"] + post_counts["unknown_run_lease_id"]
    run_config_flags = _read_run_config_flags(run_dir)
    dynamic_sampling_sources = {
        "cli_dynamic_sampling_filter_path": "dynamic-sampling-filter-path" in flags,
        "run_config_dapo_dynamic_sampling": _truthy(run_config_flags.get("dapo_dynamic_sampling")),
        "run_config_dapo_dynamic_filter_path": bool(run_config_flags.get("dapo_dynamic_filter_path")),
    }
    dynamic_sampling_enabled = (
        _truthy(run_config_flags.get("dapo_dynamic_sampling"))
        or (
            "dapo_dynamic_sampling" not in run_config_flags
            and "dynamic-sampling-filter-path" in flags
        )
    )
    post_started = post_counts["start_terminal_rollout"] > 0
    no_next_batch = bool(last_rollout and (counts["data_rollout"] == len(rollouts)))
    similar_reasons: list[str] = []
    likelihood = "low"

    if dynamic_sampling_enabled:
        similar_reasons.append("dynamic sampling is enabled")
    if post_started:
        similar_reasons.append(
            f"{post_counts['start_terminal_rollout']} terminal rollouts started after the last completed batch"
        )
    if post_reset_or_lease:
        similar_reasons.append(
            f"{post_reset_or_lease} reset/lease errors after the last completed batch"
        )
    if tail_health_ratio >= 0.5:
        similar_reasons.append(
            f"tail health-check ratio is {tail_health_ratio:.2f}"
        )
    if last_rollout and last_train:
        train_rollout_id = last_train["metrics"].get("train/rollout_id")
        if train_rollout_id == last_rollout["id"]:
            similar_reasons.append("last train step consumed the last completed rollout; no later batch was logged")

    if dynamic_sampling_enabled and post_started and post_reset_or_lease >= 1000:
        likelihood = "high"
    elif dynamic_sampling_enabled and post_started and post_reset_or_lease >= 100:
        likelihood = "medium"
    elif dynamic_sampling_enabled and post_reset_or_lease >= 1000:
        likelihood = "medium"

    if not rollouts:
        likelihood = "unknown"
        similar_reasons.append("no data.py rollout metrics were parsed")

    gap_seconds = None
    if last_rollout is not None:
        t0 = _parse_time(last_rollout.get("timestamp"))
        t1 = _parse_time(last_timestamp)
        if t0 is not None and t1 is not None:
            gap_seconds = int((t1 - t0).total_seconds())

    return {
        "log_file": str(log_path),
        "log_size_bytes": log_path.stat().st_size,
        "line_count": line_count,
        "flags": flags,
        "run_config_flags": run_config_flags,
        "dynamic_sampling_sources": dynamic_sampling_sources,
        "dynamic_sampling_enabled": dynamic_sampling_enabled,
        "rollout_count": len(rollouts),
        "max_rollout_id": max((r["id"] for r in rollouts), default=None),
        "last_rollout": last_rollout,
        "train_step_count": len(train_steps),
        "last_train_step": last_train,
        "last_log_timestamp": last_timestamp,
        "seconds_from_last_rollout_to_log_end": gap_seconds,
        "counts_total": dict(counts),
        "counts_after_last_rollout": dict(post_counts),
        "generate_failed_top_urls": [
            {"url": url, "count": count}
            for url, count in generate_failed_by_url.most_common(20)
        ],
        "tail_lines_considered": len(tail_list),
        "tail_kind_counts": dict(tail_counts),
        "tail_health_ratio": tail_health_ratio,
        "tail_last_non_health": tail_last_non_health,
        "assessment": {
            "similar_dynamic_sampling_env_hang_likelihood": likelihood,
            "similar_dynamic_sampling_env_hang": likelihood in {"high", "medium"},
            "reasons": similar_reasons,
            "post_last_rollout_reset_or_lease_errors": post_reset_or_lease,
            "post_last_rollout_started_terminal_rollouts": post_started,
            "no_next_completed_batch_observed": no_next_batch,
        },
    }


def _write_report(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hang_diagnosis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )

    assessment = result["assessment"]
    last_rollout = result.get("last_rollout") or {}
    last_train = result.get("last_train_step") or {}
    total = Counter(result.get("counts_total") or {})
    post = Counter(result.get("counts_after_last_rollout") or {})
    lines = [
        "# Hang Diagnosis",
        "",
        f"- log_file: `{result['log_file']}`",
        f"- dynamic_sampling_enabled: `{result['dynamic_sampling_enabled']}`",
        f"- likelihood: `{assessment['similar_dynamic_sampling_env_hang_likelihood']}`",
        f"- rollout_count: `{result['rollout_count']}`; max_rollout_id: `{result['max_rollout_id']}`",
        f"- last_rollout: id=`{last_rollout.get('id')}` timestamp=`{last_rollout.get('timestamp')}` line=`{last_rollout.get('line')}`",
        f"- train_step_count: `{result['train_step_count']}`",
        f"- last_train_step: label=`{last_train.get('step_label')}` timestamp=`{last_train.get('timestamp')}` metrics=`{last_train.get('metrics')}`",
        f"- last_log_timestamp: `{result.get('last_log_timestamp')}`",
        f"- seconds_from_last_rollout_to_log_end: `{result.get('seconds_from_last_rollout_to_log_end')}`",
        "",
        "## Error Counts",
        "",
        f"- total reset_500: `{total['reset_500']}`",
        f"- total unknown_run_lease_id: `{total['unknown_run_lease_id']}`",
        f"- total heartbeat_500: `{total['heartbeat_500']}`",
        f"- total evaluate_500: `{total['evaluate_500']}`",
        f"- total generate_failed: `{total['generate_failed']}`",
        f"- after_last_rollout reset_500: `{post['reset_500']}`",
        f"- after_last_rollout unknown_run_lease_id: `{post['unknown_run_lease_id']}`",
        f"- after_last_rollout start_terminal_rollout: `{post['start_terminal_rollout']}`",
        f"- after_last_rollout terminal_rollout_finished: `{post['terminal_rollout_finished']}`",
        f"- tail_health_ratio: `{result['tail_health_ratio']:.3f}`",
        "",
        "## Assessment Reasons",
        "",
    ]
    for reason in assessment.get("reasons") or []:
        lines.append(f"- {reason}")
    if not assessment.get("reasons"):
        lines.append("- No strong hang signature found.")
    top_urls = result.get("generate_failed_top_urls") or []
    if top_urls:
        lines.extend(["", "## Generate Failed Top URLs", ""])
        for item in top_urls[:10]:
            lines.append(f"- `{item['count']}` `{item['url']}`")
    tail_non_health = result.get("tail_last_non_health")
    if tail_non_health:
        lines.extend(
            [
                "",
                "## Last Non-Health Tail Line",
                "",
                f"- kind: `{tail_non_health.get('kind')}`",
                f"- timestamp: `{tail_non_health.get('timestamp')}`",
                f"- line: `{tail_non_health.get('line')}`",
            ]
        )
    (out_dir / "hang_diagnosis.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--tail-lines", type=int, default=200)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    log_path = args.log_file or run_dir / "logs" / "train.log"
    out_dir = args.out_dir or run_dir / "metrics" / "analysis"
    if not log_path.exists():
        raise SystemExit(f"log not found: {log_path}")

    result = analyze(log_path, max(1, int(args.tail_lines)), run_dir=run_dir)
    result["run_dir"] = str(run_dir)
    _write_report(result, out_dir)
    assessment = result["assessment"]
    print(f"[+] wrote {out_dir / 'hang_diagnosis.json'}")
    print(f"[+] wrote {out_dir / 'hang_diagnosis.md'}")
    print(
        "[+] likelihood:",
        assessment["similar_dynamic_sampling_env_hang_likelihood"],
        "post_reset_or_lease:",
        assessment["post_last_rollout_reset_or_lease_errors"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
