#!/usr/bin/env python3
"""Generate dataset case-study reports from saved terminal-rl trajectories.

Inputs:
  * ``--run-dir`` pointing to one training run, or ``--traj-dir`` directly.
  * ``--config`` from ``select_case_study_samples.py``.

Outputs, by default under ``<run_dir>/case_study/``:
  * ``case_study_report.md``  - human-readable report
  * ``case_study_summary.json`` - structured summary
  * ``case_study_records.jsonl`` - one representative trajectory per line

The script only reads trajectory artifacts and never writes outside the selected
run output directory unless ``--out-dir`` explicitly points elsewhere.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required for this script: pip install pyyaml") from exc


DATASET_ALIASES = {
    "": "seta",
    "seta": "seta",
    "seta_env": "seta",
    "terminal_bench": "seta",
    "agent_safetybench": "agent_safetybench",
    "agent-safety-bench": "agent_safetybench",
    "asb": "agent_safetybench",
    "safety": "agent_safetybench",
    "agentharm": "agentharm",
    "agent_harm": "agentharm",
    "ah": "agentharm",
}

NEGATIVE_HINTS = (
    "error",
    "traceback",
    "exception",
    "not found",
    "command not found",
    "permission denied",
    "failed",
    "500",
    "timeout",
)


def repo_root_from_script() -> Path:
    env = os.getenv("OPENCLAW_RL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def normalize_dataset(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return DATASET_ALIASES.get(raw, raw.replace("-", "_") or "seta")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to read JSON {path}: {exc}") from exc


def load_config(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"case-study config not found: {path}") from exc
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [jsonable(x) for x in value]
    if isinstance(value, tuple):
        return [jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return str(value)


def compact(text: Any, limit: int) -> str:
    raw = str(text if text is not None else "")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 20)].rstrip() + "\n... [truncated]"


def one_line(text: Any, limit: int = 180) -> str:
    raw = " ".join(str(text if text is not None else "").split())
    return raw if len(raw) <= limit else raw[: limit - 3].rstrip() + "..."


def md_escape(text: Any) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def messages_to_text(value: Any) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                parts.append(f"{role}: {content}" if role else str(content or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value or "")


def sample_match_values(sample: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("id", "task_name", "task_path"):
        value = sample.get(key)
        if value is not None and str(value) != "":
            values.add(str(value))
    match = sample.get("match") if isinstance(sample.get("match"), dict) else {}
    for key in ("task_name", "task_path", "id", "id_original"):
        value = match.get(key)
        if value is not None and str(value) != "":
            values.add(str(value))
    return values


def trajectory_match_values(meta: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("task_name", "task_path", "uid"):
        value = meta.get(key)
        if value is not None and str(value) != "":
            values.add(str(value))
    sample_meta = meta.get("sample_metadata")
    if isinstance(sample_meta, dict):
        for key in ("task_name", "task_path", "id", "id_original", "name"):
            value = sample_meta.get(key)
            if value is not None and str(value) != "":
                values.add(str(value))
    reward_details = meta.get("reward_details")
    if isinstance(reward_details, dict):
        value = reward_details.get("task_name")
        if value:
            values.add(str(value))
    return values


def infer_dataset_from_meta(meta: dict[str, Any], dirname: str) -> str:
    for key in ("dataset_slug", "data_source", "dataset"):
        value = meta.get(key)
        if value:
            return normalize_dataset(value)
    sample_meta = meta.get("sample_metadata")
    if isinstance(sample_meta, dict) and sample_meta.get("data_source"):
        return normalize_dataset(sample_meta.get("data_source"))
    if dirname.startswith("agent_safetybench_"):
        return "agent_safetybench"
    if dirname.startswith("agentharm_") or dirname.startswith("tagentharm_"):
        return "agentharm"
    return "seta"


def index_records_by_dir(traj_dir: Path) -> dict[Path, dict[str, Any]]:
    index_path = traj_dir / "index.jsonl"
    if not index_path.exists():
        return {}
    active: dict[str, dict[str, Any]] = {}
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        rel_path = str(record.get("rel_path") or "")
        if not rel_path:
            continue
        event = str(record.get("event") or "save")
        if event == "delete":
            active.pop(rel_path, None)
        elif event == "save":
            active[rel_path] = record

    out: dict[Path, dict[str, Any]] = {}
    for rel_path, record in active.items():
        sub_dir = traj_dir / rel_path
        if sub_dir.is_dir() and (sub_dir / "traj.json").exists():
            out[sub_dir] = record
    return out


def scan_trajectories(traj_dir: Path) -> list[dict[str, Any]]:
    if not traj_dir.exists():
        raise FileNotFoundError(f"trajectory directory not found: {traj_dir}")
    records: list[dict[str, Any]] = []
    indexed = index_records_by_dir(traj_dir)
    candidate_dirs = list(indexed)
    seen_dirs = set(candidate_dirs)
    for sub_dir in sorted(traj_dir.iterdir()):
        if sub_dir.is_dir() and sub_dir not in seen_dirs:
            candidate_dirs.append(sub_dir)

    for sub_dir in candidate_dirs:
        meta_path = sub_dir / "meta.json"
        traj_path = sub_dir / "traj.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = read_json(meta_path)
            except ValueError:
                continue
        elif traj_path.exists():
            try:
                traj = read_json(traj_path)
                info = traj.get("info") if isinstance(traj.get("info"), dict) else {}
                reward = traj.get("reward") if isinstance(traj.get("reward"), dict) else {}
                meta = {**info, "reward_details": reward.get("details")}
            except ValueError:
                continue
        else:
            continue
        dataset = infer_dataset_from_meta(meta, sub_dir.name)
        records.append(
            {
                "dir": sub_dir,
                "meta_path": meta_path if meta_path.exists() else None,
                "traj_path": traj_path if traj_path.exists() else None,
                "meta": meta,
                "index": indexed.get(sub_dir),
                "dataset": dataset,
                "match_values": trajectory_match_values(meta),
            }
        )
    return records


def iter_config_samples(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = cfg.get("datasets") if isinstance(cfg.get("datasets"), dict) else {}
    out: list[dict[str, Any]] = []
    for dataset_key, info in datasets.items():
        dataset = normalize_dataset(dataset_key)
        samples = info.get("samples") if isinstance(info, dict) else []
        for sample in samples or []:
            if not isinstance(sample, dict):
                continue
            item = dict(sample)
            item["dataset"] = dataset
            item["config_dataset_key"] = dataset_key
            out.append(item)
    return out


def find_matches(sample: dict[str, Any], trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset = normalize_dataset(sample.get("dataset"))
    wanted = sample_match_values(sample)
    matches = []
    for record in trajectories:
        if record["dataset"] != dataset:
            continue
        if wanted & record["match_values"]:
            matches.append(record)
    return matches


def float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def record_train_key(record: dict[str, Any]) -> tuple[int, int, int]:
    meta = record["meta"]
    return (
        int(meta.get("train_step") if meta.get("train_step") is not None else -1),
        int(meta.get("rollout_id") if meta.get("rollout_id") is not None else -1),
        int(meta.get("sample_index") if meta.get("sample_index") is not None else -1),
    )


def record_reward_value(record: dict[str, Any]) -> float:
    meta = record["meta"]
    for key in ("total_reward", "raw_reward", "raw_score", "task_reward"):
        value = float_or_none(meta.get(key))
        if value is not None:
            return value
    return 0.0


def choose_representatives(matches: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if len(matches) <= limit:
        return sorted(matches, key=record_train_key)

    candidates = [
        max(matches, key=record_train_key),
        max(matches, key=record_reward_value),
        min(matches, key=record_reward_value),
    ]
    for record in sorted(matches, key=record_train_key, reverse=True):
        candidates.append(record)

    chosen: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for record in candidates:
        path = record["dir"]
        if path in seen:
            continue
        seen.add(path)
        chosen.append(record)
        if len(chosen) >= limit:
            break
    return chosen


def reward_badge(value: Any) -> str:
    val = float_or_none(value)
    if val is None:
        return "`n/a`"
    if val > 0:
        color, icon = "green", "✅"
    elif val < 0:
        color, icon = "red", "❌"
    else:
        color, icon = "gray", "○"
    return f'<span style="color:{color}">{val:+.3f} {icon}</span>'


def status_badge(status: Any, reward: Any) -> str:
    text = str(status or "")
    short = text.split(".")[-1]
    val = float_or_none(reward)
    if short == "COMPLETED" and (val is None or val >= 0):
        return f"✅ `{short}`"
    if short == "TRUNCATED":
        return f"🟡 `{short}`"
    if short == "FAILED":
        return f"❌ `{short}`"
    return f"`{short or 'unknown'}`"


def load_traj(record: dict[str, Any]) -> dict[str, Any]:
    path = record.get("traj_path")
    if isinstance(path, Path) and path.exists():
        return read_json(path)
    return {"info": record["meta"], "turns": [], "reward": {}}


def reward_dict(meta: dict[str, Any], traj: dict[str, Any]) -> dict[str, Any]:
    reward = traj.get("reward") if isinstance(traj.get("reward"), dict) else {}
    details = reward.get("details") if isinstance(reward.get("details"), dict) else None
    if details is None and isinstance(meta.get("reward_details"), dict):
        details = meta.get("reward_details")
    return {
        "raw_score": reward.get("raw_score", meta.get("raw_score")),
        "raw_reward": reward.get("raw_reward", meta.get("raw_reward")),
        "task_reward": reward.get("task_reward", meta.get("task_reward")),
        "safety_score": reward.get("safety_score"),
        "exploration_reward": reward.get("exploration_reward", meta.get("exploration_reward")),
        "total_reward": reward.get("total_reward", reward.get("score", meta.get("total_reward"))),
        "accuracy": reward.get("accuracy"),
        "details": details or {},
    }


def failure_reason(meta: dict[str, Any], rewards: dict[str, Any]) -> str:
    details = rewards.get("details") if isinstance(rewards.get("details"), dict) else {}
    for key in ("reason", "failure_reason", "mode"):
        value = details.get(key)
        if value:
            return str(value)
    if meta.get("eval_error"):
        return one_line(meta.get("eval_error"), 240)
    status = str(meta.get("status") or "")
    return status.split(".")[-1] if status else "n/a"


def extract_prompt(sample: dict[str, Any], meta: dict[str, Any], max_chars: int) -> str:
    if sample.get("prompt_preview"):
        return compact(sample.get("prompt_preview"), max_chars)
    if meta.get("instruction"):
        return compact(meta.get("instruction"), max_chars)
    if meta.get("sample_prompt") is not None:
        return compact(messages_to_text(meta.get("sample_prompt")), max_chars)
    sample_meta = meta.get("sample_metadata")
    if isinstance(sample_meta, dict) and sample_meta.get("instruction"):
        return compact(sample_meta.get("instruction"), max_chars)
    return ""


def tool_call_summary(tool_call: dict[str, Any], max_result_chars: int) -> dict[str, Any]:
    args = tool_call.get("args")
    result = tool_call.get("result")
    args_text = json.dumps(args, ensure_ascii=False, default=str) if not isinstance(args, str) else args
    result_text = json.dumps(result, ensure_ascii=False, default=str) if not isinstance(result, str) else result
    lower = result_text.lower()
    return {
        "tool_call_id": tool_call.get("tool_call_id"),
        "tool_name": tool_call.get("tool_name") or tool_call.get("name"),
        "args": compact(args_text, max_result_chars),
        "result": compact(result_text, max_result_chars),
        "is_error": any(hint in lower for hint in NEGATIVE_HINTS),
        "clawsentry": tool_call.get("clawsentry"),
    }


def trajectory_record(
    *,
    sample: dict[str, Any],
    record: dict[str, Any],
    max_text_chars: int,
    max_tool_result_chars: int,
) -> dict[str, Any]:
    meta = record["meta"]
    traj = load_traj(record)
    rewards = reward_dict(meta, traj)
    turns = traj.get("turns") if isinstance(traj.get("turns"), list) else []
    turn_summaries = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        tool_calls = [
            tool_call_summary(tc, max_tool_result_chars)
            for tc in (turn.get("tool_calls") or [])
            if isinstance(tc, dict)
        ]
        turn_summaries.append(
            {
                "turn_idx": turn.get("turn_idx"),
                "finish_reason": turn.get("finish_reason"),
                "latency_ms": turn.get("latency_ms"),
                "n_input_tokens": turn.get("n_input_tokens"),
                "n_output_tokens": turn.get("n_output_tokens"),
                "parse_error_recorded": turn.get("parse_error_recorded"),
                "assistant_output": compact(turn.get("assistant_output"), max_text_chars),
                "tool_calls": tool_calls,
                "uncertainty": turn.get("uncertainty"),
                "highlight": bool(turn.get("parse_error_recorded")) or any(tc["is_error"] for tc in tool_calls),
            }
        )

    return {
        "dataset": sample.get("dataset"),
        "sample_id": sample.get("id"),
        "sample_title": sample.get("title"),
        "selection_reason": sample.get("selection_reason"),
        "prompt": extract_prompt(sample, meta, max_text_chars),
        "trajectory_dir": str(record["dir"]),
        "trajectory_name": record["dir"].name,
        "task_name": meta.get("task_name"),
        "task_path": meta.get("task_path"),
        "uid": meta.get("uid"),
        "group_index": meta.get("group_index"),
        "sample_index": meta.get("sample_index"),
        "rollout_id": meta.get("rollout_id"),
        "train_step": meta.get("train_step"),
        "rollout_step": meta.get("rollout_step"),
        "status": meta.get("status"),
        "num_turns": len(turn_summaries) or meta.get("num_turns"),
        "rewards": rewards,
        "reason": failure_reason(meta, rewards),
        "eval_error": meta.get("eval_error"),
        "turns": turn_summaries,
    }


def render_record_markdown(record: dict[str, Any]) -> list[str]:
    rewards = record["rewards"]
    lines: list[str] = []
    lines.append(
        f"### Trajectory `{record['trajectory_name']}`"
    )
    lines.append("")
    lines.append(
        f"- Status: {status_badge(record.get('status'), rewards.get('total_reward'))}"
    )
    lines.append(
        f"- Step: train=`{record.get('train_step')}` rollout=`{record.get('rollout_id')}` "
        f"group=`{record.get('group_index')}` sample=`{record.get('sample_index')}`"
    )
    lines.append(f"- Reason: `{md_escape(record.get('reason'))}`")
    lines.append(
        "- Reward: "
        f"raw={reward_badge(rewards.get('raw_reward', rewards.get('raw_score')))} "
        f"task={reward_badge(rewards.get('task_reward'))} "
        f"safety={reward_badge(rewards.get('safety_score'))} "
        f"explore={reward_badge(rewards.get('exploration_reward'))} "
        f"total={reward_badge(rewards.get('total_reward'))}"
    )
    details = rewards.get("details") if isinstance(rewards.get("details"), dict) else {}
    if details:
        compact_details = json.dumps(details, ensure_ascii=False, indent=2, default=str)
        lines.append("")
        lines.append("<details><summary>Reward details</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(compact(compact_details, 2000))
        lines.append("```")
        lines.append("</details>")
    if record.get("eval_error"):
        lines.append("")
        lines.append("> Eval error: `" + md_escape(one_line(record.get("eval_error"), 300)) + "`")
    lines.append("")
    lines.append("#### Steps")
    lines.append("")
    for turn in record.get("turns") or []:
        marker = " ⚠️" if turn.get("highlight") else ""
        lines.append(
            f"**Step {turn.get('turn_idx')}**{marker} "
            f"finish=`{turn.get('finish_reason')}` "
            f"tokens={turn.get('n_input_tokens')}/{turn.get('n_output_tokens')}"
        )
        lines.append("")
        assistant_output = turn.get("assistant_output") or ""
        if assistant_output:
            lines.append("Assistant:")
            lines.append("```text")
            lines.append(assistant_output)
            lines.append("```")
        tool_calls = turn.get("tool_calls") or []
        if tool_calls:
            lines.append("Tool calls / observations:")
            for idx, tool in enumerate(tool_calls, start=1):
                icon = "❌" if tool.get("is_error") else "✅"
                lines.append(f"- {icon} `{tool.get('tool_name')}` call {idx}")
                lines.append("  - args:")
                lines.append("```json")
                lines.append(str(tool.get("args") or ""))
                lines.append("```")
                lines.append("  - observation:")
                lines.append("```text")
                lines.append(str(tool.get("result") or ""))
                lines.append("```")
        uncertainty = turn.get("uncertainty")
        if isinstance(uncertainty, dict) and uncertainty.get("available"):
            lines.append(
                f"Uncertainty: turn_score=`{uncertainty.get('turn_level_score')}` "
                f"turn_uncertainty=`{uncertainty.get('turn_level_uncertainty')}`"
            )
        lines.append("")
    return lines


def render_markdown(
    *,
    run_dir: Path | None,
    traj_dir: Path,
    cfg_path: Path,
    samples: list[dict[str, Any]],
    sample_results: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Dataset Case-Study Report")
    lines.append("")
    if run_dir:
        lines.append(f"- Run: `{run_dir}`")
    lines.append(f"- Trajectories: `{traj_dir}`")
    lines.append(f"- Config: `{cfg_path}`")
    lines.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| dataset | sample | matches | shown | latest step | latest total reward | reason |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    for result in sample_results:
        reps = result.get("representatives") or []
        latest = reps[0] if reps else {}
        rewards = latest.get("rewards") if isinstance(latest.get("rewards"), dict) else {}
        lines.append(
            f"| `{result['dataset']}` | `{md_escape(result['sample_id'])}` | "
            f"{result['matches_total']} | {len(reps)} | "
            f"{md_escape(latest.get('train_step', 'n/a'))} | "
            f"{reward_badge(rewards.get('total_reward')) if rewards else '`n/a`'} | "
            f"{md_escape(latest.get('reason', result.get('missing_reason', '')))} |"
        )
    lines.append("")

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for result in sample_results:
        by_dataset.setdefault(result["dataset"], []).append(result)

    for dataset in ("seta", "agent_safetybench", "agentharm"):
        if dataset not in by_dataset:
            continue
        lines.append(f"## {dataset}")
        lines.append("")
        for result in by_dataset[dataset]:
            sample = result["sample"]
            lines.append(f"### Case `{sample.get('id')}` - {sample.get('title', '')}")
            lines.append("")
            lines.append(f"- Selection: {sample.get('selection_reason', '')}")
            if result["matches_total"] == 0:
                lines.append(f"- ❌ No matching trajectory found: {result.get('missing_reason')}")
                lines.append("")
                continue
            prompt = result["representatives"][0].get("prompt") or sample.get("prompt_preview") or ""
            if prompt:
                lines.append("Prompt / task:")
                lines.append("```text")
                lines.append(prompt)
                lines.append("```")
                lines.append("")
            for rep in result["representatives"]:
                lines.extend(render_record_markdown(rep))
    return "\n".join(lines) + "\n"


def analyze(
    *,
    run_dir: Path | None,
    traj_dir: Path,
    config_path: Path,
    out_dir: Path,
    max_trajectories_per_sample: int,
    max_text_chars: int,
    max_tool_result_chars: int,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    samples = iter_config_samples(cfg)
    trajectories = scan_trajectories(traj_dir)
    sample_results: list[dict[str, Any]] = []
    jsonl_records: list[dict[str, Any]] = []
    for sample in samples:
        matches = find_matches(sample, trajectories)
        representatives = choose_representatives(matches, max_trajectories_per_sample)
        rep_payloads = [
            trajectory_record(
                sample=sample,
                record=record,
                max_text_chars=max_text_chars,
                max_tool_result_chars=max_tool_result_chars,
            )
            for record in representatives
        ]
        jsonl_records.extend(rep_payloads)
        sample_results.append(
            {
                "dataset": sample["dataset"],
                "sample_id": sample.get("id"),
                "sample": sample,
                "matches_total": len(matches),
                "representatives": rep_payloads,
                "missing_reason": (
                    "No trajectory matched this sample id/task_name/task_path in the selected run."
                    if not matches
                    else None
                ),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    report = render_markdown(
        run_dir=run_dir,
        traj_dir=traj_dir,
        cfg_path=config_path,
        samples=samples,
        sample_results=sample_results,
    )
    (out_dir / "case_study_report.md").write_text(report, encoding="utf-8")
    summary = {
        "schema": "openclaw.case_study_analysis.v1",
        "run_dir": str(run_dir) if run_dir else None,
        "traj_dir": str(traj_dir),
        "config": str(config_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_trajectories_scanned": len(trajectories),
        "n_config_samples": len(samples),
        "samples": sample_results,
    }
    (out_dir / "case_study_summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "case_study_records.jsonl").open("w", encoding="utf-8") as f:
        for record in jsonl_records:
            f.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--traj-dir", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "terminal-rl/scripts/case_study_samples.yaml",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-trajectories-per-sample", type=int, default=3)
    parser.add_argument("--max-text-chars", type=int, default=1600)
    parser.add_argument("--max-tool-result-chars", type=int, default=1200)
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    traj_dir = args.traj_dir.expanduser().resolve() if args.traj_dir else None
    if traj_dir is None:
        if run_dir is None:
            print("[ERROR] pass --run-dir or --traj-dir", file=sys.stderr)
            return 1
        traj_dir = run_dir / "trajectories"
    if run_dir is None and traj_dir.name == "trajectories":
        run_dir = traj_dir.parent
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else (run_dir / "case_study" if run_dir else traj_dir / "case_study")

    try:
        summary = analyze(
            run_dir=run_dir,
            traj_dir=traj_dir,
            config_path=args.config.expanduser().resolve(),
            out_dir=out_dir,
            max_trajectories_per_sample=max(1, args.max_trajectories_per_sample),
            max_text_chars=max(200, args.max_text_chars),
            max_tool_result_chars=max(200, args.max_tool_result_chars),
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    missing = [
        item for item in summary["samples"]
        if not item.get("matches_total")
    ]
    print(f"[+] scanned {summary['n_trajectories_scanned']} trajectories")
    print(f"[+] wrote {out_dir / 'case_study_report.md'}")
    print(f"[+] wrote {out_dir / 'case_study_summary.json'}")
    print(f"[+] wrote {out_dir / 'case_study_records.jsonl'}")
    if missing:
        print(f"[WARN] {len(missing)} configured samples had no matching trajectories")
        return 2 if args.fail_on_missing else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
