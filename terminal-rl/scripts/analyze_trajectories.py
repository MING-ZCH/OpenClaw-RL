#!/usr/bin/env python3
"""Classify rollout trajectories by execution status; produce JSON + Markdown.

Reusable across runs. Reads ``traj.json`` files from
  ``<run_dir>/trajectories/<dir>/traj.json``
and writes:
  <run_dir>/metrics/analysis/trajectory_classification.json
  <run_dir>/metrics/analysis/case_analysis.md

Usage:
  python -m terminal_rl.scripts.analyze_trajectories --run-dir <run_dir>
  # or directly:
  python terminal-rl/scripts/analyze_trajectories.py --run-dir runs/<run_id>

Optional:
  --traj-dir DIR     Override location of trajectory directories
  --out-dir DIR      Override output directory
  --samples-per-class N (default 5)
  --max-iter-hint N  Used only for human-readable description in REPORT (default 10)

Exit code 0 on success, 1 on missing trajectories.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CLASS_DESCRIPTIONS = {
    "pass": "✅ 测试全通过 (accuracy=1.0)",
    "fail_eval_normal": "完成但测试未通过 (Status.COMPLETED, acc<1)",
    "truncated": "🟡 触顶 max_iteration 被截断",
    "fail_eval_500": "🔴 evaluate 端点 500 (CPU worker 评测失败)",
    "fail_env_reset_500": "🔴 reset 端点 500 (CPU worker docker 故障)",
    "fail_env_exec": "🔴 exec/run 端点失败",
    "fail_other_infra": "🔴 其他环境层错误",
    "fail_no_error_msg": "❓ FAILED 但无 error 信息",
}

FAIL_CLASSES_WITH_ERROR_FREQ = ("fail_eval_500", "fail_env_reset_500", "fail_other_infra")


def classify_trajectory(d: dict[str, Any]) -> str:
    info = d.get("info", {})
    reward = d.get("reward", {})
    status = str(info.get("status", "")).split(".")[-1]
    eval_err = info.get("eval_error") or ""
    acc = reward.get("accuracy", 0.0)
    raw = reward.get("raw_score", 0.0)

    if status == "COMPLETED" and (acc == 1.0 or raw == 1.0):
        return "pass"
    if status == "COMPLETED":
        return "fail_eval_normal"
    if status == "TRUNCATED":
        return "truncated"
    if status == "FAILED":
        if "/evaluate" in eval_err and "500" in eval_err:
            return "fail_eval_500"
        if "/reset" in eval_err and "500" in eval_err:
            return "fail_env_reset_500"
        if "/run" in eval_err or "/exec" in eval_err:
            return "fail_env_exec"
        if eval_err:
            return "fail_other_infra"
        return "fail_no_error_msg"
    return f"other:{status}"


def build_record(sub_dir: Path, d: dict[str, Any], cls: str) -> dict[str, Any]:
    info = d.get("info", {})
    reward = d.get("reward", {})
    return {
        "dir": sub_dir.name,
        "task_id": info.get("task_id"),
        "task_name": info.get("task_name"),
        "task_path": info.get("task_path"),
        "dataset_slug": info.get("dataset_slug") or info.get("data_source"),
        "uid": info.get("uid"),
        "group_index": info.get("group_index"),
        "sample_index": info.get("sample_index"),
        "rollout_id": info.get("rollout_id"),
        "train_step": info.get("train_step"),
        "status": str(info.get("status", "")).split(".")[-1],
        "num_turns": info.get("num_turns"),
        "accuracy": reward.get("accuracy"),
        "raw_score": reward.get("raw_score"),
        "base_score": reward.get("base_score"),
        "score": reward.get("score"),
        "raw_reward": reward.get("raw_reward"),
        "task_reward": reward.get("task_reward"),
        "exploration_reward": reward.get("exploration_reward"),
        "total_reward": reward.get("total_reward"),
        "safety_score": reward.get("safety_score"),
        "trajectory_save_policy": info.get("trajectory_save_policy"),
        "trajectory_save_reason": info.get("trajectory_save_reason"),
        "eval_error_short": (info.get("eval_error") or "").split("\n")[0][:200],
        "class": cls,
    }


def format_score(v: Any) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def render_markdown(
    *,
    traj_dir: Path,
    n_total: int,
    by_class: dict[str, list],
    sample_by_class: dict[str, list],
    task_total: Counter,
    task_pass_count: Counter,
    tasks_with_any_pass: set,
    tasks_never_pass: set,
    step_total: Counter,
    tasks_with_multiple_steps: set,
    policy_counter: Counter,
    save_reason_counter: Counter,
    max_iter_hint: int,
) -> str:
    desc = dict(CLASS_DESCRIPTIONS)
    desc["truncated"] = f"🟡 触顶 max_iteration={max_iter_hint} 被截断"
    lines: list[str] = []
    lines.append("# 轨迹执行状态 Case 分析")
    lines.append("")
    lines.append(f"扫描目录：`{traj_dir}`")
    lines.append(f"总轨迹数：**{n_total}**")
    lines.append(f"涉及不同 task：**{len(task_total)}**")
    lines.append(
        f"至少通过一次测试的 task：**{len(tasks_with_any_pass)}** / {len(task_total)}"
    )
    lines.append(f"从未通过的 task：**{len(tasks_never_pass)}** / {len(task_total)}")
    lines.append(f"涉及不同 train_step/iter：**{len(step_total)}**")
    lines.append(
        f"保留了多个 iter 轨迹的 task：**{len(tasks_with_multiple_steps)}** / {len(task_total)}"
    )
    lines.append("")
    if step_total or policy_counter or save_reason_counter:
        lines.append("## 留存覆盖")
        lines.append("")
        if step_total:
            lines.append("**Top train_step 分布：**")
            lines.append("")
            for step, count in step_total.most_common(12):
                lines.append(f"- `{step}`: {count}")
            lines.append("")
        if policy_counter:
            lines.append("**Trajectory policy 分布：**")
            lines.append("")
            for policy, count in policy_counter.most_common():
                lines.append(f"- `{policy}`: {count}")
            lines.append("")
        if save_reason_counter:
            lines.append("**保存原因分布：**")
            lines.append("")
            for reason, count in save_reason_counter.most_common():
                lines.append(f"- `{reason}`: {count}")
            lines.append("")
    lines.append("## 分类统计")
    lines.append("")
    lines.append("| 类别 | 数量 | 占比 | 说明 |")
    lines.append("|---|---:|---:|---|")
    for cls in sorted(by_class, key=lambda x: -len(by_class[x])):
        cnt = len(by_class[cls])
        pct = (cnt / n_total * 100) if n_total else 0.0
        lines.append(f"| `{cls}` | {cnt} | {pct:.1f}% | {desc.get(cls, '?')} |")
    lines.append("")

    pass_recs = by_class.get("pass", [])
    lines.append("## ✅ 测试全部通过的轨迹（pass）")
    lines.append("")
    lines.append(
        f"共 **{len(pass_recs)}** 条，涉及 **{len({r['task_name'] for r in pass_recs})}** 个不同 task。"
    )
    lines.append("")
    if pass_recs:
        lines.append("### Top 通过样本（按 raw_score 排序）")
        lines.append("")
        lines.append(
            "| task | uid | group | sample | turns | raw_score | safety_score | score |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for r in sample_by_class.get("pass", []):
            lines.append(
                f"| {r['task_name']} | `{r['uid']}` | {r['group_index']} | {r['sample_index']} "
                f"| {r['num_turns']} | {format_score(r['raw_score'])} "
                f"| {format_score(r['safety_score'])} | {format_score(r['score'])} |"
            )
        lines.append("")
        lines.append("### 通过次数最多的 task（多个 sample 都通过 ⇒ 该 task 已掌握）")
        lines.append("")
        lines.append("| task | 通过次数 | 该 task 总采样次数 | 通过率 |")
        lines.append("|---|---:|---:|---:|")
        for t, c in task_pass_count.most_common(15):
            tot = task_total[t]
            rate = (c / tot * 100) if tot else 0
            lines.append(f"| {t} | {c} | {tot} | {rate:.0f}% |")
        lines.append("")
    else:
        lines.append("*没有任何轨迹通过测试。*")
        lines.append("")

    fail_order = [
        "fail_eval_normal",
        "truncated",
        "fail_eval_500",
        "fail_env_reset_500",
        "fail_env_exec",
        "fail_other_infra",
        "fail_no_error_msg",
    ]
    for cls in fail_order:
        recs = by_class.get(cls, [])
        if not recs:
            continue
        lines.append(f"## {desc.get(cls, cls)} (`{cls}`) — {len(recs)} 条")
        lines.append("")
        if cls in FAIL_CLASSES_WITH_ERROR_FREQ:
            err_freq = Counter(r["eval_error_short"] for r in recs)
            lines.append("**Top error 模式：**")
            lines.append("")
            for msg, c in err_freq.most_common(5):
                lines.append(f"- `[{c}×]` {msg}")
            lines.append("")
        lines.append("**示例：**")
        lines.append("")
        lines.append("| task | uid | turns | eval_error |")
        lines.append("|---|---|---:|---|")
        for r in recs[:5]:
            err = (r["eval_error_short"] or "").replace("|", "\\|")[:120]
            lines.append(
                f"| {r['task_name']} | `{r['uid']}` | {r['num_turns']} | {err} |"
            )
        lines.append("")

    if pass_recs and sample_by_class.get("pass"):
        top = sample_by_class["pass"][0]
        case_dir = traj_dir / top["dir"]
        try:
            case_data = json.loads((case_dir / "traj.json").read_text())
        except Exception:
            case_data = None
        if case_data is not None:
            lines.append("## 完整通过案例（首个）")
            lines.append("")
            lines.append(f"- 目录：`{case_dir.name}`")
            lines.append(f"- task：{top['task_name']}")
            lines.append(f"- 轮数：{top['num_turns']}")
            lines.append(
                f"- raw_score: {top['raw_score']}, safety_score: {top['safety_score']}"
            )
            lines.append("")
            lines.append("**逐轮 finish_reason / 输出字符数 / parse_error：**")
            lines.append("")
            lines.append(
                "| turn | finish_reason | n_in_tok | n_out_tok | parse_err | tool_calls |"
            )
            lines.append("|---:|---|---:|---:|---:|---:|")
            for t in case_data.get("turns", []):
                tc = t.get("tool_calls") or []
                lines.append(
                    f"| {t.get('turn_idx')} | {t.get('finish_reason')} "
                    f"| {t.get('n_input_tokens')} | {t.get('n_output_tokens')} "
                    f"| {t.get('parse_error_recorded')} | {len(tc)} |"
                )
            lines.append("")

    return "\n".join(lines)


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


def analyze(
    run_dir: Path,
    traj_dir: Path | None = None,
    out_dir: Path | None = None,
    samples_per_class: int = 5,
    max_iter_hint: int = 10,
) -> dict[str, Any]:
    traj_dir = traj_dir or (run_dir / "trajectories")
    out_dir = out_dir or (run_dir / "metrics" / "analysis")

    if not traj_dir.is_dir():
        raise FileNotFoundError(f"trajectories directory not found: {traj_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    by_class: dict[str, list] = defaultdict(list)
    all_records: list[dict[str, Any]] = []
    err_counter: Counter = Counter()

    print(f"[+] scanning {traj_dir}")
    n = 0
    indexed = index_records_by_dir(traj_dir)
    candidate_dirs = list(indexed)
    seen_dirs = set(candidate_dirs)
    for sub in sorted(traj_dir.iterdir()):
        if sub.is_dir() and sub not in seen_dirs:
            candidate_dirs.append(sub)
    for sub in candidate_dirs:
        tj = sub / "traj.json"
        if not tj.exists():
            continue
        n += 1
        try:
            d = json.loads(tj.read_text())
        except Exception as e:
            err_counter[f"parse:{type(e).__name__}"] += 1
            continue
        cls = classify_trajectory(d)
        rec = build_record(sub, d, cls)
        by_class[cls].append(rec)
        all_records.append(rec)

    print(f"[+] total trajectories: {n}")
    print("[+] class distribution:")
    for k in sorted(by_class, key=lambda x: -len(by_class[x])):
        print(f"  {k:24s}  {len(by_class[k]):5d}")

    # Per-class samples
    sample_by_class: dict[str, list] = {}
    for cls, recs in by_class.items():
        if cls == "pass":
            sorted_recs = sorted(recs, key=lambda r: -(r.get("raw_score") or 0))
        else:
            sorted_recs = recs
        sample_by_class[cls] = sorted_recs[:samples_per_class]

    task_pass_count: Counter = Counter()
    task_total: Counter = Counter()
    step_total: Counter = Counter()
    task_step_total: Counter = Counter()
    task_steps: dict[str, set[str]] = defaultdict(set)
    policy_counter: Counter = Counter()
    save_reason_counter: Counter = Counter()
    for rec in all_records:
        t = rec["task_name"]
        task_total[t] += 1
        if rec["class"] == "pass":
            task_pass_count[t] += 1
        step = rec.get("train_step")
        step_key = str(step if step is not None else "na")
        step_total[step_key] += 1
        task_step_total[(t, step_key)] += 1
        task_steps[t].add(step_key)
        if rec.get("trajectory_save_policy"):
            policy_counter[str(rec["trajectory_save_policy"])] += 1
        if rec.get("trajectory_save_reason"):
            save_reason_counter[str(rec["trajectory_save_reason"])] += 1
    tasks_with_any_pass = {t for t in task_total if task_pass_count[t] > 0}
    tasks_never_pass = {t for t in task_total if task_pass_count[t] == 0}
    tasks_with_multiple_steps = {
        task for task, steps in task_steps.items()
        if len({step for step in steps if step != "na"}) > 1
    }

    report = {
        "run_dir": str(run_dir),
        "traj_dir": str(traj_dir),
        "n_trajectories": n,
        "class_distribution": {k: len(v) for k, v in by_class.items()},
        "samples_per_class": sample_by_class,
        "n_unique_tasks": len(task_total),
        "n_unique_train_steps": len(step_total),
        "n_tasks_with_multiple_train_steps": len(tasks_with_multiple_steps),
        "n_tasks_with_at_least_one_pass": len(tasks_with_any_pass),
        "n_tasks_never_passed": len(tasks_never_pass),
        "top_passed_tasks": task_pass_count.most_common(20),
        "top_train_steps": step_total.most_common(30),
        "top_task_step_cells": [
            {"task_name": task, "train_step": step, "count": count}
            for (task, step), count in task_step_total.most_common(50)
        ],
        "trajectory_save_policy_distribution": dict(policy_counter),
        "trajectory_save_reason_distribution": dict(save_reason_counter),
        "parse_errors": dict(err_counter),
    }

    json_path = out_dir / "trajectory_classification.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[+] wrote {json_path}")

    md = render_markdown(
        traj_dir=traj_dir,
        n_total=n,
        by_class=by_class,
        sample_by_class=sample_by_class,
        task_total=task_total,
        task_pass_count=task_pass_count,
        tasks_with_any_pass=tasks_with_any_pass,
        tasks_never_pass=tasks_never_pass,
        step_total=step_total,
        tasks_with_multiple_steps=tasks_with_multiple_steps,
        policy_counter=policy_counter,
        save_reason_counter=save_reason_counter,
        max_iter_hint=max_iter_hint,
    )
    md_path = out_dir / "case_analysis.md"
    md_path.write_text(md)
    print(f"[+] wrote {md_path}")

    print()
    print("=" * 60)
    for cls in [
        "pass", "fail_eval_normal", "truncated",
        "fail_eval_500", "fail_env_reset_500", "fail_other_infra",
        "fail_no_error_msg",
    ]:
        print(f"  {cls:24s}  {len(by_class.get(cls, [])):5d}")
    print(f"  unique tasks seen   {len(task_total):5d}")
    print(f"  tasks ever passed   {len(tasks_with_any_pass):5d}")
    print(f"  unique train steps  {len(step_total):5d}")
    print(f"  multi-step tasks    {len(tasks_with_multiple_steps):5d}")

    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Run root, e.g. runs/<run_id>")
    p.add_argument("--traj-dir", type=Path, default=None,
                   help="Override trajectory dir (default: <run_dir>/trajectories)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output dir (default: <run_dir>/metrics/analysis)")
    p.add_argument("--samples-per-class", type=int, default=5)
    p.add_argument("--max-iter-hint", type=int, default=10,
                   help="Used in markdown table for 'truncated' description")
    args = p.parse_args(argv)

    try:
        analyze(
            run_dir=args.run_dir.resolve(),
            traj_dir=args.traj_dir.resolve() if args.traj_dir else None,
            out_dir=args.out_dir.resolve() if args.out_dir else None,
            samples_per_class=args.samples_per_class,
            max_iter_hint=args.max_iter_hint,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
