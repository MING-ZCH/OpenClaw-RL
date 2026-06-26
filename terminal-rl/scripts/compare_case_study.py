#!/usr/bin/env python3
"""Compare fixed case-study samples across multiple terminal-rl runs.

For each configured case-study sample, the script picks the latest matching
trajectory from each run and writes a compact Markdown + CSV comparison table.

Default output:
  <first_run_dir>/case_study/case_study_compare.md
  <first_run_dir>/case_study/case_study_compare.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import analyze_case_study as acs


def latest_record_for_sample(
    sample: dict[str, Any],
    trajectories: list[dict[str, Any]],
    max_text_chars: int,
) -> dict[str, Any] | None:
    matches = acs.find_matches(sample, trajectories)
    reps = acs.choose_representatives(matches, 1)
    if not reps:
        return None
    return acs.trajectory_record(
        sample=sample,
        record=reps[0],
        max_text_chars=max_text_chars,
        max_tool_result_chars=600,
    )


def row_from_record(run_dir: Path, sample: dict[str, Any], record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {
            "dataset": sample.get("dataset"),
            "sample_id": sample.get("id"),
            "run": run_dir.name,
            "train_step": "",
            "rollout_id": "",
            "status": "missing",
            "raw_reward": "",
            "task_reward": "",
            "total_reward": "",
            "reason": "no matching trajectory",
            "num_turns": "",
            "trajectory": "",
        }
    rewards = record.get("rewards") if isinstance(record.get("rewards"), dict) else {}
    return {
        "dataset": sample.get("dataset"),
        "sample_id": sample.get("id"),
        "run": run_dir.name,
        "train_step": record.get("train_step"),
        "rollout_id": record.get("rollout_id"),
        "status": str(record.get("status") or "").split(".")[-1],
        "raw_reward": rewards.get("raw_reward", rewards.get("raw_score")),
        "task_reward": rewards.get("task_reward"),
        "total_reward": rewards.get("total_reward"),
        "reason": record.get("reason"),
        "num_turns": record.get("num_turns"),
        "trajectory": record.get("trajectory_name"),
    }


def md_escape(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def render_markdown(rows: list[dict[str, Any]], run_dirs: list[Path], config_path: Path) -> str:
    lines: list[str] = []
    lines.append("# Case-Study Run Comparison")
    lines.append("")
    lines.append(f"- Config: `{config_path}`")
    lines.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("- Runs:")
    for run_dir in run_dirs:
        lines.append(f"  - `{run_dir}`")
    lines.append("")
    lines.append("| dataset | sample | run | step | status | raw | task | total | turns | reason |")
    lines.append("|---|---|---|---:|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| `{md_escape(row['dataset'])}` | `{md_escape(row['sample_id'])}` | "
            f"`{md_escape(row['run'])}` | {md_escape(row['train_step'])} | "
            f"{md_escape(row['status'])} | {fmt(row['raw_reward'])} | "
            f"{fmt(row['task_reward'])} | {fmt(row['total_reward'])} | "
            f"{md_escape(row['num_turns'])} | {md_escape(row['reason'])} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Each cell uses the latest matching trajectory for that sample in the run.")
    lines.append("- `missing` means the configured sample was not sampled or its trajectory was not saved in that run.")
    lines.append("- For full step-by-step behavior, run `analyze_case_study.py` on each run and open `case_study_report.md`.")
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def parse_args() -> argparse.Namespace:
    root = acs.repo_root_from_script()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "terminal-rl/scripts/case_study_samples.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <first_run_dir>/case_study",
    )
    parser.add_argument("--max-text-chars", type=int, default=800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = [path.expanduser().resolve() for path in args.run_dir]
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"[ERROR] run dir not found: {run_dir}", file=sys.stderr)
            return 1
    config_path = args.config.expanduser().resolve()
    try:
        cfg = acs.load_config(config_path)
        samples = acs.iter_config_samples(cfg)
    except Exception as exc:
        print(f"[ERROR] failed to load config: {exc}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        traj_dir = run_dir / "trajectories"
        try:
            trajectories = acs.scan_trajectories(traj_dir)
        except Exception as exc:
            print(f"[WARN] failed to scan {traj_dir}: {exc}", file=sys.stderr)
            trajectories = []
        for sample in samples:
            record = latest_record_for_sample(sample, trajectories, max(200, args.max_text_chars))
            rows.append(row_from_record(run_dir, sample, record))

    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else run_dirs[0] / "case_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "case_study_compare.md"
    csv_path = out_dir / "case_study_compare.csv"
    json_path = out_dir / "case_study_compare.json"

    md_path.write_text(render_markdown(rows, run_dirs, config_path), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "schema": "openclaw.case_study_compare.v1",
                "config": str(config_path),
                "runs": [str(path) for path in run_dirs],
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[+] wrote {md_path}")
    print(f"[+] wrote {csv_path}")
    print(f"[+] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
