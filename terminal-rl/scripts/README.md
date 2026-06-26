# terminal-rl/scripts/

Reusable analysis tools for terminal-rl training runs. All scripts accept
`--run-dir <path>` and read/write under that directory.

## analyze_trajectories.py

Classifies per-rollout trajectories saved at `<run_dir>/trajectories/<dir>/traj.json`.

```bash
python terminal-rl/scripts/analyze_trajectories.py --run-dir runs/<run_id>
```

Outputs:
- `<run_dir>/metrics/analysis/trajectory_classification.json` — counts per class,
  sample records, task-level pass rates
- `<run_dir>/metrics/analysis/case_analysis.md` — human-readable Markdown report

Classes: `pass`, `fail_eval_normal`, `truncated`, `fail_eval_500`,
`fail_env_reset_500`, `fail_env_exec`, `fail_other_infra`, `fail_no_error_msg`.

Options:
- `--traj-dir DIR` override trajectory directory
- `--out-dir DIR` override output directory
- `--samples-per-class N` (default 5)
- `--max-iter-hint N` only used in the markdown table header (default 10)

## plot_training_metrics.py

Parses `<run_dir>/logs/train.log` plus `<run_dir>/logs/metrics.jsonl` when
available, then emits curves + summary.

```bash
python terminal-rl/scripts/plot_training_metrics.py --run-dir runs/<run_id>
```

Outputs:
- `<run_dir>/metrics/analysis/summary_stats.json` — aggregated metrics
- `<run_dir>/metrics/analysis/figs/{overview,reward_curve,response_length,loss_curve,grad_norm,kl_entropy}.png`

Key features:
- Detects mode-collapse (response_length drops below threshold after rollout 5)
- Splits overview reward panels by dataset and reward type using
  `TERMINAL_RL_METRIC_JSON` records: `raw_reward`, `exploration_reward`,
  `total_reward`, reward std, and sample counts. Legacy logs without structured
  fields fall back to the old aggregate rollout curves.
- Recovers `agent_safetybench` / `agentharm` / `seta` splits from legacy
  `dataset reward breakdown` text tables when old structured logs only stored
  collapsed `security` records.
- Breaks sparse dataset curves at missing rollout ranges instead of connecting
  distant points with long straight lines.
- Plots KL on a separate y-axis from entropy; when `train/kl_loss` is absent
  because KL loss is disabled, falls back to the logged `train/ppo_kl`.
- Plots `truncated_fraction` as `truncated / sample_count` by dataset instead
  of mixing legacy global fractions with structured truncated counts.
- Counts `/reset 500` events bucketed per minute (signals CPU worker docker failure)
- Counts ClawSentry pre_action fail-open events (rate-limit / offline)

Options:
- `--log-file PATH` override (default `<run_dir>/logs/train.log`)
- `--out-dir DIR` override (default `<run_dir>/metrics/analysis`)
- `--no-figs` skip image generation

## analyze_hang_diagnostics.py

Parses `<run_dir>/logs/train.log` and checks whether the tail has the same
signature as a DAPO dynamic-sampling/env-reset stall: last completed rollout is
followed by more terminal rollout starts, repeated `/reset 500` or
`Unknown run_lease_id`, and no next completed batch.

```bash
python terminal-rl/scripts/analyze_hang_diagnostics.py --run-dir runs/<run_id>
```

Outputs:
- `<run_dir>/metrics/analysis/hang_diagnosis.json` — machine-readable counts
  and assessment
- `<run_dir>/metrics/analysis/hang_diagnosis.md` — compact human-readable
  report

Options:
- `--log-file PATH` override (default `<run_dir>/logs/train.log`)
- `--out-dir DIR` override (default `<run_dir>/metrics/analysis`)
- `--tail-lines N` number of final log lines to classify (default 200)

## Case-study tools

These tools keep a fixed set of representative SetA / agent-safety-bench /
AgentHarm samples and compare saved trajectories for those samples across runs.
They use only Python stdlib plus PyYAML, which is already used by the training
launch scripts.

### select_case_study_samples.py

Builds `case_study_samples.yaml` from the converted JSONL datasets.

```bash
python terminal-rl/scripts/select_case_study_samples.py \
  --output terminal-rl/scripts/case_study_samples.yaml
```

Manual IDs override the default selection for a dataset:

```bash
python terminal-rl/scripts/select_case_study_samples.py \
  --seta-id 661 --seta-id 1072 \
  --asb-id 0,2,17 \
  --agentharm-id agentharm_harmful_test_public_1-1 \
  --output /tmp/case_study_samples.yaml
```

To add a new fixed sample, edit `terminal-rl/scripts/case_study_samples.yaml`
and append an item under `datasets.<dataset>.samples`; at minimum set `id`,
`task_name`, `task_path`, and a short `selection_reason`.

### analyze_case_study.py

Reads one run's `trajectories/` and writes the report under
`<run_dir>/case_study/`.

```bash
python terminal-rl/scripts/analyze_case_study.py \
  --run-dir runs/<run_id> \
  --config terminal-rl/scripts/case_study_samples.yaml
```

Outputs:
- `<run_dir>/case_study/case_study_report.md`
- `<run_dir>/case_study/case_study_summary.json`
- `<run_dir>/case_study/case_study_records.jsonl`

The Markdown report contains prompt/task text, per-step assistant actions, tool
calls, observations, reward breakdown, status, failure reason, and uncertainty
fields when present.

### compare_case_study.py

Compares latest matching trajectories for each fixed sample across runs.

```bash
python terminal-rl/scripts/compare_case_study.py \
  --run-dir runs/run_a runs/run_b runs/run_c \
  --config terminal-rl/scripts/case_study_samples.yaml
```

Outputs by default under the first run's `case_study/` directory:
- `case_study_compare.md`
- `case_study_compare.csv`
- `case_study_compare.json`

### run_case_study.sh

One-command wrapper for a single run:

```bash
bash terminal-rl/scripts/run_case_study.sh runs/<run_id>
```

Environment overrides:
- `CASE_STUDY_CONFIG=/path/to/case_study_samples.yaml`
- `CASE_STUDY_MAX_TRAJ_PER_SAMPLE=3`
- `CASE_STUDY_MAX_TEXT_CHARS=1600`
- `CASE_STUDY_MAX_TOOL_RESULT_CHARS=1200`

Training integration is optional: set `CASE_STUDY_ON_EXIT=1` in the training
launch environment to run this wrapper after a successful Ray job. Set
`CASE_STUDY_ON_FAILURE=1` to also analyze partial trajectories after a failed job.

## Typical workflow

```bash
RUN=runs/terminal-rl_qwen3-8b_8gpu_2026-05-21_124958
python terminal-rl/scripts/plot_training_metrics.py --run-dir $RUN
python terminal-rl/scripts/analyze_trajectories.py --run-dir $RUN
python terminal-rl/scripts/analyze_hang_diagnostics.py --run-dir $RUN
bash terminal-rl/scripts/run_case_study.sh $RUN
ls $RUN/metrics/analysis/
# case_analysis.md  figs/  hang_diagnosis.json  hang_diagnosis.md
# summary_stats.json  trajectory_classification.json
ls $RUN/case_study/
# case_study_report.md  case_study_summary.json  case_study_records.jsonl
```
