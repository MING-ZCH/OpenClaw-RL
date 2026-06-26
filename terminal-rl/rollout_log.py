from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import wandb
from slime.utils import logging_utils
from slime.utils.types import Sample
from slime.ray.rollout import compute_rollout_step

logger = logging.getLogger(__name__)

_METRIC_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_REWARD_COMPONENT_KEYS = (
    "raw_score",
    "base_score",
    "prm_turn_score",
    "safety_score",
    "explore_intrinsic",
    "explore_intrinsic_scaled",
    "explore_intrinsic_in_total",
    "explore_intrinsic_effective_coef",
    "explore_intrinsic_schedule_multiplier",
    "explore_safety_penalty",
    "explore_lprnd",
    "explore_lprnd_raw",
    "explore_lprnd_effective_coef",
    "explore_lprnd_schedule_multiplier",
    "explore_agent57_enabled",
    "explore_agent57_arm_id",
    "explore_agent57_k",
    "explore_agent57_beta",
    "explore_agent57_episodic_backend",
    "explore_agent57_max_bonus",
    "explore_agent57_ucb_c",
    "explore_agent57_ucb_window",
    "explore_agent57_ucb_epsilon",
    "explore_agent57_ucb_min_per_arm",
    "explore_agent57_ucb_dataset_aware",
    "explore_agent57_ucb_random_seed",
    "explore_agent57_lifelong_bonus_unclipped",
    "explore_agent57_lifelong_enabled",
    "explore_agent57_lifelong_coef",
    "explore_agent57_lifelong_clip",
    "explore_agent57_lifelong_include_dataset",
    "explore_agent57_lifelong_include_task",
    "explore_agent57_lifelong_include_turn",
    "explore_agent57_lifelong_raw",
    "explore_agent57_lifelong_bonus",
    "explore_agent57_lifelong_unique_keys",
    "explore_agent57_lifelong_seen_before",
    "explore_agent57_lifelong_warmup_remaining",
    "explore_agent57_lifelong_eligible",
    "explore_agent57_episodic_action_count",
    "explore_agent57_episodic_empty_bucket_count",
    "explore_agent57_episodic_empty_bucket_rate",
    "explore_agent57_episodic_exact_repeat_count",
    "explore_agent57_episodic_candidate_count_mean",
    "explore_agent57_episodic_probe_count_mean",
    "explore_agent57_ngu_mod_clip",
    "explore_agent57_ngu_episodic",
    "explore_agent57_ngu_life_mod",
    "explore_agent57_intrinsic_signal",
    "explore_agent57_ngu_bonus",
    "explore_agent57_ngu_bonus_unclipped",
    "explore_agent57_bonus_unclipped",
    "explore_agent57_bonus_clipped",
    "explore_cde_actor_bonus",
    "explore_cde_actor_log_ppl",
    "explore_cde_actor_eligible",
    "explore_cde_actor_base_mean",
    "explore_cde_actor_cap",
    "explore_cde_actor_scaled",
    "explore_cde_actor_clipped",
    "explore_cde_actor_omega",
    "explore_cde_actor_base_magnitude",
    "explore_post_norm_bonus_raw",
    "explore_post_norm_bonus",
    "explore_total_bonus",
    "explore_all_bonus",
    "explore_score_bonus",
    "explore_base_score_before_bonus",
    "explore_bonus_to_base_abs_ratio",
    "explore_curiosity_pressure",
    "explore_tool_intrinsic_pressure",
    "explore_safety_pressure",
    "explore_mood_code",
    "explore_turn_count",
    "explore_tool_call_count",
    "explore_action_count",
    "explore_danger_command_count",
    "explore_parse_error_count",
)
_REWARD_DETAIL_NUMERIC_KEYS = (
    "base",
    "n_tool_calls",
    "tool_successes",
    "n_turns",
    "parse_errors",
    "response_words",
    "progress",
    "progress_adjust",
    "turn_penalty",
    "parse_penalty",
    "truncate_penalty",
    "unsafe_tool_penalty",
    "tool_success_bonus",
    "warning_bonus",
    "refusal_quality_bonus",
    "safe_completion_bonus",
    "concise_bonus",
    "concise_refusal_bonus",
)
_STRUCTURED_LOG_PREFIX = "TERMINAL_RL_METRIC_JSON"
_STRUCTURED_SCHEMA = "terminal_rl.per_dataset_metrics.v1"
_STRUCTURED_SCHEMA_VERSION = 7
_LAST_EVAL_BY_DATASET: dict[str, dict[str, Any]] = {}
_METRIC_SEMANTICS_LOGGED = False
_COMPACT_EXACT_KEYS = {
    "rollout/step",
    "eval/step",
    "terminal/total_samples",
    "terminal/completed",
    "terminal/truncated",
    "terminal/failed",
    "terminal/aborted",
    "terminal/failed_ratio",
    "terminal/non_trainable_ratio",
    "terminal/reward_mean",
    "terminal/reward_std",
    "terminal/reward_min",
    "terminal/reward_max",
    "terminal/accuracy",
    "terminal/pass_rate",
    "terminal/train_batch_pass_rate",
    "terminal/safety_score_mean",
    "terminal/safety_negative_ratio",
    "terminal/clawsentry_error_rate",
    "terminal/rollout_time",
    "terminal/task/unique_count",
    "terminal/task/top_ratio",
    "terminal/trajectory/considered_count",
    "terminal/trajectory/saved_count",
    "terminal/trajectory/save_rate",
    "terminal/trajectory/considered_unique_tasks",
    "terminal/trajectory/saved_unique_tasks",
    "terminal/explore/explore_intrinsic/mean",
    "terminal/explore/explore_intrinsic/p90",
    "terminal/explore/explore_intrinsic_scaled/mean",
    "terminal/explore/explore_intrinsic_scaled/p90",
    "terminal/explore/explore_total_bonus/mean",
    "terminal/explore/explore_bonus_to_base_abs_ratio/mean",
    "terminal/explore/explore_curiosity_pressure/mean",
    "terminal/explore/explore_safety_pressure/mean",
    "terminal/explore/explore_reward_hacking_risk_rate",
    "terminal/explore/explore_over_exploration_risk_rate",
    "terminal/explore/explore_safety_tension_rate",
    "terminal/explore/agent57/active_rate",
    "terminal/explore/agent57/lifelong_enabled_rate",
    "terminal/explore/agent57/lifelong_eligible_rate",
    "terminal/explore/agent57/lifelong_state_error_rate",
    "terminal/explore/agent57/arm_count",
    "terminal/explore/agent57/top_arm",
    "terminal/explore/agent57/top_arm_ratio",
    "terminal/explore/agent57/lifelong_raw/mean",
    "terminal/explore/agent57/lifelong_bonus/mean",
    "terminal/explore/agent57/ngu_episodic/mean",
    "terminal/explore/agent57/ngu_life_mod/mean",
    "terminal/explore/agent57/intrinsic_signal/mean",
    "terminal/explore/agent57/intrinsic_signal/p90",
    "terminal/explore/agent57/episodic_empty_bucket_rate/mean",
    "terminal/explore/agent57/episodic_exact_repeat_count/mean",
    "terminal/explore/agent57/ngu_bonus/mean",
    "terminal/explore/agent57/bonus_clipped/mean",
    "terminal/turn_uncertainty/mean_neg_logprob/mean",
    "terminal/turn_uncertainty/mean_score/mean",
    "terminal/turn_uncertainty/low_progress_turn_ratio",
}
_COMPACT_PER_DATASET_SUFFIXES = {
    "sample_count",
    "trainable_count",
    "reward/total",
    "reward/task",
    "reward/raw",
    "reward/exploration",
    "reward/exploration_abs",
    "reward/exploration_score",
    "reward/exploration_signal",
    "reward/exploration_post_norm",
    "reward/exploration_post_norm_abs",
    "reward/exploration_ratio",
    "reward/exploration_abs_to_task_ratio",
    "total_reward",
    "task_reward",
    "raw_reward",
    "exploration_reward",
    "exploration_reward_abs",
    "exploration_reward_score",
    "exploration_reward_signal",
    "exploration_reward_post_norm",
    "exploration_reward_post_norm_abs",
    "truncated_fraction",
    "pass_rate",
    "unit_test_pass_rate",
    "test_acc",
    "reward_std",
    "response_length",
    "kl",
    "entropy",
    "turn_uncertainty/mean_neg_logprob",
    "turn_uncertainty/mean_score",
    "turn_uncertainty/mean_abs_score_delta",
    "turn_uncertainty/low_progress_fraction",
    "agent57/active",
    "agent57/lifelong_enabled",
    "agent57/lifelong_bonus",
    "agent57/lifelong_raw",
    "agent57/lifelong_unique_keys",
    "agent57/lifelong_seen_before",
    "agent57/lifelong_warmup_remaining",
    "agent57/lifelong_eligible_rate",
    "agent57/lifelong_state_error_rate",
    "agent57/intrinsic_signal",
    "agent57/episodic_empty_bucket_rate",
    "agent57/episodic_exact_repeat_count",
    "agent57/arm_count",
    "agent57/top_arm",
    "agent57/top_arm_ratio",
    "agent57/top_suppressed_ratio",
    "task/unique_count",
    "task/top_ratio",
    "trajectory/considered_count",
    "trajectory/saved_count",
    "trajectory/save_rate",
    "trajectory/considered_unique_tasks",
    "trajectory/saved_unique_tasks",
}


def _ensure_terminal_step_metric(args) -> None:
    if not getattr(args, "use_wandb", False):
        return
    try:
        wandb.define_metric("terminal/*", step_metric="rollout/step")
        wandb.define_metric("per_dataset/*", step_metric="rollout/step")
    except Exception as e:
        logger.warning("Failed to define wandb step metric for terminal/*: %s", e)


def _wandb_metric_profile() -> str:
    return str(os.getenv("TERMINAL_WANDB_METRIC_PROFILE", "full")).strip().lower()


def _compact_per_dataset_suffix(metric_key: str) -> str | None:
    prefix = "per_dataset/"
    if not metric_key.startswith(prefix):
        return None
    parts = metric_key[len(prefix):].split("/", 1)
    if len(parts) != 2 or not parts[0]:
        return None
    return parts[1]


def _filter_wandb_metrics(log_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Keep wandb dashboards low-cardinality by default.

    The detailed rollout diagnostics are still emitted to structured JSONL and
    train logs. This filter only controls the wandb payload.
    """
    profile = _wandb_metric_profile()
    if profile in {"full", "legacy", "all", "verbose"}:
        return log_dict
    if profile not in {"compact", "minimal", "key", "keys"}:
        logger.warning(
            "Unknown TERMINAL_WANDB_METRIC_PROFILE=%r; using compact wandb metrics",
            profile,
        )

    filtered: Dict[str, Any] = {}
    for key, value in log_dict.items():
        if key in _COMPACT_EXACT_KEYS:
            filtered[key] = value
            continue
        per_dataset_suffix = _compact_per_dataset_suffix(key)
        if per_dataset_suffix in _COMPACT_PER_DATASET_SUFFIXES:
            filtered[key] = value
    dropped = len(log_dict) - len(filtered)
    if dropped > 0 and _env_enabled("TERMINAL_WANDB_FILTER_LOG_ONCE", "1"):
        logger.info(
            "wandb metric profile=%s kept=%d dropped=%d; detailed metrics remain in structured JSONL/log tables",
            profile or "compact",
            len(filtered),
            dropped,
        )
        os.environ["TERMINAL_WANDB_FILTER_LOG_ONCE"] = "0"
    return filtered


def _log_metric_semantics_once() -> None:
    global _METRIC_SEMANTICS_LOGGED
    if _METRIC_SEMANTICS_LOGGED:
        return
    _METRIC_SEMANTICS_LOGGED = True
    logger.info(
        "metric semantics: test_acc is a legacy alias for pass_rate. "
        "For train rollouts it is the current sampled task/unit-test pass rate, "
        "not the SETA held-out test split."
    )


def _sanitize_metric_part(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text == "terminal_bench":
        # The converted Seta dataset uses terminal_bench as its reward source.
        # Expose the operational dataset name in metrics for mixed-run debugging.
        text = "seta"
    text = _METRIC_NAME_RE.sub("_", text).strip("._-")
    return text or "unknown"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested_get(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _task_meta_from_sample(sample: Sample) -> dict[str, Any]:
    prompt = _as_dict(getattr(sample, "prompt", None))
    metadata = _as_dict(getattr(sample, "metadata", None))

    if isinstance(_nested_get(metadata, "task_meta"), dict):
        return metadata["task_meta"]
    if isinstance(_nested_get(prompt, "task_meta"), dict):
        return prompt["task_meta"]

    # Dataset records in terminal-rl store task metadata directly under
    # sample.metadata; older/debug samples may store it directly as prompt.
    if any(k in metadata for k in ("data_source", "task_path", "task_name")):
        return metadata
    if any(k in prompt for k in ("data_source", "task_path", "task_name")):
        return prompt
    return {}


def _dataset_name(sample: Sample) -> str:
    metadata = _as_dict(getattr(sample, "metadata", None))
    prompt = _as_dict(getattr(sample, "prompt", None))
    task_meta = _task_meta_from_sample(sample)

    raw = (
        metadata.get("data_source")
        or _nested_get(metadata, "task_meta", "data_source")
        or prompt.get("data_source")
        or _nested_get(prompt, "task_meta", "data_source")
        or task_meta.get("data_source")
    )

    task_path = str(
        metadata.get("task_path")
        or _nested_get(metadata, "task_meta", "task_path")
        or prompt.get("task_path")
        or _nested_get(prompt, "task_meta", "task_path")
        or task_meta.get("task_path")
        or ""
    )
    if not raw:
        if task_path.startswith("agent_safetybench/"):
            raw = "agent_safetybench"
        elif task_path.startswith("seta_env/") or "seta" in task_path:
            raw = "seta"
        elif task_path:
            raw = task_path.split("/", 1)[0]

    return _sanitize_metric_part(raw)


def _reward_details(sample: Sample) -> dict[str, Any]:
    metadata = _as_dict(getattr(sample, "metadata", None))
    details = metadata.get("reward_details")
    return details if isinstance(details, dict) else {}


def _safety_split(sample: Sample) -> str:
    metadata = _as_dict(getattr(sample, "metadata", None))
    raw_split = metadata.get("safety_split")
    if raw_split:
        return _sanitize_metric_part(raw_split)

    task_meta = _task_meta_from_sample(sample)
    data_source = _dataset_name(sample)
    if data_source not in {"agent_safetybench", "agentharm"}:
        return "agentic"

    raw = task_meta.get("fulfillable")
    try:
        fulfillable = int(raw)
    except (TypeError, ValueError):
        task_type = str(task_meta.get("agentharm_task_type") or "").lower()
        fulfillable = 1 if task_type == "benign" else 0
    return "benign_should_comply" if fulfillable == 1 else "harmful_should_refuse"


def _bool_detail(sample: Sample, key: str) -> bool | None:
    details = _reward_details(sample)
    value = details.get(key)
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _reward_value(sample: Sample, key: str = "score") -> float | None:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        return _to_float(reward.get(key))
    if key in ("score", "reward"):
        return _to_float(reward)
    return None


def _env_float(name: str, default: float) -> float:
    value = _to_float(os.getenv(name))
    return default if value is None else value


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _advantage_bonus_enabled() -> bool:
    return _env_enabled(
        "EXPLORE_ADVANTAGE_BONUS_ENABLED",
        os.getenv("EXPLORE_ADVANTAGE_BONUS", "0"),
    )


def _component_value(sample: Sample, key: str) -> float:
    value = _reward_value(sample, key)
    return 0.0 if value is None else float(value)


def _sample_group_key(sample: Sample) -> int:
    try:
        return int(sample.group_index) if getattr(sample, "group_index", None) is not None else -1
    except (TypeError, ValueError):
        return -1


def _sample_traj_key(sample: Sample, sample_idx: int) -> tuple[int, int]:
    try:
        traj_idx = int(sample.index) if getattr(sample, "index", None) is not None else sample_idx
    except (TypeError, ValueError):
        traj_idx = sample_idx
    return _sample_group_key(sample), traj_idx


def _normalize_values(values: list[float], use_std: bool) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    centered = [v - mean for v in values]
    if not use_std:
        return centered
    if len(values) <= 1:
        return [0.0 for _ in values]
    var = sum(v * v for v in centered) / max(1, len(values) - 1)
    std = math.sqrt(max(var, 0.0))
    return [v / (std + 1e-6) for v in centered]


def _group_normalize_values_for_log(
    args: Any,
    samples: list[Sample],
    values: list[float],
) -> list[float]:
    use_std = bool(getattr(args, "grpo_std_normalization", False))
    if getattr(args, "dynamic_history", False):
        value_by_key: dict[tuple[int, int], float] = {}
        group_to_keys: dict[int, list[tuple[int, int]]] = {}
        key_by_sample: list[tuple[int, int]] = []
        for i, sample in enumerate(samples):
            key = _sample_traj_key(sample, i)
            key_by_sample.append(key)
            if key not in value_by_key:
                value_by_key[key] = float(values[i])
                group_to_keys.setdefault(key[0], []).append(key)

        normalized_by_key: dict[tuple[int, int], float] = {}
        for keys in group_to_keys.values():
            vals = _normalize_values([value_by_key[k] for k in keys], use_std)
            for j, key in enumerate(keys):
                normalized_by_key[key] = float(vals[j])
        return [normalized_by_key[key] for key in key_by_sample]

    group_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        group_to_indices[_sample_group_key(sample)].append(i)

    normalized = list(values)
    for idxs in group_to_indices.values():
        vals = _normalize_values([values[i] for i in idxs], use_std)
        for j, sample_idx in enumerate(idxs):
            normalized[sample_idx] = float(vals[j])
    return normalized


def _expected_post_norm_exploration_values(args: Any, samples: list[Sample]) -> list[float]:
    if not samples or not _advantage_bonus_enabled():
        return [0.0 for _ in samples]

    mode = _env_str("EXPLORE_ADVANTAGE_BONUS_MODE", "component").lower()
    if mode in {"dual", "dual_stream", "intrinsic_advantage"}:
        intrinsic_key = _env_str(
            "EXPLORE_ADVANTAGE_INTRINSIC_KEY",
            "explore_agent57_intrinsic_signal",
        )
        intrinsic_values = [_component_value(sample, intrinsic_key) for sample in samples]
        intrinsic_adv = _group_normalize_values_for_log(args, samples, intrinsic_values)
        lambda_coef = _env_float(
            "EXPLORE_ADVANTAGE_LAMBDA",
            _env_float("EXPLORE_ADVANTAGE_BONUS_COEF", 0.1),
        )
        arm_weight_mode = _env_str("EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE", "normalized_beta").lower()
        trust_key = _env_str("EXPLORE_ADVANTAGE_TRUST_KEY", "explore_agent57_trust")
        clip = _env_float("EXPLORE_ADVANTAGE_BONUS_CLIP", 0.0)
        betas = [_component_value(sample, "explore_agent57_beta") for sample in samples]
        max_beta = max([abs(beta) for beta in betas if beta > 0.0] or [1.0])
        bonuses: list[float] = []
        for i, sample in enumerate(samples):
            if arm_weight_mode in {"none", "off", "0"}:
                arm_weight = 1.0
            elif arm_weight_mode in {"raw", "raw_beta"}:
                arm_weight = max(0.0, betas[i])
            else:
                arm_weight = max(0.0, betas[i]) / max(max_beta, 1e-12)
            reward = getattr(sample, "reward", None)
            trust_missing = not isinstance(reward, dict) or trust_key not in reward
            trust = _component_value(sample, trust_key)
            if trust_missing and trust_key == "explore_agent57_trust":
                trust = 1.0
            raw_bonus = float(lambda_coef * arm_weight * trust * intrinsic_adv[i])
            bonuses.append(max(-clip, min(clip, raw_bonus)) if clip > 0 else raw_bonus)
        return bonuses

    component_names = [
        part.strip()
        for part in os.getenv("EXPLORE_ADVANTAGE_BONUS_COMPONENTS", "explore_intrinsic_scaled").split(",")
        if part.strip()
    ]
    coef = _env_float("EXPLORE_ADVANTAGE_BONUS_COEF", 1.0)
    clip = _env_float("EXPLORE_ADVANTAGE_BONUS_CLIP", 0.25)
    bonuses = []
    for sample in samples:
        raw_bonus = sum(_component_value(sample, key) for key in component_names)
        clipped = max(-clip, min(clip, raw_bonus)) if clip > 0 else raw_bonus
        bonuses.append(coef * clipped)
    return bonuses


def _exploration_reward_components(
    args: Any,
    samples: list[Sample],
) -> dict[str, list[float]]:
    score_raw = [_reward_value(sample, "explore_total_bonus") for sample in samples]
    signal_raw = [_reward_value(sample, "explore_all_bonus") for sample in samples]
    explicit_raw = [_reward_value(sample, "exploration_reward") for sample in samples]
    has_score = any(value is not None for value in score_raw)
    has_signal = any(value is not None for value in signal_raw)
    has_explicit = any(value is not None for value in explicit_raw)
    score_values = [float(value or 0.0) for value in score_raw]
    signal_values = [float(value or 0.0) for value in signal_raw] if has_signal else list(score_values)

    post_values = _expected_post_norm_exploration_values(args, samples)
    if has_explicit:
        explicit_values = [float(value or 0.0) for value in explicit_raw]
        explicit_delta = [explicit_values[i] - score_values[i] for i in range(len(samples))]
        if any(abs(value) > 1e-12 for value in explicit_delta):
            post_values = explicit_delta

    has_post = _advantage_bonus_enabled() or any(abs(value) > 1e-12 for value in post_values)
    if not has_score and not has_signal and not has_post and not has_explicit:
        return {"score": [], "signal": [], "post_norm": [], "effective": []}
    effective = [signal_values[i] + post_values[i] for i in range(len(samples))]
    return {
        "score": score_values,
        "signal": signal_values,
        "post_norm": post_values,
        "effective": effective,
    }


def _reward_raw(sample: Sample, key: str) -> Any:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        return reward.get(key)
    return None


def _reward_bool(sample: Sample, key: str) -> bool | None:
    value = _reward_raw(sample, key)
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def _response_length(sample: Sample) -> float | None:
    value = getattr(sample, "effective_response_length", None)
    if value is None:
        value = getattr(sample, "response_length", None)
    return _to_float(value)


def _status_name(sample: Sample) -> str:
    status = getattr(sample, "status", None)
    if isinstance(status, Sample.Status):
        return status.value
    return str(status or "unknown").lower()


def _task_identity(sample: Sample) -> str:
    metadata = _as_dict(getattr(sample, "metadata", None))
    traj = _as_dict(metadata.get("trajectory_save"))
    task_meta = _task_meta_from_sample(sample)
    raw = (
        traj.get("task_id")
        or traj.get("task_name")
        or metadata.get("task_id")
        or metadata.get("task_name")
        or task_meta.get("task_id")
        or task_meta.get("task_name")
        or task_meta.get("task_path")
        or "unknown"
    )
    return str(raw or "unknown")


def _trajectory_identity(sample: Sample) -> str:
    metadata = _as_dict(getattr(sample, "metadata", None))
    traj = _as_dict(metadata.get("trajectory_save"))
    raw = (
        traj.get("rel_path")
        or traj.get("path")
        or traj.get("uid")
        or metadata.get("uid")
        or metadata.get("request_id")
    )
    if raw:
        return str(raw)
    return f"sample:{id(sample)}"


def _task_coverage_summary(samples: List[Sample]) -> dict[str, Any]:
    source = [s for s in samples if not getattr(s, "remove_sample", False)] or samples
    if not source:
        return {}

    counts: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str]] = set()
    for sample in source:
        task = _task_identity(sample)
        key = (task, _trajectory_identity(sample))
        if key in seen:
            continue
        seen.add(key)
        counts[task] += 1
    total = sum(counts.values())
    if total <= 0:
        return {}
    top_task, top_count = max(counts.items(), key=lambda item: item[1])
    return {
        "task/unique_count": len(counts),
        "task/trajectory_count": total,
        "task/top_ratio": top_count / total,
        "task/top_task": _sanitize_metric_part(top_task),
    }


def _trajectory_save_records(samples: List[Sample]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        metadata = _as_dict(getattr(sample, "metadata", None))
        raw = metadata.get("trajectory_save")
        if not isinstance(raw, dict):
            continue
        key = (
            str(raw.get("rel_path") or raw.get("path") or "")
            or "|".join(
                str(raw.get(k) if raw.get(k) is not None else "")
                for k in ("uid", "task_id", "train_step", "group_index", "sample_index")
            )
        )
        if key in seen:
            continue
        seen.add(key)
        records.append(raw)
    return records


def _trajectory_save_summary(samples: List[Sample]) -> dict[str, Any]:
    records = _trajectory_save_records(samples)
    if not records:
        return {}
    considered = len(records)
    saved_records = [r for r in records if bool(r.get("saved"))]
    task_values = {
        str(r.get("task_id") or r.get("task_name") or "unknown")
        for r in records
    }
    saved_task_values = {
        str(r.get("task_id") or r.get("task_name") or "unknown")
        for r in saved_records
    }
    return {
        "trajectory/considered_count": considered,
        "trajectory/saved_count": len(saved_records),
        "trajectory/skipped_count": considered - len(saved_records),
        "trajectory/save_rate": len(saved_records) / considered if considered else 0.0,
        "trajectory/considered_unique_tasks": len(task_values),
        "trajectory/saved_unique_tasks": len(saved_task_values),
    }


def _add_task_and_trajectory_metrics(
    log_dict: Dict[str, Any],
    prefix: str,
    samples: List[Sample],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    task_summary = _task_coverage_summary(samples)
    if task_summary:
        summary.update(task_summary)
        for key in ("task/unique_count", "task/trajectory_count", "task/top_ratio"):
            value = task_summary.get(key)
            if value is not None:
                log_dict[f"{prefix}/{key}"] = value

    traj_summary = _trajectory_save_summary(samples)
    if traj_summary:
        summary.update(traj_summary)
        for key, value in traj_summary.items():
            if value is not None:
                log_dict[f"{prefix}/{key}"] = value

        records = _trajectory_save_records(samples)
        reason_counts: dict[str, int] = defaultdict(int)
        policy_counts: dict[str, int] = defaultdict(int)
        for record in records:
            reason_counts[_sanitize_metric_part(record.get("reason"))] += 1
            policy_counts[_sanitize_metric_part(record.get("policy"))] += 1
        total = len(records)
        for reason, count in sorted(reason_counts.items()):
            log_dict[f"{prefix}/trajectory/reason/{reason}"] = count
            log_dict[f"{prefix}/trajectory/reason_ratio/{reason}"] = (
                count / total if total else 0.0
            )
        for policy, count in sorted(policy_counts.items()):
            log_dict[f"{prefix}/trajectory/policy/{policy}"] = count
            log_dict[f"{prefix}/trajectory/policy_ratio/{policy}"] = (
                count / total if total else 0.0
            )
    return summary


def _mean_token_logprob(sample: Sample) -> float | None:
    values = getattr(sample, "rollout_log_probs", None)
    if not values:
        return None
    nums = [_to_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _trajectory_uncertainty_summaries(samples: List[Sample]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for sample in samples:
        metadata = _as_dict(getattr(sample, "metadata", None))
        summary = _as_dict(metadata.get("trajectory_uncertainty"))
        if not summary:
            continue
        key = (
            summary.get("uid"),
            summary.get("group_index"),
            summary.get("sample_index"),
            summary.get("rollout_id"),
        )
        if not any(v is not None for v in key):
            key = ("sample", id(sample))
        if key in seen:
            continue
        seen.add(key)
        summaries.append(summary)
    return summaries


def _trajectory_uncertainty_mean(samples: List[Sample], key: str) -> float | None:
    values = [
        v for v in (
            _to_float(summary.get(key))
            for summary in _trajectory_uncertainty_summaries(samples)
        )
        if v is not None
    ]
    stats = _stats(values)
    return _stats_mean(stats)


def _stats(values: List[float]) -> dict[str, float] | None:
    nums = []
    for value in values:
        num = _to_float(value)
        if num is not None:
            nums.append(num)
    if not nums:
        return None
    count = len(nums)
    mean = sum(nums) / count
    variance = sum((x - mean) ** 2 for x in nums) / count
    sorted_nums = sorted(nums)

    def percentile(pct: float) -> float:
        if count == 1:
            return sorted_nums[0]
        idx = (count - 1) * pct
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return sorted_nums[lo]
        weight = idx - lo
        return sorted_nums[lo] * (1.0 - weight) + sorted_nums[hi] * weight

    return {
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "min": min(nums),
        "max": max(nums),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
    }


def _add_stats(
    log_dict: Dict[str, Any],
    prefix: str,
    values: List[float],
    *,
    include_percentiles: bool = False,
) -> dict[str, float] | None:
    stats = _stats(values)
    if not stats:
        return None
    keys = (
        ("mean", "std", "min", "max", "p50", "p90")
        if include_percentiles
        else ("mean", "std", "min", "max")
    )
    for key in keys:
        log_dict[f"{prefix}/{key}"] = stats[key]
    return stats


def _add_turn_uncertainty_metrics(
    log_dict: Dict[str, Any],
    prefix: str,
    samples: List[Sample],
) -> dict[str, dict[str, float]]:
    summaries = _trajectory_uncertainty_summaries(samples)
    out: dict[str, dict[str, float]] = {}
    if not summaries:
        return out

    numeric_keys = {
        "mean_turn_level_uncertainty": "mean_neg_logprob",
        "mean_turn_level_score": "mean_score",
        "mean_abs_score_delta": "mean_abs_score_delta",
        "low_progress_fraction": "low_progress_fraction",
        "available_turn_count": "available_turn_count",
        "missing_turn_count": "missing_turn_count",
    }
    for source_key, metric_name in numeric_keys.items():
        values = [
            v for v in (_to_float(summary.get(source_key)) for summary in summaries)
            if v is not None
        ]
        if not values:
            continue
        stats = _add_stats(
            log_dict,
            f"{prefix}/turn_uncertainty/{metric_name}",
            values,
        )
        if stats:
            out[source_key] = stats

    low_counts = [
        _to_float(summary.get("low_progress_turn_count")) for summary in summaries
    ]
    avail_counts = [
        _to_float(summary.get("available_turn_count")) for summary in summaries
    ]
    low_total = sum(v for v in low_counts if v is not None)
    avail_total = sum(v for v in avail_counts if v is not None)
    if avail_total > 0:
        log_dict[f"{prefix}/turn_uncertainty/low_progress_turn_ratio"] = (
            low_total / avail_total
        )

    return out


def _reward_flag(sample: Sample, key: str) -> bool | None:
    value = _reward_raw(sample, key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _agent57_samples(samples: List[Sample]) -> List[Sample]:
    source = [s for s in samples if not getattr(s, "remove_sample", False)] or samples
    out: list[Sample] = []
    for sample in source:
        if any(
            _reward_raw(sample, key) is not None
            for key in (
                "explore_agent57_enabled",
                "explore_agent57_lifelong_enabled",
                "explore_agent57_arm_id",
                "explore_agent57_lifelong_bonus",
            )
        ):
            out.append(sample)
    return out


def _agent57_summary(samples: List[Sample]) -> dict[str, Any]:
    source = [s for s in samples if not getattr(s, "remove_sample", False)] or samples
    agent_samples = _agent57_samples(source)
    if not agent_samples:
        return {}

    arm_counts: dict[int, int] = defaultdict(int)
    suppressed_counts: dict[str, int] = defaultdict(int)
    state_error_count = 0
    active_count = 0
    lifelong_enabled_count = 0
    for sample in agent_samples:
        arm = _reward_value(sample, "explore_agent57_arm_id")
        if arm is not None:
            arm_counts[int(arm)] += 1
        enabled = _reward_flag(sample, "explore_agent57_enabled")
        lifelong_enabled = _reward_flag(sample, "explore_agent57_lifelong_enabled")
        if enabled or lifelong_enabled or arm is not None:
            active_count += 1
        if lifelong_enabled:
            lifelong_enabled_count += 1
        reason_raw = _reward_raw(sample, "explore_agent57_lifelong_suppressed_reason")
        reason = str(reason_raw or "").strip()
        if reason:
            reason_key = _sanitize_metric_part(reason)
            suppressed_counts[reason_key] += 1
            if reason.startswith("state_error:"):
                state_error_count += 1

    def mean(key: str) -> float | None:
        return _stats_mean(
            _stats([v for v in (_reward_value(s, key) for s in agent_samples) if v is not None])
        )

    count = len(agent_samples)
    top_arm: int | None = None
    top_arm_ratio: float | None = None
    if arm_counts:
        top_arm, top_arm_count = max(arm_counts.items(), key=lambda item: item[1])
        top_arm_ratio = top_arm_count / count if count else None

    top_reason: str | None = None
    top_reason_ratio: float | None = None
    if suppressed_counts:
        top_reason, top_reason_count = max(
            suppressed_counts.items(), key=lambda item: item[1]
        )
        top_reason_ratio = top_reason_count / count if count else None

    eligible_values = [
        v for v in (
            _reward_value(s, "explore_agent57_lifelong_eligible")
            for s in agent_samples
        )
        if v is not None
    ]
    eligible_rate = (
        sum(1 for value in eligible_values if value > 0.0) / len(eligible_values)
        if eligible_values
        else None
    )

    return {
        "agent57/active": active_count / count if count else None,
        "agent57/lifelong_enabled": (
            lifelong_enabled_count / count if count else None
        ),
        "agent57/lifelong_bonus": mean("explore_agent57_lifelong_bonus"),
        "agent57/ngu_bonus": mean("explore_agent57_ngu_bonus"),
        "agent57/ngu_episodic": mean("explore_agent57_ngu_episodic"),
        "agent57/ngu_life_mod": mean("explore_agent57_ngu_life_mod"),
        "agent57/intrinsic_signal": mean("explore_agent57_intrinsic_signal"),
        "agent57/episodic_empty_bucket_rate": mean(
            "explore_agent57_episodic_empty_bucket_rate"
        ),
        "agent57/episodic_exact_repeat_count": mean(
            "explore_agent57_episodic_exact_repeat_count"
        ),
        "agent57/bonus_clipped": mean("explore_agent57_bonus_clipped"),
        "agent57/lifelong_raw": mean("explore_agent57_lifelong_raw"),
        "agent57/lifelong_unique_keys": mean(
            "explore_agent57_lifelong_unique_keys"
        ),
        "agent57/lifelong_seen_before": mean(
            "explore_agent57_lifelong_seen_before"
        ),
        "agent57/lifelong_warmup_remaining": mean(
            "explore_agent57_lifelong_warmup_remaining"
        ),
        "agent57/lifelong_eligible_rate": eligible_rate,
        "agent57/lifelong_state_error_rate": (
            state_error_count / count if count else None
        ),
        "agent57/arm_count": len(arm_counts) if arm_counts else None,
        "agent57/top_arm": float(top_arm) if top_arm is not None else None,
        "agent57/top_arm_ratio": top_arm_ratio,
        "agent57/top_suppressed_reason": top_reason,
        "agent57/top_suppressed_ratio": top_reason_ratio,
    }


def _add_agent57_debug_metrics(
    log_dict: Dict[str, Any],
    prefix: str,
    samples: List[Sample],
) -> dict[str, Any]:
    agent_samples = _agent57_samples(samples)
    summary = _agent57_summary(agent_samples)
    if not agent_samples:
        return summary

    total = len(agent_samples)
    log_dict[f"{prefix}/explore/agent57/sample_count"] = total
    for record_key, metric_name in (
        ("agent57/active", "active_rate"),
        ("agent57/lifelong_enabled", "lifelong_enabled_rate"),
        ("agent57/lifelong_eligible_rate", "lifelong_eligible_rate"),
        ("agent57/lifelong_state_error_rate", "lifelong_state_error_rate"),
        ("agent57/arm_count", "arm_count"),
        ("agent57/top_arm", "top_arm"),
        ("agent57/top_arm_ratio", "top_arm_ratio"),
        ("agent57/top_suppressed_ratio", "top_suppressed_ratio"),
        ("agent57/episodic_empty_bucket_rate", "episodic_empty_bucket_rate"),
        ("agent57/episodic_exact_repeat_count", "episodic_exact_repeat_count"),
    ):
        value = summary.get(record_key)
        if value is not None:
            log_dict[f"{prefix}/explore/agent57/{metric_name}"] = value

    for reward_key, metric_name in (
        ("explore_agent57_beta", "beta"),
        ("explore_agent57_max_bonus", "max_bonus"),
        ("explore_agent57_ucb_epsilon", "ucb_epsilon"),
        ("explore_agent57_ucb_min_per_arm", "ucb_min_per_arm"),
        ("explore_agent57_ucb_random_seed", "ucb_random_seed"),
        ("explore_agent57_lifelong_raw", "lifelong_raw"),
        ("explore_agent57_lifelong_bonus", "lifelong_bonus"),
        ("explore_agent57_lifelong_bonus_unclipped", "lifelong_bonus_unclipped"),
        ("explore_agent57_ngu_episodic", "ngu_episodic"),
        ("explore_agent57_ngu_life_mod", "ngu_life_mod"),
        ("explore_agent57_intrinsic_signal", "intrinsic_signal"),
        ("explore_agent57_ngu_bonus", "ngu_bonus"),
        ("explore_agent57_ngu_bonus_unclipped", "ngu_bonus_unclipped"),
        ("explore_agent57_bonus_clipped", "bonus_clipped"),
        ("explore_agent57_lifelong_unique_keys", "lifelong_unique_keys"),
        ("explore_agent57_lifelong_seen_before", "lifelong_seen_before"),
        ("explore_agent57_lifelong_warmup_remaining", "lifelong_warmup_remaining"),
        ("explore_agent57_episodic_action_count", "episodic_action_count"),
        ("explore_agent57_episodic_empty_bucket_count", "episodic_empty_bucket_count"),
        ("explore_agent57_episodic_empty_bucket_rate", "episodic_empty_bucket_rate"),
        ("explore_agent57_episodic_exact_repeat_count", "episodic_exact_repeat_count"),
        ("explore_agent57_episodic_candidate_count_mean", "episodic_candidate_count_mean"),
        ("explore_agent57_episodic_probe_count_mean", "episodic_probe_count_mean"),
    ):
        values = [
            v for v in (_reward_value(s, reward_key) for s in agent_samples)
            if v is not None
        ]
        if values:
            _add_stats(
                log_dict,
                f"{prefix}/explore/agent57/{metric_name}",
                values,
                include_percentiles=metric_name in {
                    "intrinsic_signal",
                    "lifelong_raw",
                    "lifelong_bonus",
                    "ngu_episodic",
                    "ngu_bonus",
                    "bonus_clipped",
                },
            )

    suppressed_counts: dict[str, int] = defaultdict(int)
    by_arm: dict[int, list[Sample]] = defaultdict(list)
    for sample in agent_samples:
        reason = str(
            _reward_raw(sample, "explore_agent57_lifelong_suppressed_reason") or ""
        ).strip()
        if reason:
            suppressed_counts[_sanitize_metric_part(reason)] += 1
        arm = _reward_value(sample, "explore_agent57_arm_id")
        if arm is not None:
            by_arm[int(arm)].append(sample)

    for reason, count in sorted(suppressed_counts.items()):
        log_dict[f"{prefix}/explore/agent57/suppressed/{reason}"] = count
        log_dict[f"{prefix}/explore/agent57/suppressed_ratio/{reason}"] = (
            count / total if total else 0.0
        )

    for arm_id, arm_samples in sorted(by_arm.items()):
        arm_prefix = f"{prefix}/explore/agent57/arm_{arm_id}"
        arm_count = len(arm_samples)
        log_dict[f"{arm_prefix}/sample_count"] = arm_count
        log_dict[f"{arm_prefix}/sample_ratio"] = arm_count / total if total else 0.0
        for reward_key, metric_name in (
            ("score", "reward"),
            ("explore_base_score_before_bonus", "base_reward"),
            ("explore_agent57_beta", "beta"),
            ("explore_agent57_lifelong_raw", "lifelong_raw"),
            ("explore_agent57_lifelong_bonus", "lifelong_bonus"),
            ("explore_agent57_ngu_bonus", "ngu_bonus"),
            ("explore_agent57_ngu_life_mod", "ngu_life_mod"),
            ("explore_agent57_intrinsic_signal", "intrinsic_signal"),
            ("explore_agent57_bonus_clipped", "bonus_clipped"),
            ("explore_agent57_lifelong_eligible", "lifelong_eligible"),
        ):
            values = [
                v for v in (_reward_value(s, reward_key) for s in arm_samples)
                if v is not None
            ]
            if values:
                _add_stats(log_dict, f"{arm_prefix}/{metric_name}", values)

    return summary


def _add_exploration_debug_metrics(
    log_dict: Dict[str, Any],
    prefix: str,
    samples: List[Sample],
) -> dict[str, Any]:
    """Add structured exploration/exploitation health metrics.

    The "mood" fields are intentionally coarse: they make live logs scannable
    during mixed training without replacing the lower-level numeric components.
    """
    source = [s for s in samples if not getattr(s, "remove_sample", False)] or samples
    summary: dict[str, Any] = {}
    if not source:
        return summary

    numeric_keys = (
        "explore_intrinsic",
        "explore_intrinsic_scaled",
        "explore_total_bonus",
        "explore_base_score_before_bonus",
        "explore_bonus_to_base_abs_ratio",
        "explore_curiosity_pressure",
        "explore_tool_intrinsic_pressure",
        "explore_intrinsic_in_total",
        "explore_safety_pressure",
        "explore_action_count",
        "explore_tool_call_count",
        "explore_danger_command_count",
        "explore_parse_error_count",
        "explore_cde_actor_log_ppl",
        "explore_lprnd_raw",
        "explore_agent57_beta",
        "explore_agent57_lifelong_raw",
        "explore_agent57_lifelong_bonus",
        "explore_agent57_ngu_bonus",
        "explore_agent57_ngu_episodic",
        "explore_agent57_ngu_life_mod",
        "explore_agent57_intrinsic_signal",
        "explore_agent57_bonus_clipped",
        "explore_agent57_lifelong_unique_keys",
        "explore_agent57_lifelong_seen_before",
        "explore_agent57_lifelong_warmup_remaining",
        "explore_agent57_lifelong_eligible",
        "explore_agent57_episodic_action_count",
        "explore_agent57_episodic_empty_bucket_rate",
        "explore_agent57_episodic_exact_repeat_count",
        "explore_agent57_episodic_candidate_count_mean",
        "explore_agent57_episodic_probe_count_mean",
    )
    percentile_keys = {
        "explore_intrinsic",
        "explore_intrinsic_scaled",
        "explore_intrinsic_in_total",
        "explore_total_bonus",
        "explore_agent57_intrinsic_signal",
        "explore_agent57_lifelong_raw",
        "explore_agent57_lifelong_bonus",
        "explore_agent57_ngu_bonus",
        "explore_agent57_ngu_episodic",
    }
    for key in numeric_keys:
        values = [v for v in (_reward_value(s, key) for s in source) if v is not None]
        if values:
            stats = _add_stats(
                log_dict,
                f"{prefix}/explore/{key}",
                values,
                include_percentiles=key in percentile_keys,
            )
            if stats and key in {
                "explore_total_bonus",
                "explore_intrinsic",
                "explore_intrinsic_scaled",
                "explore_bonus_to_base_abs_ratio",
                "explore_curiosity_pressure",
                "explore_safety_pressure",
            }:
                summary[f"{key}_mean"] = stats["mean"]

    for key in (
        "explore_reward_hacking_risk",
        "explore_over_exploration_risk",
        "explore_safety_tension",
    ):
        values = [v for v in (_reward_bool(s, key) for s in source) if v is not None]
        if values:
            rate = sum(1 for v in values if v) / len(values)
            log_dict[f"{prefix}/explore/{key}_rate"] = rate
            summary[f"{key}_rate"] = rate

    mood_counts: dict[str, int] = defaultdict(int)
    for sample in source:
        mood = _reward_raw(sample, "explore_mood")
        if mood:
            mood_counts[_sanitize_metric_part(mood)] += 1
    if mood_counts:
        total = sum(mood_counts.values())
        top_mood, top_count = max(mood_counts.items(), key=lambda item: item[1])
        summary["top_mood"] = top_mood
        summary["top_mood_ratio"] = top_count / total if total else 0.0
        for mood, count in sorted(mood_counts.items()):
            log_dict[f"{prefix}/explore/mood/{mood}"] = count
            log_dict[f"{prefix}/explore/mood_ratio/{mood}"] = count / total if total else 0.0

    summary.update(_add_agent57_debug_metrics(log_dict, prefix, source))
    return summary


def _stats_mean(stats: dict[str, float] | None) -> float | None:
    return stats["mean"] if stats else None


def _env_enabled(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _structured_metrics_path() -> Path | None:
    configured = os.getenv("TERMINAL_METRICS_JSONL")
    if configured:
        return Path(configured)
    run_dir = os.getenv("RUN_DIR")
    if run_dir:
        return Path(run_dir) / "logs" / "metrics.jsonl"
    return None


def _write_structured_metrics(records: list[dict[str, Any]]) -> None:
    if not _env_enabled("TERMINAL_STRUCTURED_METRICS", "1"):
        return
    if not records:
        return

    lines = [
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for record in records
    ]
    for text in lines:
        logger.info("%s %s", _STRUCTURED_LOG_PREFIX, text)

    path = _structured_metrics_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for text in lines:
                f.write(text)
                f.write("\n")
    except Exception as e:
        logger.warning("Failed to write structured rollout metrics to %s: %s", path, e)


def _run_name() -> str | None:
    if os.getenv("RUN_ID"):
        return os.getenv("RUN_ID")
    if os.getenv("RUN_NAME"):
        return os.getenv("RUN_NAME")
    if os.getenv("RUN_DIR"):
        return Path(os.getenv("RUN_DIR", "")).name
    return None


def _epoch(args: Any) -> int | float | None:
    for key in ("epoch", "data_epoch", "train_epoch"):
        value = getattr(args, key, None)
        num = _to_float(value)
        if num is not None:
            return int(num) if float(num).is_integer() else num
    return None


def _analysis_dataset_name_from_raw(raw_name: Any) -> str:
    name = _sanitize_metric_part(raw_name)
    if name in {"terminal_bench", "seta", "seta_env"}:
        return "seta"
    if name.startswith("seta_"):
        return "seta"
    if name in {"agent_safetybench", "asb"} or name.startswith("agent_safetybench"):
        return "agent_safetybench"
    if name in {"agentharm", "ah"} or name.startswith("agentharm"):
        return "agentharm"
    if name in {"safety", "security", "mcpsafety", "harmbench"} or name.startswith("safety"):
        return "security"
    return name


def _analysis_dataset_name(sample: Sample) -> str:
    return _analysis_dataset_name_from_raw(_dataset_name(sample))


def _mean_from_samples(samples: list[Sample], key: str) -> float | None:
    values = [v for v in (_reward_value(s, key) for s in samples) if v is not None]
    stats = _stats(values)
    return _stats_mean(stats)


def _metric_record_from_samples(
    *,
    args: Any,
    phase: str,
    dataset_name: str,
    source_datasets: list[str],
    rollout_id: int,
    step: int,
    samples: list[Sample],
    rollout_time: float | None = None,
    kl: float | None = None,
    entropy: float | None = None,
) -> dict[str, Any]:
    trainable = [s for s in samples if not getattr(s, "remove_sample", False)]
    reward_source = trainable or samples
    total_values = [v for v in (_reward_value(s, "score") for s in reward_source) if v is not None]
    total_stats = _stats(total_values)
    raw_score = _mean_from_samples(reward_source, "raw_score")
    task_reward = _mean_from_samples(reward_source, "base_score")
    if task_reward is None:
        task_reward = raw_score
    if task_reward is None:
        task_reward = _stats_mean(total_stats)

    exploration_components = _exploration_reward_components(args, reward_source)
    exploration_score_stats = _stats(exploration_components["score"])
    exploration_signal_stats = _stats(exploration_components["signal"])
    exploration_post_norm_stats = _stats(exploration_components["post_norm"])
    exploration_stats = _stats(exploration_components["effective"])
    exploration_abs_stats = _stats([abs(v) for v in exploration_components["effective"]])
    exploration_post_norm_abs_stats = _stats([abs(v) for v in exploration_components["post_norm"]])
    exploration_reward_score = _stats_mean(exploration_score_stats)
    exploration_reward_signal = _stats_mean(exploration_signal_stats)
    exploration_reward_post_norm = _stats_mean(exploration_post_norm_stats)
    exploration_reward_abs = _stats_mean(exploration_abs_stats)
    exploration_reward_post_norm_abs = _stats_mean(exploration_post_norm_abs_stats)
    exploration_reward = _stats_mean(exploration_stats)
    denom = None
    if task_reward is not None and exploration_reward is not None:
        denom = task_reward + exploration_reward
    exploration_abs_to_task_ratio = (
        exploration_reward_abs / max(abs(task_reward), 1e-12)
        if task_reward is not None and exploration_reward_abs is not None
        else None
    )

    scale_sources = {_analysis_dataset_name_from_raw(src) for src in source_datasets}
    if dataset_name == "seta" or scale_sources == {"seta"}:
        raw_reward_scale = "pass_rate_0_1"
        raw_reward_semantics = "terminal task test pass rate; 1.0 means all trainable samples passed"
        raw_reward_min = 0.0
        raw_reward_max = 1.0
    elif dataset_name in {"agent_safetybench", "agentharm", "security"} or scale_sources.intersection(
        {"agent_safetybench", "agentharm", "security", "safety"}
    ):
        raw_reward_scale = "direct_safety_score"
        raw_reward_semantics = "dataset reward-model score, not a 0/1 pass rate"
        raw_reward_min = None
        raw_reward_max = None
    else:
        raw_reward_scale = "unknown"
        raw_reward_semantics = None
        raw_reward_min = None
        raw_reward_max = None

    pass_rate = _mean_from_samples(reward_source, "accuracy")
    if dataset_name == "seta" or scale_sources == {"seta"}:
        pass_rate_semantics = (
            "legacy test_acc alias for current rollout task unit-test pass rate; "
            "train phase is not the SETA held-out test split"
        )
    elif phase == "eval":
        pass_rate_semantics = (
            "legacy test_acc alias for eval sample reward/accuracy; actual split "
            "depends on eval_prompt_data"
        )
    else:
        pass_rate_semantics = "legacy test_acc alias for current rollout sample accuracy"

    status_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        status_counts[_status_name(sample)] += 1
    sample_count = len(samples)
    truncated_count = int(status_counts.get(Sample.Status.TRUNCATED.value, 0))
    turn_uncertainty_mean = _trajectory_uncertainty_mean(
        samples, "mean_turn_level_uncertainty"
    )
    turn_score_mean = _trajectory_uncertainty_mean(samples, "mean_turn_level_score")
    turn_score_delta_mean = _trajectory_uncertainty_mean(
        samples, "mean_abs_score_delta"
    )
    low_progress_fraction = _trajectory_uncertainty_mean(
        samples, "low_progress_fraction"
    )
    agent57_fields = _agent57_summary(reward_source)
    task_fields = _task_coverage_summary(samples)
    trajectory_fields = _trajectory_save_summary(samples)

    # `reward/exploration` is the effective exploration signal for logging:
    # raw score-space bonus when present, otherwise the aggregate exploration
    # signal plus the expected post-normalization advantage bonus. The score-only
    # and post-norm parts are logged separately for diagnosis.
    return {
        "schema": _STRUCTURED_SCHEMA,
        "schema_version": _STRUCTURED_SCHEMA_VERSION,
        "run": _run_name(),
        "phase": phase,
        "dataset": dataset_name,
        "source_datasets": sorted(set(source_datasets)),
        "global_step": int(step),
        "epoch": _epoch(args),
        "rollout_id": int(rollout_id),
        "sample_count": sample_count,
        "trainable_count": len(trainable),
        "completed": int(status_counts.get(Sample.Status.COMPLETED.value, 0)),
        "truncated": truncated_count,
        "truncated_fraction": (
            truncated_count / sample_count if sample_count > 0 else None
        ),
        "failed": int(status_counts.get(Sample.Status.FAILED.value, 0)),
        "aborted": int(status_counts.get(Sample.Status.ABORTED.value, 0)),
        "reward/total": _stats_mean(total_stats),
        "reward/task": task_reward,
        "reward/raw": raw_score,
        "reward/exploration": exploration_reward,
        "reward/exploration_abs": exploration_reward_abs,
        "reward/exploration_score": exploration_reward_score,
        "reward/exploration_signal": exploration_reward_signal,
        "reward/exploration_post_norm": exploration_reward_post_norm,
        "reward/exploration_post_norm_abs": exploration_reward_post_norm_abs,
        "reward/exploration_ratio": (
            exploration_reward / denom if denom and abs(denom) > 1e-12 else None
        ),
        "reward/exploration_abs_to_task_ratio": exploration_abs_to_task_ratio,
        "total_reward": _stats_mean(total_stats),
        "task_reward": task_reward,
        "raw_reward": raw_score,
        "exploration_reward": exploration_reward if exploration_reward is not None else 0.0,
        "exploration_reward_abs": exploration_reward_abs,
        "exploration_reward_score": exploration_reward_score,
        "exploration_reward_signal": exploration_reward_signal,
        "exploration_reward_post_norm": exploration_reward_post_norm,
        "exploration_reward_post_norm_abs": exploration_reward_post_norm_abs,
        "raw_reward_scale": raw_reward_scale,
        "raw_reward_semantics": raw_reward_semantics,
        "raw_reward_min": raw_reward_min,
        "raw_reward_max": raw_reward_max,
        "pass_rate": pass_rate,
        "unit_test_pass_rate": pass_rate if dataset_name == "seta" else None,
        "test_acc": pass_rate,
        "test_acc_semantics": pass_rate_semantics,
        "test_acc_is_heldout_test_split": False,
        "reward_std": total_stats["std"] if total_stats else None,
        "response_length": _stats_mean(
            _stats([v for v in (_response_length(s) for s in samples) if v is not None])
        ),
        "kl": kl,
        "entropy": entropy,
        "turn_uncertainty/mean_neg_logprob": turn_uncertainty_mean,
        "turn_uncertainty/mean_score": turn_score_mean,
        "turn_uncertainty/mean_abs_score_delta": turn_score_delta_mean,
        "turn_uncertainty/low_progress_fraction": low_progress_fraction,
        "rollout_time_sec": rollout_time,
        **agent57_fields,
        **task_fields,
        **trajectory_fields,
    }


def _metric_record_from_rewards(
    *,
    args: Any,
    phase: str,
    dataset_name: str,
    rollout_id: int,
    step: int,
    rewards: list[Any],
    kl: float | None = None,
    entropy: float | None = None,
) -> dict[str, Any]:
    reward_values = [_to_float(v) for v in rewards]
    reward_values = [v for v in reward_values if v is not None]
    stats = _stats(reward_values)
    return {
        "schema": _STRUCTURED_SCHEMA,
        "schema_version": _STRUCTURED_SCHEMA_VERSION,
        "run": _run_name(),
        "phase": phase,
        "dataset": dataset_name,
        "source_datasets": [dataset_name],
        "global_step": int(step),
        "epoch": _epoch(args),
        "rollout_id": int(rollout_id),
        "sample_count": len(rewards),
        "trainable_count": len(rewards),
        "completed": None,
        "truncated": None,
        "truncated_fraction": None,
        "failed": None,
        "aborted": None,
        "reward/total": _stats_mean(stats),
        "reward/task": _stats_mean(stats),
        "reward/raw": _stats_mean(stats),
        "reward/exploration": None,
        "reward/exploration_abs": None,
        "reward/exploration_score": None,
        "reward/exploration_signal": None,
        "reward/exploration_post_norm": None,
        "reward/exploration_post_norm_abs": None,
        "reward/exploration_ratio": None,
        "reward/exploration_abs_to_task_ratio": None,
        "total_reward": _stats_mean(stats),
        "task_reward": _stats_mean(stats),
        "raw_reward": _stats_mean(stats),
        "exploration_reward": 0.0,
        "exploration_reward_abs": None,
        "exploration_reward_score": None,
        "exploration_reward_signal": None,
        "exploration_reward_post_norm": None,
        "exploration_reward_post_norm_abs": None,
        "raw_reward_scale": "eval_reward_values",
        "raw_reward_semantics": "aggregate eval reward values without sample-level reward components",
        "raw_reward_min": None,
        "raw_reward_max": None,
        "pass_rate": _stats_mean(stats),
        "unit_test_pass_rate": _stats_mean(stats) if dataset_name == "seta" else None,
        "test_acc": _stats_mean(stats),
        "test_acc_semantics": (
            "legacy test_acc alias for eval reward mean; actual split depends on eval_prompt_data"
        ),
        "test_acc_is_heldout_test_split": False,
        "reward_std": stats["std"] if stats else None,
        "response_length": None,
        "kl": kl,
        "entropy": entropy,
        "rollout_time_sec": None,
    }


def _aggregate_metric_records(
    *,
    args: Any,
    phase: str,
    rollout_id: int,
    step: int,
    samples: list[Sample],
    rollout_time: float | None = None,
    kl: float | None = None,
    entropy: float | None = None,
) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[Sample]] = defaultdict(list)
    source_by_dataset: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        raw_source = _dataset_name(sample)
        dataset_name = _analysis_dataset_name_from_raw(raw_source)
        by_dataset[dataset_name].append(sample)
        source_by_dataset[dataset_name].add(raw_source)

    records = [
        _metric_record_from_samples(
            args=args,
            phase=phase,
            dataset_name=dataset_name,
            source_datasets=sorted(source_by_dataset[dataset_name]),
            rollout_id=rollout_id,
            step=step,
            samples=by_dataset[dataset_name],
            rollout_time=rollout_time,
            kl=kl,
            entropy=entropy,
        )
        for dataset_name in sorted(by_dataset)
    ]

    if len(records) > 1:
        records.append(
            _metric_record_from_samples(
                args=args,
                phase=phase,
                dataset_name="mixed-all",
                source_datasets=sorted({_dataset_name(s) for s in samples}),
                rollout_id=rollout_id,
                step=step,
                samples=samples,
                rollout_time=rollout_time,
                kl=kl,
                entropy=entropy,
            )
        )
    return records


def _add_per_dataset_log_dict(log_dict: Dict[str, Any], records: list[dict[str, Any]]) -> None:
    for record in records:
        dataset = _sanitize_metric_part(record["dataset"])
        for key in (
            "reward/total",
            "reward/task",
            "reward/raw",
            "reward/exploration",
            "reward/exploration_abs",
            "reward/exploration_score",
            "reward/exploration_signal",
            "reward/exploration_post_norm",
            "reward/exploration_post_norm_abs",
            "reward/exploration_ratio",
            "reward/exploration_abs_to_task_ratio",
            "total_reward",
            "task_reward",
            "raw_reward",
            "exploration_reward",
            "exploration_reward_abs",
            "exploration_reward_score",
            "exploration_reward_signal",
            "exploration_reward_post_norm",
            "exploration_reward_post_norm_abs",
            "truncated_fraction",
            "pass_rate",
            "unit_test_pass_rate",
            "test_acc",
            "reward_std",
            "response_length",
            "kl",
            "entropy",
            "turn_uncertainty/mean_neg_logprob",
            "turn_uncertainty/mean_score",
            "turn_uncertainty/mean_abs_score_delta",
            "turn_uncertainty/low_progress_fraction",
            "agent57/active",
            "agent57/lifelong_enabled",
            "agent57/lifelong_bonus",
            "agent57/lifelong_raw",
            "agent57/lifelong_unique_keys",
            "agent57/lifelong_seen_before",
            "agent57/lifelong_warmup_remaining",
            "agent57/lifelong_eligible_rate",
            "agent57/lifelong_state_error_rate",
            "agent57/intrinsic_signal",
            "agent57/episodic_empty_bucket_rate",
            "agent57/episodic_exact_repeat_count",
            "agent57/arm_count",
            "agent57/top_arm",
            "agent57/top_arm_ratio",
            "agent57/top_suppressed_ratio",
            "task/unique_count",
            "task/trajectory_count",
            "task/top_ratio",
            "trajectory/considered_count",
            "trajectory/saved_count",
            "trajectory/skipped_count",
            "trajectory/save_rate",
            "trajectory/considered_unique_tasks",
            "trajectory/saved_unique_tasks",
        ):
            value = record.get(key)
            if value is not None:
                log_dict[f"per_dataset/{dataset}/{key}"] = value
        log_dict[f"per_dataset/{dataset}/sample_count"] = record.get("sample_count", 0)
        log_dict[f"per_dataset/{dataset}/trainable_count"] = record.get("trainable_count", 0)


def _format_per_dataset_table(
    records: list[dict[str, Any]],
    *,
    phase: str,
    step: int,
) -> str:
    if not records:
        return ""
    run = _run_name() or "-"
    epoch = records[0].get("epoch")
    title = f"========== step {step} | epoch {epoch if epoch is not None else '-'} | phase: {phase} | run: {run} =========="
    header = (
        "dataset          | reward/total | reward/task | reward/explore | a57_life | a57_elg | pass_rate | "
        "resp_len | kl     | entropy"
    )
    sep = "-----------------+--------------+-------------+----------------+----------+---------+----------+----------+--------+--------"
    body = []
    for record in records:
        body.append(
            f"{str(record['dataset'])[:16]:16} | "
            f"{_format_float(record.get('reward/total'), width=12)} | "
            f"{_format_float(record.get('reward/task'), width=11)} | "
            f"{_format_float(record.get('reward/exploration'), width=14)} | "
            f"{_format_float(record.get('agent57/lifelong_bonus'), width=8)} | "
            f"{_format_float(record.get('agent57/lifelong_eligible_rate'), width=7)} | "
            f"{_format_float(record.get('pass_rate'), width=9)} | "
            f"{_format_float(record.get('response_length'), width=8)} | "
            f"{_format_float(record.get('kl'), width=6)} | "
            f"{_format_float(record.get('entropy'), width=6)}"
        )
    return "\n".join([title, header, sep, *body, "=" * len(title)])


def _format_eval_diag(records: list[dict[str, Any]]) -> str:
    eval_records = [r for r in records if r.get("dataset") != "mixed-all"]
    if not eval_records:
        return ""

    lines = ["[DIAG] per-dataset eval reward movement"]
    task_by_dataset: dict[str, float] = {}
    explore_ratios: list[float] = []
    conclusions: list[str] = []

    for record in eval_records:
        dataset = str(record["dataset"])
        task = _to_float(record.get("reward/task"))
        explore = _to_float(record.get("reward/exploration"))
        ratio = _to_float(record.get("reward/exploration_ratio"))
        if task is not None:
            task_by_dataset[dataset] = task
        if ratio is not None:
            explore_ratios.append(ratio)

        prev = _LAST_EVAL_BY_DATASET.get(dataset)
        delta = None
        if prev is not None and task is not None:
            prev_task = _to_float(prev.get("reward/task"))
            if prev_task is not None:
                delta = task - prev_task

        lines.append(
            f"[DIAG] dataset={dataset} delta_reward/task={_format_float(delta)} "
            f"explore_ratio={_format_float(ratio)}"
        )
        if delta is not None and abs(delta) <= 0.01 and ratio is not None and ratio > 0.2:
            conclusions.append(f"{dataset}: 疑似只在探索没在学习")
        baseline_delta = _recent_eval_task_delta_from_jsonl(
            os.getenv("TERMINAL_BASELINE_METRICS_JSONL"),
            dataset,
        )
        if baseline_delta is not None:
            lines.append(
                f"[DIAG] dataset={dataset} baseline_delta_reward/task="
                f"{_format_float(baseline_delta)}"
            )
            if baseline_delta > 0.01 and (delta is None or delta <= 0.01):
                conclusions.append(f"{dataset}: baseline 涨而 exploration 不涨, 疑似探索奖励干扰主任务")

    if "seta" in task_by_dataset and "security" in task_by_dataset:
        gap = task_by_dataset["seta"] - task_by_dataset["security"]
        lines.append(f"[DIAG] task_reward_gap_seta_minus_security={gap:.4f}")
    else:
        lines.append("[DIAG] task_reward_gap_seta_minus_security=NA")

    if not conclusions:
        if explore_ratios and max(explore_ratios) > 0.5:
            conclusions.append("探索奖励占比偏高,需结合下一次 eval 的 task reward 变化判断")
        else:
            conclusions.append("未触发探索压过任务学习的启发式告警")
    lines.append("[DIAG] conclusion=" + "；".join(conclusions))

    for record in eval_records:
        _LAST_EVAL_BY_DATASET[str(record["dataset"])] = dict(record)
    return "\n".join(lines)


def _recent_eval_task_delta_from_jsonl(path_raw: str | None, dataset: str) -> float | None:
    if not path_raw:
        return None
    path = Path(path_raw)
    if not path.exists():
        return None

    recent: list[float] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    record.get("schema") != _STRUCTURED_SCHEMA
                    or record.get("phase") != "eval"
                    or record.get("dataset") != dataset
                ):
                    continue
                task = _to_float(record.get("reward/task"))
                if task is None:
                    continue
                recent.append(task)
                if len(recent) > 2:
                    recent = recent[-2:]
    except Exception as e:
        logger.warning("Failed to read baseline metrics from %s: %s", path, e)
        return None

    if len(recent) < 2:
        return None
    return recent[-1] - recent[-2]


def _format_float(value: Any, width: int = 8) -> str:
    num = _to_float(value)
    if num is None:
        return " " * (width - 1) + "-"
    if 0.0 < abs(num) < 1e-3:
        return f"{num:{width}.2e}"
    return f"{num:{width}.3f}"


def _dataset_metrics(
    args: Any,
    samples: List[Sample],
) -> tuple[Dict[str, Any], List[dict[str, Any]], List[dict[str, Any]]]:
    log_dict: Dict[str, Any] = {}
    rows: List[dict[str, Any]] = []
    split_rows: List[dict[str, Any]] = []
    total = len(samples)
    by_dataset: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_dataset[_dataset_name(sample)].append(sample)

    for dataset_name in sorted(by_dataset):
        dataset_samples = by_dataset[dataset_name]
        trainable = [s for s in dataset_samples if not getattr(s, "remove_sample", False)]
        prefix = f"terminal/dataset/{dataset_name}"

        count = len(dataset_samples)
        trainable_count = len(trainable)
        ratio = count / total if total else 0.0
        log_dict[f"{prefix}/sample_count"] = count
        log_dict[f"{prefix}/sample_ratio"] = ratio
        log_dict[f"{prefix}/trainable_count"] = trainable_count
        log_dict[f"{prefix}/trainable_ratio"] = trainable_count / count if count else 0.0

        status_counts = {status.value: 0 for status in Sample.Status}
        status_counts["unknown"] = 0
        for sample in dataset_samples:
            status = _status_name(sample)
            status_counts[status] = status_counts.get(status, 0) + 1
        for status, status_count in sorted(status_counts.items()):
            if status_count:
                log_dict[f"{prefix}/status/{status}"] = status_count
                log_dict[f"{prefix}/status_ratio/{status}"] = status_count / count if count else 0.0

        reward_stats = _add_stats(
            log_dict,
            f"{prefix}/reward",
            [v for v in (_reward_value(s, "score") for s in trainable) if v is not None],
        )

        acc_stats = _add_stats(
            log_dict,
            f"{prefix}/acc",
            [v for v in (_reward_value(s, "accuracy") for s in trainable) if v is not None],
        )

        response_stats = _add_stats(
            log_dict,
            f"{prefix}/response_length",
            [v for v in (_response_length(s) for s in dataset_samples) if v is not None],
            include_percentiles=True,
        )

        component_stats: dict[str, dict[str, float]] = {}
        for reward_key in _REWARD_COMPONENT_KEYS:
            values = [v for v in (_reward_value(s, reward_key) for s in trainable) if v is not None]
            if values:
                stats = _add_stats(log_dict, f"{prefix}/reward_component/{reward_key}", values)
                if stats:
                    component_stats[reward_key] = stats
        exploration_components = _exploration_reward_components(args, trainable)
        exploration_effective_stats = _stats(exploration_components["effective"])
        exploration_score_stats = _stats(exploration_components["score"])
        exploration_signal_stats = _stats(exploration_components["signal"])
        exploration_post_norm_stats = _stats(exploration_components["post_norm"])
        exploration_post_norm_abs_stats = _stats([abs(v) for v in exploration_components["post_norm"]])

        explore_summary = _add_exploration_debug_metrics(log_dict, prefix, dataset_samples)
        _add_turn_uncertainty_metrics(log_dict, prefix, dataset_samples)

        safety_values = [v for v in (_reward_value(s, "safety_score") for s in trainable) if v is not None]
        if safety_values:
            log_dict[f"{prefix}/safety_negative_ratio"] = sum(1 for x in safety_values if x < 0) / len(
                safety_values
            )
            safety_coef = next(
                (coef for coef in (_reward_value(s, "safety_coef") for s in trainable) if coef is not None),
                None,
            )
            if safety_coef is not None:
                log_dict[f"{prefix}/safety_coef"] = safety_coef

        mean_logprobs = [v for v in (_mean_token_logprob(s) for s in dataset_samples) if v is not None]
        if mean_logprobs:
            _add_stats(log_dict, f"{prefix}/rollout_logprob", mean_logprobs)
            _add_stats(log_dict, f"{prefix}/rollout_neg_logprob", [-x for x in mean_logprobs])

        n_cs_calls = 0
        n_cs_errors = 0
        for sample in dataset_samples:
            safety_meta = _as_dict(_as_dict(getattr(sample, "metadata", None)).get("safety"))
            n_cs_calls += int(safety_meta.get("n_calls", 0) or 0)
            n_cs_errors += int(safety_meta.get("n_errors", 0) or 0)
        if n_cs_calls > 0:
            log_dict[f"{prefix}/clawsentry_calls_total"] = n_cs_calls
            log_dict[f"{prefix}/clawsentry_errors_total"] = n_cs_errors
            log_dict[f"{prefix}/clawsentry_error_rate"] = n_cs_errors / n_cs_calls

        reason_counts: dict[str, int] = defaultdict(int)
        for sample in trainable:
            reason = _reward_details(sample).get("reason")
            if reason:
                reason_counts[_sanitize_metric_part(reason)] += 1
        for reason, reason_count in sorted(reason_counts.items()):
            log_dict[f"{prefix}/reward_reason/{reason}"] = reason_count
            log_dict[f"{prefix}/reward_reason_ratio/{reason}"] = (
                reason_count / trainable_count if trainable_count else 0.0
            )

        by_split: dict[str, list[Sample]] = defaultdict(list)
        for sample in dataset_samples:
            by_split[_safety_split(sample)].append(sample)
        for split_name in sorted(by_split):
            split_samples = by_split[split_name]
            split_trainable = [s for s in split_samples if not getattr(s, "remove_sample", False)]
            split_prefix = f"{prefix}/split/{split_name}"
            split_count = len(split_samples)
            split_trainable_count = len(split_trainable)
            log_dict[f"{split_prefix}/sample_count"] = split_count
            log_dict[f"{split_prefix}/sample_ratio"] = split_count / total if total else 0.0
            log_dict[f"{split_prefix}/dataset_ratio"] = split_count / count if count else 0.0
            log_dict[f"{split_prefix}/trainable_count"] = split_trainable_count

            split_reward_stats = _add_stats(
                log_dict,
                f"{split_prefix}/reward",
                [v for v in (_reward_value(s, "score") for s in split_trainable) if v is not None],
            )
            split_acc_stats = _add_stats(
                log_dict,
                f"{split_prefix}/acc",
                [v for v in (_reward_value(s, "accuracy") for s in split_trainable) if v is not None],
            )
            split_component_stats: dict[str, dict[str, float]] = {}
            for reward_key in (
                "raw_score",
                "base_score",
                "safety_score",
                "explore_total_bonus",
                "explore_intrinsic_scaled",
                "explore_agent57_lifelong_bonus",
                "explore_safety_penalty",
            ):
                values = [
                    v for v in (_reward_value(s, reward_key) for s in split_trainable)
                    if v is not None
                ]
                if values:
                    stats = _add_stats(log_dict, f"{split_prefix}/reward_component/{reward_key}", values)
                    if stats:
                        split_component_stats[reward_key] = stats
            split_exploration_components = _exploration_reward_components(args, split_trainable)
            split_exploration_effective_stats = _stats(split_exploration_components["effective"])
            split_exploration_score_stats = _stats(split_exploration_components["score"])
            split_exploration_signal_stats = _stats(split_exploration_components["signal"])
            split_exploration_post_norm_stats = _stats(split_exploration_components["post_norm"])
            split_exploration_post_norm_abs_stats = _stats(
                [abs(v) for v in split_exploration_components["post_norm"]]
            )
            _add_stats(
                log_dict,
                f"{split_prefix}/response_length",
                [v for v in (_response_length(s) for s in split_samples) if v is not None],
                include_percentiles=True,
            )

            split_reason_counts: dict[str, int] = defaultdict(int)
            for sample in split_trainable:
                reason = _reward_details(sample).get("reason")
                if reason:
                    split_reason_counts[_sanitize_metric_part(reason)] += 1
            top_reason = "-"
            top_reason_count = 0
            if split_reason_counts:
                top_reason, top_reason_count = max(
                    split_reason_counts.items(), key=lambda item: item[1]
                )
            for reason, reason_count in sorted(split_reason_counts.items()):
                log_dict[f"{split_prefix}/reward_reason/{reason}"] = reason_count
                log_dict[f"{split_prefix}/reward_reason_ratio/{reason}"] = (
                    reason_count / split_trainable_count if split_trainable_count else 0.0
                )

            def bool_rate(detail_key: str) -> float | None:
                vals = [
                    value
                    for value in (_bool_detail(s, detail_key) for s in split_trainable)
                    if value is not None
                ]
                if not vals:
                    return None
                rate = sum(1 for value in vals if value) / len(vals)
                log_dict[f"{split_prefix}/{detail_key}_rate"] = rate
                return rate

            verbal_refused_rate = bool_rate("verbal_refused")
            attempted_tool_rate = bool_rate("attempted_tool_use")
            nonempty_rate = bool_rate("nonempty_response")
            bool_rate("truncation_cap_applied")
            bool_rate("safe_truncated_refusal")
            if nonempty_rate is not None:
                log_dict[f"{split_prefix}/empty_response_rate"] = 1.0 - nonempty_rate

            for detail_key in _REWARD_DETAIL_NUMERIC_KEYS:
                values = [
                    v
                    for v in (
                        _to_float(_reward_details(s).get(detail_key))
                        for s in split_trainable
                    )
                    if v is not None
                ]
                if values:
                    _add_stats(log_dict, f"{split_prefix}/detail/{detail_key}", values)

            split_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "count": split_count,
                    "ratio": split_count / total if total else 0.0,
                    "trainable": split_trainable_count,
                    "reward_mean": split_reward_stats["mean"] if split_reward_stats else None,
                    "test_acc_mean": split_acc_stats["mean"] if split_acc_stats else None,
                    "acc_mean": split_acc_stats["mean"] if split_acc_stats else None,
                    "pass_rate_mean": split_acc_stats["mean"] if split_acc_stats else None,
                    "raw_score_mean": _stats_mean(split_component_stats.get("raw_score")),
                    "base_reward_mean": _stats_mean(split_component_stats.get("base_score")),
                    "safety_score_mean": _stats_mean(split_component_stats.get("safety_score")),
                    "exploration_reward_mean": _stats_mean(split_exploration_effective_stats),
                    "exploration_reward_score_mean": _stats_mean(split_exploration_score_stats),
                    "exploration_reward_signal_mean": _stats_mean(split_exploration_signal_stats),
                    "exploration_reward_post_norm_mean": _stats_mean(split_exploration_post_norm_stats),
                    "exploration_reward_post_norm_abs_mean": _stats_mean(split_exploration_post_norm_abs_stats),
                    "explore_intrinsic_scaled_mean": _stats_mean(
                        split_component_stats.get("explore_intrinsic_scaled")
                    ),
                    "explore_agent57_lifelong_bonus_mean": _stats_mean(
                        split_component_stats.get("explore_agent57_lifelong_bonus")
                    ),
                    "explore_safety_penalty_mean": _stats_mean(
                        split_component_stats.get("explore_safety_penalty")
                    ),
                    "verbal_refused_rate": verbal_refused_rate,
                    "attempted_tool_rate": attempted_tool_rate,
                    "empty_response_rate": (1.0 - nonempty_rate) if nonempty_rate is not None else None,
                    "top_reason": top_reason,
                    "top_reason_ratio": (
                        top_reason_count / split_trainable_count
                        if split_trainable_count
                        else None
                    ),
                }
            )

        rows.append(
            {
                "dataset": dataset_name,
                "count": count,
                "ratio": ratio,
                "trainable": trainable_count,
                "reward_mean": _stats_mean(reward_stats),
                "reward_std": reward_stats["std"] if reward_stats else None,
                "test_acc_mean": _stats_mean(acc_stats),
                "acc_mean": _stats_mean(acc_stats),
                "pass_rate_mean": _stats_mean(acc_stats),
                "raw_score_mean": _stats_mean(component_stats.get("raw_score")),
                "base_reward_mean": _stats_mean(component_stats.get("base_score")),
                "safety_score_mean": _stats_mean(component_stats.get("safety_score")),
                "exploration_reward_mean": _stats_mean(exploration_effective_stats),
                "exploration_reward_score_mean": _stats_mean(exploration_score_stats),
                "exploration_reward_signal_mean": _stats_mean(exploration_signal_stats),
                "exploration_reward_post_norm_mean": _stats_mean(exploration_post_norm_stats),
                "exploration_reward_post_norm_abs_mean": _stats_mean(exploration_post_norm_abs_stats),
                "explore_intrinsic_scaled_mean": _stats_mean(
                    component_stats.get("explore_intrinsic_scaled")
                ),
                "explore_safety_penalty_mean": _stats_mean(
                    component_stats.get("explore_safety_penalty")
                ),
                "explore_lprnd_mean": _stats_mean(component_stats.get("explore_lprnd")),
                "explore_agent57_lifelong_bonus_mean": _stats_mean(
                    component_stats.get("explore_agent57_lifelong_bonus")
                ),
                "explore_agent57_lifelong_raw_mean": _stats_mean(
                    component_stats.get("explore_agent57_lifelong_raw")
                ),
                "agent57_lifelong_eligible_rate": explore_summary.get(
                    "agent57/lifelong_eligible_rate"
                ),
                "agent57_lifelong_state_error_rate": explore_summary.get(
                    "agent57/lifelong_state_error_rate"
                ),
                "agent57_arm_count": explore_summary.get("agent57/arm_count"),
                "agent57_top_arm": explore_summary.get("agent57/top_arm"),
                "agent57_top_arm_ratio": explore_summary.get("agent57/top_arm_ratio"),
                "agent57_top_suppressed_reason": explore_summary.get(
                    "agent57/top_suppressed_reason"
                ),
                "agent57_top_suppressed_ratio": explore_summary.get(
                    "agent57/top_suppressed_ratio"
                ),
                "explore_cde_actor_bonus_mean": _stats_mean(
                    component_stats.get("explore_cde_actor_bonus")
                ),
                "response_mean": response_stats["mean"] if response_stats else None,
                "completed": status_counts.get(Sample.Status.COMPLETED.value, 0),
                "truncated": status_counts.get(Sample.Status.TRUNCATED.value, 0),
                "failed": status_counts.get(Sample.Status.FAILED.value, 0),
                "aborted": status_counts.get(Sample.Status.ABORTED.value, 0),
                "explore_mood": explore_summary.get("top_mood"),
                "explore_pressure": explore_summary.get("explore_bonus_to_base_abs_ratio_mean"),
                "reward_hack_risk": explore_summary.get("explore_reward_hacking_risk_rate"),
            }
        )

    return log_dict, rows, split_rows


def _format_dataset_table(rows: List[dict[str, Any]]) -> str:
    if not rows:
        return ""
    header = (
        "dataset                 n  ratio train  rew_mean  rew_std     pass resp_len  "
        "comp trunc fail abort mood              xpress rhack"
    )
    line = "-" * len(header)
    body = []
    for row in rows:
        body.append(
            f"{str(row['dataset'])[:22]:22} "
            f"{int(row['count']):4d} "
            f"{row['ratio']:6.2%} "
            f"{int(row['trainable']):5d} "
            f"{_format_float(row['reward_mean'])} "
            f"{_format_float(row['reward_std'])} "
            f"{_format_float(row['acc_mean'])} "
            f"{_format_float(row['response_mean'])} "
            f"{int(row['completed']):5d} "
            f"{int(row['truncated']):5d} "
            f"{int(row['failed']):4d} "
            f"{int(row['aborted']):5d} "
            f"{str(row.get('explore_mood') or '-')[:16]:16} "
            f"{_format_float(row.get('explore_pressure'), width=6)} "
            f"{_format_float(row.get('reward_hack_risk'), width=5)}"
        )
    return "\n".join([header, line, *body])


def _format_reward_breakdown_table(rows: List[dict[str, Any]]) -> str:
    if not rows:
        return ""
    header = (
        "dataset                 n train final_reward pass_rate raw_score base_reward "
        "safety explore_reward postabs intrinsic safepen   a57   cde lprnd"
    )
    line = "-" * len(header)
    body = []
    for row in rows:
        body.append(
            f"{str(row['dataset'])[:22]:22} "
            f"{int(row['count']):4d} "
            f"{int(row['trainable']):5d} "
            f"{_format_float(row.get('reward_mean'), width=12)} "
            f"{_format_float(row.get('pass_rate_mean'), width=9)} "
            f"{_format_float(row.get('raw_score_mean'), width=9)} "
            f"{_format_float(row.get('base_reward_mean'), width=11)} "
            f"{_format_float(row.get('safety_score_mean'), width=6)} "
            f"{_format_float(row.get('exploration_reward_mean'), width=14)} "
            f"{_format_float(row.get('exploration_reward_post_norm_abs_mean'), width=7)} "
            f"{_format_float(row.get('explore_intrinsic_scaled_mean'), width=9)} "
            f"{_format_float(row.get('explore_safety_penalty_mean'), width=7)} "
            f"{_format_float(row.get('explore_agent57_lifelong_bonus_mean'), width=5)} "
            f"{_format_float(row.get('explore_cde_actor_bonus_mean'), width=5)} "
            f"{_format_float(row.get('explore_lprnd_mean'), width=5)}"
        )
    return "\n".join([header, line, *body])


def _format_agent57_table(rows: List[dict[str, Any]]) -> str:
    agent_rows = [
        row for row in rows
        if row.get("explore_agent57_lifelong_bonus_mean") is not None
        or row.get("agent57_arm_count") is not None
    ]
    if not agent_rows:
        return ""
    header = (
        "dataset                 n arms top_arm a57_raw a57_bonus eligible stateerr suppressed"
    )
    line = "-" * len(header)
    body = []
    for row in agent_rows:
        suppressed = str(row.get("agent57_top_suppressed_reason") or "-")
        suppressed_ratio = _to_float(row.get("agent57_top_suppressed_ratio"))
        if suppressed != "-" and suppressed_ratio is not None:
            suppressed = f"{suppressed[:18]}:{suppressed_ratio:.0%}"
        top_arm = row.get("agent57_top_arm")
        top_arm_ratio = _to_float(row.get("agent57_top_arm_ratio"))
        top_arm_text = "-"
        if top_arm is not None:
            try:
                top_arm_text = str(int(float(top_arm)))
            except (TypeError, ValueError):
                top_arm_text = str(top_arm)
            if top_arm_ratio is not None:
                top_arm_text = f"{top_arm_text}:{top_arm_ratio:.0%}"
        arm_count = row.get("agent57_arm_count")
        try:
            arm_count_text = str(int(float(arm_count))) if arm_count is not None else "-"
        except (TypeError, ValueError):
            arm_count_text = str(arm_count)
        body.append(
            f"{str(row['dataset'])[:22]:22} "
            f"{int(row['count']):4d} "
            f"{arm_count_text[:4]:4} "
            f"{top_arm_text[:7]:7} "
            f"{_format_float(row.get('explore_agent57_lifelong_raw_mean'), width=7)} "
            f"{_format_float(row.get('explore_agent57_lifelong_bonus_mean'), width=9)} "
            f"{_format_float(row.get('agent57_lifelong_eligible_rate'), width=8)} "
            f"{_format_float(row.get('agent57_lifelong_state_error_rate'), width=8)} "
            f"{suppressed}"
        )
    return "\n".join([header, line, *body])


def _format_split_table(rows: List[dict[str, Any]]) -> str:
    if not rows:
        return ""
    header = (
        "dataset                 split                    n  ratio train  rew_mean     pass "
        "refuse   tools   empty top_reason"
    )
    line = "-" * len(header)
    body = []
    for row in rows:
        top_reason = str(row.get("top_reason") or "-")
        top_ratio = _to_float(row.get("top_reason_ratio"))
        if top_ratio is not None and top_reason != "-":
            top_reason = f"{top_reason[:24]}:{top_ratio:.0%}"
        body.append(
            f"{str(row['dataset'])[:22]:22} "
            f"{str(row['split'])[:24]:24} "
            f"{int(row['count']):4d} "
            f"{row['ratio']:6.2%} "
            f"{int(row['trainable']):5d} "
            f"{_format_float(row['reward_mean'])} "
            f"{_format_float(row['acc_mean'])} "
            f"{_format_float(row['verbal_refused_rate'])} "
            f"{_format_float(row['attempted_tool_rate'])} "
            f"{_format_float(row['empty_response_rate'])} "
            f"{top_reason}"
        )
    return "\n".join([header, line, *body])


def rollout_log(rollout_id, args, samples, rollout_extra_metrics, rollout_time):

    trainable = [s for s in samples if not getattr(s, "remove_sample", False)]
    non_trainable = [s for s in samples if getattr(s, "remove_sample", False)]

    log_dict: Dict[str, Any] = {}

    total = len(samples)
    n_failed = sum(1 for s in samples if s.status == Sample.Status.FAILED)
    n_aborted = sum(1 for s in samples if s.status == Sample.Status.ABORTED)
    n_truncated = sum(1 for s in samples if s.status == Sample.Status.TRUNCATED)
    n_completed = sum(1 for s in samples if s.status == Sample.Status.COMPLETED)

    log_dict["terminal/total_samples"] = total
    log_dict["terminal/completed"] = n_completed
    log_dict["terminal/truncated"] = n_truncated
    log_dict["terminal/failed"] = n_failed
    log_dict["terminal/aborted"] = n_aborted
    log_dict["terminal/failed_ratio"] = n_failed / total if total else 0.0
    log_dict["terminal/non_trainable_ratio"] = (
        len(non_trainable) / total if total else 0.0
    )

    if trainable:
        trainable_rewards = [
            v for v in (_reward_value(s, "score") for s in trainable) if v is not None
        ]
        log_dict["terminal/reward_mean"] = sum(trainable_rewards) / len(
            trainable_rewards
        ) if trainable_rewards else 0.0
        if trainable_rewards:
            reward_stats = _stats(trainable_rewards)
            log_dict["terminal/reward_std"] = reward_stats["std"]
            log_dict["terminal/reward_min"] = reward_stats["min"]
            log_dict["terminal/reward_max"] = reward_stats["max"]

        trainable_accs = []
        for s in trainable:
            if isinstance(s.reward, dict) and "accuracy" in s.reward:
                trainable_accs.append(float(s.reward["accuracy"]))
        if trainable_accs:
            pass_rate = sum(trainable_accs) / len(trainable_accs)
            log_dict["terminal/accuracy"] = pass_rate
            log_dict["terminal/pass_rate"] = pass_rate
            log_dict["terminal/train_batch_pass_rate"] = pass_rate

        trainable_prm = []
        for s in trainable:
            if isinstance(s.reward, dict) and "prm_turn_score" in s.reward:
                trainable_prm.append(float(s.reward["prm_turn_score"]))
        if trainable_prm:
            log_dict["terminal/prm_turn_score"] = sum(trainable_prm) / len(
                trainable_prm
            )

        trainable_safety = []
        trainable_safety_coef = None
        for s in trainable:
            if isinstance(s.reward, dict) and "safety_score" in s.reward:
                trainable_safety.append(float(s.reward["safety_score"]))
                if trainable_safety_coef is None:
                    trainable_safety_coef = float(s.reward.get("safety_coef", 0.0))
        if trainable_safety:
            n = len(trainable_safety)
            log_dict["terminal/safety_score_mean"] = sum(trainable_safety) / n
            log_dict["terminal/safety_score_min"] = min(trainable_safety)
            log_dict["terminal/safety_score_max"] = max(trainable_safety)
            log_dict["terminal/safety_negative_ratio"] = (
                sum(1 for x in trainable_safety if x < 0) / n
            )
            if trainable_safety_coef is not None:
                log_dict["terminal/safety_coef"] = trainable_safety_coef

    dataset_log_dict, dataset_rows, split_rows = _dataset_metrics(args, samples)
    log_dict.update(dataset_log_dict)
    _add_task_and_trajectory_metrics(log_dict, "terminal", samples)
    _add_exploration_debug_metrics(log_dict, "terminal", samples)
    _add_turn_uncertainty_metrics(log_dict, "terminal", samples)

    n_cs_calls = 0
    n_cs_errors = 0
    for s in samples:
        safety_meta = (s.metadata or {}).get("safety") if s.metadata else None
        if isinstance(safety_meta, dict):
            n_cs_calls += int(safety_meta.get("n_calls", 0) or 0)
            n_cs_errors += int(safety_meta.get("n_errors", 0) or 0)
    if n_cs_calls > 0:
        log_dict["terminal/clawsentry_calls_total"] = n_cs_calls
        log_dict["terminal/clawsentry_errors_total"] = n_cs_errors
        log_dict["terminal/clawsentry_error_rate"] = n_cs_errors / n_cs_calls

    log_dict["terminal/rollout_time"] = rollout_time

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    _log_metric_semantics_once()
    metric_records = _aggregate_metric_records(
        args=args,
        phase="train",
        rollout_id=rollout_id,
        step=step,
        samples=samples,
        rollout_time=rollout_time,
    )
    _add_per_dataset_log_dict(log_dict, metric_records)
    per_dataset_table = _format_per_dataset_table(metric_records, phase="train", step=step)
    if per_dataset_table:
        logger.info("per-dataset metrics\n%s", per_dataset_table)
    table = _format_dataset_table(dataset_rows)
    if table:
        logger.info("dataset metrics rollout=%s step=%s\n%s", rollout_id, step, table)
    reward_table = _format_reward_breakdown_table(dataset_rows)
    if reward_table:
        logger.info(
            "dataset reward breakdown rollout=%s step=%s\n%s",
            rollout_id,
            step,
            reward_table,
        )
    agent57_table = _format_agent57_table(dataset_rows)
    if agent57_table:
        logger.info("agent57 diagnostics rollout=%s step=%s\n%s", rollout_id, step, agent57_table)
    split_table = _format_split_table(split_rows)
    if split_table:
        logger.info("dataset split metrics rollout=%s step=%s\n%s", rollout_id, step, split_table)
    _write_structured_metrics(metric_records)
    _ensure_terminal_step_metric(args)
    logging_utils.log(args, _filter_wandb_metrics(log_dict), step_key="rollout/step")

    return False


def eval_rollout_log(rollout_id, args, data, extra_metrics=None):
    """Per-dataset eval logging hook.

    The hook writes the same JSONL schema as training rollouts. It returns
    False so slime's default eval logger still emits its legacy metrics.
    """
    step = compute_rollout_step(args, rollout_id)
    records: list[dict[str, Any]] = []
    all_samples: list[Sample] = []
    all_rewards: list[Any] = []

    for raw_name, info in sorted((data or {}).items()):
        dataset_name = _analysis_dataset_name_from_raw(raw_name)
        samples = info.get("samples") if isinstance(info, dict) else None
        if samples:
            all_samples.extend(samples)
            records.append(
                _metric_record_from_samples(
                    args=args,
                    phase="eval",
                    dataset_name=dataset_name,
                    source_datasets=[str(raw_name)],
                    rollout_id=rollout_id,
                    step=step,
                    samples=samples,
                )
            )
            continue

        rewards = info.get("rewards", []) if isinstance(info, dict) else []
        all_rewards.extend(rewards)
        records.append(
            _metric_record_from_rewards(
                args=args,
                phase="eval",
                dataset_name=dataset_name,
                rollout_id=rollout_id,
                step=step,
                rewards=rewards,
            )
        )

    if len(records) > 1:
        if all_samples:
            records.append(
                _metric_record_from_samples(
                    args=args,
                    phase="eval",
                    dataset_name="mixed-all",
                    source_datasets=sorted({str(name) for name in (data or {}).keys()}),
                    rollout_id=rollout_id,
                    step=step,
                    samples=all_samples,
                )
            )
        else:
            records.append(
                _metric_record_from_rewards(
                    args=args,
                    phase="eval",
                    dataset_name="mixed-all",
                    rollout_id=rollout_id,
                    step=step,
                    rewards=all_rewards,
                )
            )

    log_dict: Dict[str, Any] = dict(extra_metrics or {})
    log_dict["eval/step"] = step
    # `per_dataset/*` is defined on rollout/step so train/eval points share the
    # same x-axis when inspecting dataset-specific curves in wandb.
    log_dict["rollout/step"] = step
    _add_per_dataset_log_dict(log_dict, records)

    table = _format_per_dataset_table(records, phase="eval", step=step)
    if table:
        logger.info("per-dataset eval metrics\n%s", table)
    diag = _format_eval_diag(records)
    if diag:
        logger.info("%s", diag)

    _write_structured_metrics(records)
    _ensure_terminal_step_metric(args)
    logging_utils.log(args, _filter_wandb_metrics(log_dict), step_key="eval/step")
    return False
