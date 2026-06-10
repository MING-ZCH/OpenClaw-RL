import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std"]


def _representative_sample(sample_or_turns) -> Sample | None:
    """Return one reward-bearing Sample per original completion.

    terminal-rl multi-turn generation returns a list of per-turn training
    samples for each completion. Dynamic sampling should compare completions
    inside the prompt group, not treat each per-turn sample as a separate
    completion or crash on nested lists.
    """
    if isinstance(sample_or_turns, Sample):
        return sample_or_turns
    if isinstance(sample_or_turns, list):
        for item in reversed(sample_or_turns):
            sample = _representative_sample(item)
            if sample is not None:
                return sample
    return None


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    representatives = [_representative_sample(sample) for sample in samples]
    representatives = [sample for sample in representatives if sample is not None]
    if not representatives:
        return DynamicFilterOutput(keep=False, reason="empty_reward_group")

    rewards = [sample.get_reward_value(args) for sample in representatives]
    keep = bool(torch.tensor(rewards, dtype=torch.float).std(unbiased=False) > 0.0)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
