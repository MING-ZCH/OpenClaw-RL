from types import SimpleNamespace

from slime.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std
from slime.utils.types import Sample


def _sample(score):
    return Sample(reward={"score": score})


def test_check_reward_nonzero_std_accepts_nested_terminal_rl_turns():
    args = SimpleNamespace(reward_key="score")
    samples = [
        [_sample(0.0), _sample(0.0)],
        [_sample(0.0), _sample(1.0)],
        [_sample(0.0), _sample(0.0)],
        [_sample(0.0), _sample(1.0)],
    ]

    output = check_reward_nonzero_std(args, samples)

    assert output.keep is True
    assert output.reason is None


def test_check_reward_nonzero_std_drops_zero_std_nested_group():
    args = SimpleNamespace(reward_key="score")
    samples = [
        [_sample(0.0), _sample(0.0)],
        [_sample(0.0), _sample(0.0)],
    ]

    output = check_reward_nonzero_std(args, samples)

    assert output.keep is False
    assert output.reason == "zero_std_0.0"


def test_check_reward_nonzero_std_drops_empty_nested_group():
    args = SimpleNamespace(reward_key="score")

    output = check_reward_nonzero_std(args, [[], []])

    assert output.keep is False
    assert output.reason == "empty_reward_group"
