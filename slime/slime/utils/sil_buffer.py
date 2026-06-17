"""
SPEAR Self-Imitation Learning (SIL) buffer.

Stores high-score trajectories for replay training. Only admits trajectories
whose reward >= score_threshold, optionally restricted to those with positive
advantage (enable_trajectory_posadv).

Advantage re-estimation modes at sample time:
    weight_decay = -1.0  (default)
        A_new = R - baseline  (baseline = caller-supplied p50 of current batch)
    weight_decay in [0, 1]
        A_new = (weight_decay ** age_in_steps) * A_stored

Ref: "Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive
     Exploration" (Qin et al., 2026)
     https://github.com/TencentYoutuResearch/SPEAR
"""

import random
from collections import deque
from typing import Any, Dict, List, Optional

__all__ = ["SILBuffer"]


class SILBuffer:
    """Fixed-capacity FIFO buffer for SPEAR self-imitation replay.

    Each stored entry is a dict with at minimum the following keys:
        tokens          list[int]              full sequence (prompt+response)
        response_length int                    number of response tokens
        loss_mask       list[float/int]        per-response-token loss mask
        reward          float                  scalar reward
        advantage       float                  advantage at collection time
        rollout_log_probs  Optional[tensor]    behavior policy log probs
        step_collected  int                    global step at admission
    """

    def __init__(
        self,
        buffer_size: int = 2048,
        score_threshold: float = 1.0,
        posadv_only: bool = False,
        weight_decay: float = -1.0,
    ) -> None:
        if not (weight_decay == -1.0 or 0.0 <= weight_decay <= 1.0):
            raise ValueError(
                f"weight_decay must be -1.0 (p50 recompute) or in [0,1] (decay), got {weight_decay}"
            )
        self.buffer_size = buffer_size
        self.score_threshold = score_threshold
        self.posadv_only = posadv_only
        self.weight_decay = weight_decay
        self._buf: deque = deque(maxlen=buffer_size)
        self.total_admitted: int = 0
        self.total_rejected: int = 0

    def push(self, entries: List[Dict[str, Any]], current_step: int) -> None:
        """Attempt to admit trajectory dicts into the buffer."""
        for entry in entries:
            reward = float(entry.get("reward", 0.0))
            advantage = float(entry.get("advantage", reward))
            if reward < self.score_threshold:
                self.total_rejected += 1
                continue
            if self.posadv_only and advantage <= 0.0:
                self.total_rejected += 1
                continue
            record = dict(entry)
            record["step_collected"] = current_step
            record["advantage"] = advantage
            self._buf.append(record)
            self.total_admitted += 1

    def sample(
        self,
        n: int,
        current_step: int,
        baseline_reward: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Sample up to n entries with re-estimated advantages."""
        if len(self._buf) == 0 or n <= 0:
            return []
        raw = random.sample(list(self._buf), min(n, len(self._buf)))
        result = []
        for entry in raw:
            e = dict(entry)
            if self.weight_decay == -1.0:
                if baseline_reward is not None:
                    e["advantage"] = e["reward"] - baseline_reward
            else:
                age = max(int(current_step) - int(e.get("step_collected", 0)), 0)
                e["advantage"] = (self.weight_decay ** age) * e["advantage"]
            result.append(e)
        return result

    def __len__(self) -> int:
        return len(self._buf)

    def stats(self) -> Dict[str, float]:
        total = self.total_admitted + self.total_rejected
        return {
            "sil_buffer_size": float(len(self._buf)),
            "sil_buffer_capacity": float(self.buffer_size),
            "sil_total_admitted": float(self.total_admitted),
            "sil_total_rejected": float(self.total_rejected),
            "sil_admit_rate": float(self.total_admitted) / max(total, 1),
        }
