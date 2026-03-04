"""Q-baseline advantage: discounted returns minus a running baseline."""

from __future__ import annotations

import torch

from src.advantages.base import Advantage
from src.baselines.base import Baseline
from src.utils.returns import compute_returns


class QBaselineAdvantage(Advantage):
    """
    Advantage = G_t - b, where b is maintained by an internal Baseline instance.

    The baseline is updated once per batch (in `update`) before `compute`
    is called per episode, so all episodes in a batch see the same b.

    Parameters
    ----------
    baseline : any Baseline implementation (e.g. ExponentialBaseline)
    """

    def __init__(self, baseline: Baseline) -> None:
        self._baseline = baseline

    def update(self, batch_returns: list[torch.Tensor]) -> None:
        """Update the baseline with the batch mean return."""
        self._baseline.update(batch_returns)

    def compute(
        self,
        rewards: list[float],
        states: torch.Tensor,
        gamma: float,
        device: torch.device,
    ) -> torch.Tensor:
        returns = compute_returns(rewards, gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        b = self._baseline.get()
        return returns_t - b