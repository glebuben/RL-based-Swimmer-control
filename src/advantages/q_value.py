"""Q-value advantage: raw discounted returns with no adjustment."""

from __future__ import annotations

import torch

from src.advantages.base import Advantage
from src.utils.returns import compute_returns


class QValueAdvantage(Advantage):
    """
    Advantage = discounted return G_t.
    Stateless — no baseline or value function.
    """

    def compute(
        self,
        rewards: list[float],
        states: torch.Tensor,
        gamma: float,
        device: torch.device,
    ) -> torch.Tensor:
        returns = compute_returns(rewards, gamma)
        return torch.tensor(returns, dtype=torch.float32, device=device)