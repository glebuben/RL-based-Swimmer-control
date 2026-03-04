from __future__ import annotations
from src.baselines.base import Baseline
import torch

class ExponentialBaseline(Baseline):
    """
    Exponential moving average of batch mean returns.

    alpha close to 1 = fast adaptation, alpha close to 0 = slow/stable.
    Initialised from the first batch to avoid cold-start bias from zero.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value: float | None = None

    def update(self, batch_returns: list[torch.Tensor]) -> None:
        
         # stack → shape (batch_size, T)
        stacked = torch.stack(batch_returns)

        # mean over batch → shape (T,)
        batch_mean = stacked.mean(dim=0)

        if self.value is None:
            self.value = batch_mean
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * batch_mean

    def get(self) -> float:
        if self.value is None:
            return 0.0
        return self.value