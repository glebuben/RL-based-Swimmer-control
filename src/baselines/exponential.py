from __future__ import annotations
from src.baselines.base import Baseline


class ExponentialBaseline(Baseline):
    """
    Exponential moving average of batch mean returns.

    alpha close to 1 = fast adaptation, alpha close to 0 = slow/stable.
    Initialised from the first batch to avoid cold-start bias from zero.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value: float | None = None

    def update(self, batch_mean_return: float) -> None:
        if self.value is None:
            self.value = batch_mean_return
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * batch_mean_return

    def get(self) -> float:
        return self.value if self.value is not None else 0.0