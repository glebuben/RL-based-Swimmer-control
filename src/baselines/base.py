from __future__ import annotations
from abc import ABC, abstractmethod


class Baseline(ABC):

    @abstractmethod
    def update(self, batch_mean_return: float) -> None:
        """Update the baseline with the latest batch mean return."""

    @abstractmethod
    def get(self) -> float:
        """Return the current baseline value."""