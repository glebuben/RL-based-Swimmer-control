"""Abstract base class for advantage estimators."""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch

from src.utils.returns import compute_returns

class Advantage(ABC):

    @abstractmethod
    def compute(
        self,
        rewards: list[float],
        states: torch.Tensor,
        gamma: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute advantage estimates for a single episode.

        Parameters
        ----------
        rewards : per-step rewards for the episode
        states  : (T, state_dim) tensor of visited states
        gamma   : discount factor
        device  : torch device

        Returns
        -------
        (T,) tensor of advantage estimates
        """

    def update(self, episode_returns: list[float]) -> None:
        """
        Update any internal state after a batch of episodes.
        Stateless advantages can leave this as a no-op.
        """

    def compute_batch(
        self,
        all_rewards: list[list[float]],
        all_states: torch.Tensor,
        gamma: float,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[float]]:
        """
        Compute advantages for a full batch of parallel episodes.

        Parameters
        ----------
        all_rewards : list[n_envs] of per-step reward lists
        all_states  : (n_envs, T, state_dim) tensor
        gamma       : discount factor
        device      : torch device

        Returns
        -------
        batch_advantages : list[n_envs] of (T,) advantage tensors
        episode_returns  : list[n_envs] of scalar undiscounted returns
        """
        batch_returns   = [torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32, device=device) for rewards in all_rewards]
        episode_returns = [sum(rewards) for rewards in all_rewards]


        batch_advantages = [
            self.compute(rewards, all_states[i], gamma, device)
            for i, rewards in enumerate(all_rewards)
        ]

        # Update internal state (e.g. baseline) using this batch before computing
        self.update(batch_returns)

        return batch_advantages, episode_returns