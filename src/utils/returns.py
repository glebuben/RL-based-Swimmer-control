"""Utilities for computing discounted returns from reward sequences."""

from __future__ import annotations

import torch
import numpy as np


def compute_returns(rewards: list[float], gamma: float) -> list[float]:
    """
    Compute discounted returns G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …

    Parameters
    ----------
    rewards : per-step rewards for a single episode.
    gamma   : discount factor in [0, 1].

    Returns
    -------
    List of the same length as `rewards`, where element t is G_t.
    """
    returns: list[float] = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def normalise_returns_batch(
    batch_returns: list[torch.Tensor],
) -> list[torch.Tensor]:
    """
    Standardise a batch of per-episode return tensors to zero mean, unit std.

    All tensors are concatenated to compute global statistics, then each
    tensor is normalised independently using those statistics.  This keeps
    relative differences between episodes meaningful while preventing
    exploding / vanishing gradients from raw return magnitudes.

    Parameters
    ----------
    batch_returns : list of 1-D float tensors, one per episode.

    Returns
    -------
    List of normalised tensors with the same shapes as the input.
    """
    if not batch_returns:
        return batch_returns
    all_G  = torch.cat(batch_returns)
    G_mean = all_G.mean()
    G_std  = all_G.std().clamp(min=1e-8)
    return [(G - G_mean) / G_std for G in batch_returns]