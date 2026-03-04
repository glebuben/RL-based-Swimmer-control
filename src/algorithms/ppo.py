"""
REINFORCE for Swimmer-v5 using AsyncVectorEnv for parallel episode collection.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch import optim
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from src.nn.nn_policy import ContinuousPolicy
from src.utils.returns import normalise_returns_batch
from src.utils.metrics import MetricsManager
from src.advantages.base import Advantage
from src.advantages import make_advantage


def _collect_batch(
    envs: AsyncVectorEnv,
    target_policy: ContinuousPolicy,
    old_policy: ContinuousPolicy,
    max_steps: int,
    device: torch.device,
) -> tuple[list[list[torch.Tensor]], list[list[float]], torch.Tensor, list[float]]:
    """
    Run all parallel envs for one episode each and collect experience.

    Swimmer only ever truncates (at max_steps), never terminates early,
    so x_position is simply read from infos at the start and end.

    Returns
    -------
    all_lh_ratios : list of n_env lists of max_steps scalar tensors
    all_rewards   : list of n_env lists of max_steps floats
    all_states: tensor of shape (n_envs, max_steps, state_dim) containing states visited in the batch
    x_distances   : x displacement per env over the episode
    """
    n_envs = envs.num_envs

    obs, infos = envs.reset()
    x_starts = np.array(infos["x_position"])

    all_lh_ratios = [[] for _ in range(n_envs)]
    all_rewards = [[] for _ in range(n_envs)]
    all_states = []

    for _ in range(max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        all_states.append(obs_t)
        action = old_policy.sample_actions(obs_t)
        target_policy_lh = target_policy.prob(obs_t, action)
        old_policy_lh = old_policy.prob(obs_t, action)
        lh_ratio = target_policy_lh / (old_policy_lh + 1e-8)

        obs, rewards, terminated, truncated, infos = envs.step(
            action.detach().cpu().numpy()
        )

        for i in range(n_envs):
            all_lh_ratios[i].append(lh_ratio[i].squeeze())
            all_rewards[i].append(float(rewards[i]))

        if (terminated | truncated).all():
            break

    x_distances = list(np.array(infos["x_position"]) - x_starts)
    all_states = torch.stack(all_states, dim=1)

    return all_lh_ratios, all_rewards, all_states, x_distances


def compute_surrogate_objective(
    batch_likelihood_ratios: list[torch.Tensor],
    batch_advantages: list[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Compute surrogate objective

    Inputs:
    batch_likelihood_ratios: list of n_env lists of max_steps scalar tensors
    batch_advantages: list of n_env lists of max_steps scalar tensors
    eps: clipping epsilon
    """
    surrogate = torch.tensor(0.0, device=batch_advantages[0].device)
    for lh_ratios, advantages in zip(batch_likelihood_ratios, batch_advantages):
        clipped_lr = torch.clamp(lh_ratios, 1 - eps, 1 + eps)
        clipped_product = torch.minimum(clipped_lr * advantages, lh_ratios * advantages)
        surrogate = surrogate + clipped_product.mean() / len(batch_advantages)
        
    return surrogate


def step(
    envs: AsyncVectorEnv,
    target_policy: ContinuousPolicy,
    old_policy: ContinuousPolicy,
    optimizer: optim.Optimizer,
    eps: float,
    *,
    max_steps: int,
    gamma: float,
    normalise_returns: bool,
    advantage: Advantage | None,
    device: torch.device,
) -> tuple[float, float]:
    """
    Collect one batch of parallel episodes and perform one gradient update.

    Returns
    -------
    mean_return    : mean total reward across episodes in the batch
    mean_x_distance: mean x displacement across episodes in the batch
    """

    with torch.no_grad():
        old_policy.load_state_dict(target_policy.state_dict())

    all_lh_ratios, all_rewards, all_states, x_distances = _collect_batch(
        envs, target_policy, old_policy, max_steps, device
    )

    batch_likelihood_ratios = []
    batch_likelihood_ratios = [
        torch.stack(lh_ratios) for lh_ratios in all_lh_ratios
    ]

    # compute_batch handles baseline.update() internally after computing per-episode advantages
    batch_advantages, episode_returns = advantage.compute_batch(
        all_rewards=all_rewards,
        all_states=all_states,
        gamma=gamma,
        device=device
    )

    if normalise_returns:
        batch_advantages = normalise_returns_batch(batch_advantages)

    surrogate_objective = compute_surrogate_objective(
        batch_likelihood_ratios, batch_advantages, eps
    )

    optimizer.zero_grad()
    loss = -surrogate_objective
    loss.backward()
    optimizer.step()

    return float(np.mean(episode_returns)), float(np.mean(x_distances))


def train_loop(
    env_name:            str = "Swimmer-v5",
    batch_size:          int = 16,
    hidden_dim:          int = 64,
    action_bound:        float = 1.0,
    covariance_scale:    float = 0.1,
    gamma:               float = 0.99,
    lr:                  float = 3e-4,
    eps:                 float = 0.2,
    num_updates:         int = 1000,
    max_steps:           int = 1000,
    normalise_returns:   bool = True,
    advantage_name:      str   = "QValue",
    checkpoint_base_dir: str = "checkpoints",
) -> MetricsManager:
    """
    Full training loop. Runs `num_updates` gradient steps, each collecting
    `batch_size` episodes in parallel. Saves results via MetricsManager at the end.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = AsyncVectorEnv([(lambda: gym.make(env_name)) for _ in range(batch_size)])

    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    covariance = covariance_scale * torch.eye(
        action_dim, dtype=torch.float32, device=device
    )
    target_policy = ContinuousPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        covariance=covariance,
        hidden_dim=hidden_dim,
    ).to(device)

    old_policy = ContinuousPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        covariance=covariance,
        hidden_dim=hidden_dim,
    ).to(device)

    advantage = make_advantage(advantage_name)

    if advantage is None:
        raise ValueError("PPO requires an advantage estimator. Got None from make_advantage().")

    optimizer = optim.Adam(target_policy.parameters(), lr=lr)

    run_dir = os.path.join(
        checkpoint_base_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    hyperparams: dict[str, Any] = {
        "env_name": env_name,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "action_bound": action_bound,
        "covariance_scale": covariance_scale,
        "gamma": gamma,
        "lr": lr,
        "eps": eps,
        "num_updates": num_updates,
        "max_steps": max_steps,
        "normalise_returns": normalise_returns,
        "checkpoint_base_dir": checkpoint_base_dir,
        "run_dir": run_dir,
        "device": device.type,
        "advantage_name": advantage_name,
    }

    mm = MetricsManager()

    print(f"Device        : {device}")
    print(
        f"Environment   : {env_name}  (state_dim={state_dim}, action_dim={action_dim})"
    )
    print(f"Parallel envs : {batch_size}")
    print(f"Updates       : {num_updates}")
    print(f"Run dir       : {run_dir}")
    print()

    try:
        for update_idx in range(1, num_updates + 1):
            t0 = time.time()

            mean_return, mean_x_dist = step(
                envs,
                target_policy,
                old_policy,
                optimizer,
                eps = eps,
                max_steps=max_steps,
                gamma=gamma,
                normalise_returns=normalise_returns,
                advantage=advantage,
                device=device,
            )

            mm.record(mean_return, mean_x_dist, target_policy)

            new_best = (
                "  *** NEW BEST ***" if mean_return >= mm.best_mean_return else ""
            )
            print(
                f"Update {update_idx:4d} | "
                f"MeanReturn {mean_return:+.3f} | "
                f"MeanXDist {mean_x_dist:.3f} | "
                f"Time {time.time() - t0:.2f}s"
                f"{new_best}"
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    finally:
        try:
            envs.close()
        except BaseException:
            pass  # workers are already dead on Windows Ctrl+C, ignore
        mm.save(run_dir, hyperparams, target_policy)

    return mm


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file to override defaults",
    )
    args = parser.parse_args()

    # Start with defaults; override with anything found in the config file
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    train_loop(**cfg)
