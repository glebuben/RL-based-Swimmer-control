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
from src.utils.returns import compute_returns, normalise_returns_batch
from src.utils.metrics import MetricsManager


def _collect_batch(
    envs: AsyncVectorEnv,
    policy: ContinuousPolicy,
    max_steps: int,
    device: torch.device,
) -> tuple[list[list[torch.Tensor]], list[list[float]], list[float]]:
    """
    Run all parallel envs for one episode each and collect experience.

    Swimmer only ever truncates (at max_steps), never terminates early,
    so x_position is simply read from infos at the start and end.

    Returns
    -------
    all_log_probs : one list of tensors per env
    all_rewards   : one list of floats per env
    x_distances   : x displacement per env over the episode
    """
    n_envs = envs.num_envs

    obs, infos = envs.reset()
    x_starts = np.array(infos["x_position"])

    all_log_probs = [[] for _ in range(n_envs)]
    all_rewards   = [[] for _ in range(n_envs)]

    for _ in range(max_steps):
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=device)
        action = policy.sample_actions(obs_t)
        lp     = policy.log_prob(obs_t, action)

        obs, rewards, terminated, truncated, infos = envs.step(
            action.detach().cpu().numpy()
        )

        for i in range(n_envs):
            all_log_probs[i].append(lp[i].squeeze())
            all_rewards[i].append(float(rewards[i]))

        if (terminated | truncated).all():
            break

    x_distances = list(np.array(infos["x_position"]) - x_starts)

    return all_log_probs, all_rewards, x_distances


def step(
    envs: AsyncVectorEnv,
    policy: ContinuousPolicy,
    optimizer: optim.Optimizer,
    *,
    max_steps: int,
    gamma: float,
    normalise_returns: bool,
    device: torch.device,
) -> tuple[float, float]:
    """
    Collect one batch of parallel episodes and perform one gradient update.

    Returns
    -------
    mean_return    : mean total reward across episodes in the batch
    mean_x_distance: mean x displacement across episodes in the batch
    """
    all_log_probs, all_rewards, x_distances = _collect_batch(
        envs, policy, max_steps, device
    )

    # Build per-episode tensors for the loss
    batch_log_probs = []
    batch_returns   = []
    episode_returns = []

    for log_probs, rewards in zip(all_log_probs, all_rewards):
        returns = compute_returns(rewards, gamma)
        episode_returns.append(sum(rewards))
        batch_log_probs.append(torch.stack(log_probs))
        batch_returns.append(torch.tensor(returns, dtype=torch.float32, device=device))

    if normalise_returns:
        batch_returns = normalise_returns_batch(batch_returns)

    # REINFORCE loss: push up log-probs of actions that led to high returns
    loss = torch.tensor(0.0, device=device)
    for lp, G in zip(batch_log_probs, batch_returns):
        loss = loss + torch.sum(-lp * G) / len(batch_log_probs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(np.mean(episode_returns)), float(np.mean(x_distances))


def train_loop(
    env_name:            str   = "Swimmer-v5",
    n_envs:              int   = 16,
    hidden_dim:          int   = 64,
    action_bound:        float = 1.0,
    covariance_scale:    float = 0.1,
    gamma:               float = 0.99,
    lr:                  float = 3e-4,
    num_updates:         int   = 1000,
    max_steps:           int   = 1000,
    normalise_returns:   bool  = True,
    checkpoint_base_dir: str   = "checkpoints",
) -> MetricsManager:
    """
    Full training loop. Runs `num_updates` gradient steps, each collecting
    `n_envs` episodes in parallel. Saves results via MetricsManager at the end.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = AsyncVectorEnv([
        (lambda: gym.make(env_name))
        for _ in range(n_envs)
    ])

    state_dim  = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    covariance = covariance_scale * torch.eye(action_dim, dtype=torch.float32, device=device)
    policy = ContinuousPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        covariance=covariance,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    run_dir = os.path.join(checkpoint_base_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    hyperparams: dict[str, Any] = {
        "env_name": env_name, "n_envs": n_envs,
        "hidden_dim": hidden_dim, "action_bound": action_bound,
        "covariance_scale": covariance_scale, "gamma": gamma,
        "lr": lr, "num_updates": num_updates, "max_steps": max_steps,
        "normalise_returns": normalise_returns,
        "checkpoint_base_dir": checkpoint_base_dir, "run_dir": run_dir,
        "device": device.type,
    }

    mm = MetricsManager()

    print(f"Device        : {device}")
    print(f"Environment   : {env_name}  (state_dim={state_dim}, action_dim={action_dim})")
    print(f"Parallel envs : {n_envs}")
    print(f"Updates       : {num_updates}")
    print(f"Run dir       : {run_dir}")
    print()

    try:
        for update_idx in range(1, num_updates + 1):
            t0 = time.time()

            mean_return, mean_x_dist = step(
                envs, policy, optimizer,
                max_steps=max_steps,
                gamma=gamma,
                normalise_returns=normalise_returns,
                device=device,
            )

            mm.record(mean_return, mean_x_dist, policy)

            new_best = "  *** NEW BEST ***" if mean_return >= mm.best_mean_return else ""
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
        mm.save(run_dir, hyperparams, policy)

    return mm


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Optional YAML config file to override defaults")
    args = parser.parse_args()

    # Start with defaults; override with anything found in the config file
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    train_loop(**cfg)