"""
REINFORCE for continuous-action Gymnasium environments (e.g. Swimmer-v5).

Public API
----------
step(env, policy, optimizer, ...)  ->  (mean_return, domain_metrics)
train_loop(...)                    ->  MetricsManager
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

from src.nn.nn_policy import ContinuousPolicy
from src.utils.returns import compute_returns, normalise_returns_batch
from src.utils.metrics import MetricsManager


# ---------------------------------------------------------------------------
# Internal: single episode rollout
# ---------------------------------------------------------------------------

def _collect_episode(
    env: gym.Env,
    policy: ContinuousPolicy,
    max_steps: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[float], dict[str, float]]:
    """
    Roll out one full episode.

    Returns
    -------
    log_probs      : scalar log-prob tensor per step
    rewards        : float reward per step
    domain_metrics : env-specific scalars, e.g. {"x_distance": ...}
    """
    obs, info = env.reset()
    x_start: float = float(info.get("x_position", 0.0))

    log_probs: list[torch.Tensor] = []
    rewards:   list[float]        = []

    for _ in range(max_steps):
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = policy.sample_actions(obs_t)        # (1, action_dim)
        lp     = policy.log_prob(obs_t, action)      # (1,)
        log_probs.append(lp.squeeze())

        obs, reward, terminated, truncated, info = env.step(
            action.squeeze(0).detach().cpu().numpy()
        )
        rewards.append(float(reward))

        if terminated or truncated:
            break

    x_end      = float(info.get("x_position", 0.0))
    domain_metrics = {"x_distance": x_end - x_start}

    return log_probs, rewards, domain_metrics


# ---------------------------------------------------------------------------
# step  (one batch of episodes + one gradient update)
# ---------------------------------------------------------------------------

def step(
    env: gym.Env,
    policy: ContinuousPolicy,
    optimizer: optim.Optimizer,
    *,
    episodes_per_update: int,
    max_steps: int,
    gamma: float,
    normalise_returns: bool,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """
    Collect one batch of episodes and perform one REINFORCE gradient update.

    Parameters
    ----------
    env                 : Gymnasium environment instance.
    policy              : ContinuousPolicy to act with and update.
    optimizer           : Optimizer attached to policy parameters.
    episodes_per_update : Episodes to collect before each gradient step.
    max_steps           : Maximum timesteps per episode.
    gamma               : Discount factor.
    normalise_returns   : Standardise returns across the batch before loss.
    device              : Torch device.

    Returns
    -------
    mean_return    : Mean undiscounted return across collected episodes.
    domain_metrics : Dict of mean env-specific metrics, e.g. {"x_distance": …}.
    """
    batch_log_probs:     list[torch.Tensor]       = []
    batch_returns:       list[torch.Tensor]       = []
    episode_returns:     list[float]              = []
    batch_domain:        list[dict[str, float]]   = []

    # ---- collect ----
    for _ in range(episodes_per_update):
        log_probs, rewards, ep_domain = _collect_episode(env, policy, max_steps, device)

        if not rewards:
            continue

        returns = compute_returns(rewards, gamma)

        episode_returns.append(sum(rewards))
        batch_domain.append(ep_domain)
        batch_log_probs.append(torch.stack(log_probs))
        batch_returns.append(
            torch.tensor(returns, dtype=torch.float32, device=device)
        )

    # ---- optional return normalisation ----
    if normalise_returns:
        batch_returns = normalise_returns_batch(batch_returns)

    # ---- REINFORCE loss:  -E[ Σ_t log π(a_t|s_t) · G_t ] ----
    loss = torch.tensor(0.0, device=device)
    n = len(batch_log_probs)
    for lp, G in zip(batch_log_probs, batch_returns):
        loss = loss + torch.sum(-lp * G) / n

    # ---- gradient update ----
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ---- aggregate metrics ----
    mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0

    # Average each domain metric key across the episode batch
    all_keys = {k for d in batch_domain for k in d}
    mean_domain: dict[str, float] = {
        k: float(np.mean([d[k] for d in batch_domain if k in d]))
        for k in all_keys
    }

    return mean_return, mean_domain


# ---------------------------------------------------------------------------
# train_loop
# ---------------------------------------------------------------------------

def train_loop(
    # environment
    env_name:            str   = "Swimmer-v5",
    # policy architecture
    hidden_dim:          int   = 64,
    action_bound:        float = 1.0,
    covariance_scale:    float = 0.1,
    # training
    gamma:               float = 0.99,
    lr:                  float = 3e-4,
    num_updates:         int   = 1000,
    episodes_per_update: int   = 16,
    max_steps:           int   = 1000,
    normalise_returns:   bool  = True,
    # checkpointing
    checkpoint_base_dir: str   = "checkpoints",
) -> MetricsManager:
    """
    Full REINFORCE training loop.

    Calls `step()` each update, records via MetricsManager (which owns
    best-policy tracking), and persists everything via `mm.save()` at the
    end — or on KeyboardInterrupt via a finally block.

    Returns
    -------
    MetricsManager with the completed run's full history.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- environment ----
    env        = gym.make(env_name)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ---- policy ----
    covariance = (covariance_scale * torch.eye(action_dim, dtype=torch.float32)).to(device)
    policy = ContinuousPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        covariance=covariance,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # ---- run directory ----
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(checkpoint_base_dir, f"run_{run_id}")

    # ---- hyperparams dict (everything non-network) ----
    hyperparams: dict[str, Any] = {
        "env_name":            env_name,
        "gamma":               gamma,
        "lr":                  lr,
        "num_updates":         num_updates,
        "episodes_per_update": episodes_per_update,
        "max_steps":           max_steps,
        "normalise_returns":   normalise_returns,
        "hidden_dim":          hidden_dim,
        "action_bound":        action_bound,
        "covariance_scale":    covariance_scale,
        "checkpoint_base_dir": checkpoint_base_dir,
        "run_dir":             run_dir,
        "device":              device.type,
    }

    mm = MetricsManager()

    print(f"Device      : {device}")
    print(f"Environment : {env_name}  |  state_dim={state_dim}  action_dim={action_dim}")
    print(f"Updates     : {num_updates}  |  episodes/update={episodes_per_update}")
    print(f"Run dir     : {run_dir}")
    print()

    try:
        for update_idx in range(1, num_updates + 1):
            t0 = time.time()

            mean_return, domain_metrics = step(
                env, policy, optimizer,
                episodes_per_update=episodes_per_update,
                max_steps=max_steps,
                gamma=gamma,
                normalise_returns=normalise_returns,
                device=device,
            )

            # MetricsManager owns best-policy snapshot tracking
            mm.record(mean_return, policy, domain_metrics)

            new_best = "  *** NEW BEST ***" if mean_return >= mm.best_mean_return else ""

            domain_str = "  ".join(
                f"{k} {v:.3f}" for k, v in domain_metrics.items()
            )
            print(
                f"Update {update_idx:4d} | "
                f"MeanReturn {mean_return:+.3f} | "
                f"{domain_str} | "
                f"Time {time.time() - t0:.2f}s"
                f"{new_best}"
            )

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training interrupted by user (Ctrl+C)")
        print("=" * 60)

    finally:
        env.close()
        mm.save(run_dir, hyperparams, policy)

    return mm


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Train REINFORCE on a Gymnasium environment."
    )
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg: dict = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    env_cfg    = cfg.get("env",    {})
    policy_cfg = cfg.get("policy", {})

    train_loop(
        env_name            = env_cfg.get("env_name",            "Swimmer-v5"),
        hidden_dim          = policy_cfg.get("hidden_dim",       64),
        action_bound        = policy_cfg.get("action_bound",     1.0),
        covariance_scale    = policy_cfg.get("covariance_scale", 0.1),
        gamma               = cfg.get("gamma",               0.99),
        lr                  = cfg.get("lr",                  3e-4),
        num_updates         = cfg.get("num_updates",         1000),
        episodes_per_update = cfg.get("episodes_per_update", 16),
        max_steps           = cfg.get("max_steps",           1000),
        normalise_returns   = cfg.get("normalise_returns",   True),
        checkpoint_base_dir = cfg.get("checkpoint_base_dir", "checkpoints"),
    )