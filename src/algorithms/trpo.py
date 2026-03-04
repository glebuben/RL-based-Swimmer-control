from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any
import copy

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from src.nn.nn_policy import ContinuousPolicy
from src.utils.returns import normalise_returns_batch
from src.utils.metrics import MetricsManager

from src.utils.returns import compute_returns
from src.algorithms.conjugate_gradient import solve_CG
from src.algorithms.kl_divergence import FisherInfoOperator
from src.utils.returns import normalise_returns_batch
from src.advantages.base import Advantage
from src.advantages import make_advantage

from torch.distributions import kl_divergence


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


def _subsample_states_batch(
    all_states: torch.Tensor, subsample_fraction: float
) -> torch.Tensor:

    all_states = all_states.detach()
    ### bs x max_steps x state_dim -> (bs*max_steps) x state_dim
    all_states = all_states.view((-1, all_states.shape[-1]))

    sample_size = int(all_states.shape[0] * subsample_fraction)
    indices = torch.randint(
        0, all_states.shape[0], (sample_size,), device=all_states.device
    )
    return all_states[indices]


def compute_surrogate_objective(
    batch_likelihood_ratios: list[torch.Tensor],
    batch_advantages: list[torch.Tensor],
) -> torch.Tensor:
    """Compute surrogate objective

    Inputs:
    batch_likelihood_ratios: list of n_env lists of max_steps scalar tensors
    batch_advantages: list of n_env lists of max_steps scalar tensors
    """
    surrogate = torch.tensor(0.0, device=batch_advantages[0].device)
    for lh_ratios, advantages in zip(batch_likelihood_ratios, batch_advantages):
        surrogate = surrogate + (lh_ratios * advantages).mean() / len(batch_advantages)
    return surrogate


def kl_line_search(
    step_direction: list[torch.Tensor],
    initial_step_size: float,
    step_size_multiplier: float,
    max_kl: float,
    target_policy: ContinuousPolicy,
    old_policy: ContinuousPolicy,
    state_batch: torch.Tensor,
    batch_likelihood_ratios: torch.Tensor,
    batch_advantages: list[torch.Tensor],
):
    """Perform line search along step_direction on surrogate objective
    ensuring that KL constraint is satisfied. Updates target_policy weights in-place.
    Exponentially shrinks step size until either surrogate objective improves"""

    step_size = initial_step_size
    objective = -np.inf

    while True:

        old_params = copy.deepcopy(target_policy.state_dict())
        delta_w = torch.cat([s.view(-1) for s in step_direction]) * step_size
        target_policy.add_weights(delta_w)

        with torch.no_grad():
            target_dist = target_policy._get_distribution(state_batch)
            old_dist = old_policy._get_distribution(state_batch)
            kl = kl_divergence(old_dist, target_dist).mean()

            surrogate_objective = compute_surrogate_objective(
                batch_likelihood_ratios=batch_likelihood_ratios,
                batch_advantages=batch_advantages,
            )

            surrogate_objective = surrogate_objective.item()
            kl = kl.item()

            new_objective = surrogate_objective - kl

            if new_objective > objective and kl < max_kl:
                break
            else:
                step_size *= step_size_multiplier
                target_policy.load_state_dict(old_params)


def step(
    envs: AsyncVectorEnv,
    target_policy: ContinuousPolicy,
    old_policy: ContinuousPolicy,
    delta_kl: float,
    advantage: Advantage,
    *,
    max_steps: int,
    gamma: float,
    normalise_returns: bool,
    kl_subsample: float,
    line_search_step_multiplier: float,
    max_CG_iters: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Collect one batch of parallel episodes and perform one gradient update.

    Returns
    -------
    mean_return     : mean total reward across episodes in the batch
    mean_x_distance : mean x displacement across episodes in the batch
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
        device=device,
    )

    if normalise_returns:
        batch_advantages = normalise_returns_batch(batch_advantages)

    surrogate_objective = compute_surrogate_objective(
        batch_likelihood_ratios, batch_advantages
    )

    all_states_subsampled = _subsample_states_batch(all_states, kl_subsample)

    obj_grad = torch.autograd.grad(surrogate_objective, target_policy.parameters())
    obj_grad = torch.cat([g.view(-1) for g in obj_grad]).detach()

    # Compute the step direction using the conjugate gradient method
    fisher_info_op = FisherInfoOperator(
        target_policy, old_policy, all_states_subsampled
    )
    step_direction = solve_CG(fisher_info_op, obj_grad, max_iters=max_CG_iters)
    step_direction_flat = torch.cat([s.view(-1) for s in step_direction])

    FIM_by_step_direction = fisher_info_op(step_direction_flat)

    step_size = torch.sqrt(
        2 * delta_kl / (FIM_by_step_direction.dot(step_direction_flat) + 1e-8))

    kl_line_search(
        step_direction=step_direction,
        initial_step_size=step_size,
        step_size_multiplier=line_search_step_multiplier,
        max_kl=delta_kl,
        target_policy=target_policy,
        old_policy=old_policy,
        state_batch=all_states_subsampled,
        batch_likelihood_ratios=batch_likelihood_ratios,
        batch_advantages=batch_advantages,
    )

    return float(np.mean(episode_returns)), float(np.mean(x_distances))


def train_loop(
    env_name:                    str   = "Swimmer-v5",
    batch_size:                  int   = 16,
    hidden_dim:                  int   = 64,
    action_bound:                float = 1.0,
    covariance_scale:            float = 0.1,
    gamma:                       float = 0.99,
    delta_kl:                    float = 0.01,
    line_search_step_multiplier: float = 0.5,
    kl_subsample:                float = 0.1,
    max_CG_iters:                int   = 10,
    num_updates:                 int   = 1000,
    max_steps:                   int   = 1000,
    normalise_returns:           bool  = True,
    advantage_name:              str   = "QBaseline_Exponential_0.1",
    checkpoint_base_dir:         str   = "checkpoints",
) -> MetricsManager:
    """
    Full training loop. Runs `num_updates` gradient steps, each collecting
    `batch_size` episodes in parallel. Saves results via MetricsManager at the end.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = AsyncVectorEnv([
        (lambda: gym.make(env_name)) for _ in range(batch_size)
    ])

    state_dim  = envs.single_observation_space.shape[0]
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
        raise ValueError("TRPO requires an advantage estimator. Got None from make_advantage().")

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
        "delta_kl": delta_kl,
        "line_search_step_multiplier": line_search_step_multiplier,
        "kl_subsample": kl_subsample, 
        "max_CG_iters": max_CG_iters,
        "num_updates": num_updates, 
        "max_steps": max_steps,
        "normalise_returns": normalise_returns,
        "advantage_name": advantage_name,
        "checkpoint_base_dir": checkpoint_base_dir,
        "run_dir": run_dir, 
        "device": device.type,
    }

    mm = MetricsManager()

    print(f"Device        : {device}")
    print(f"Environment   : {env_name}  (state_dim={state_dim}, action_dim={action_dim})")
    print(f"Parallel envs : {batch_size}")
    print(f"Updates       : {num_updates}")
    print(f"Advantage     : {advantage_name}")
    print(f"Run dir       : {run_dir}")
    print()

    try:
        for update_idx in range(1, num_updates + 1):
            t0 = time.time()

            mean_return, mean_x_dist = step(
                envs, 
                target_policy, 
                old_policy, 
                delta_kl, 
                advantage,
                max_steps=max_steps,
                gamma=gamma,
                normalise_returns=normalise_returns,
                kl_subsample=kl_subsample,
                line_search_step_multiplier=line_search_step_multiplier,
                max_CG_iters=max_CG_iters,
                device=device,
            )

            mm.record(mean_return, mean_x_dist, target_policy)

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
            pass
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
