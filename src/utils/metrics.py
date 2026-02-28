"""
MetricsManager
--------------
Accumulates per-update training metrics in memory throughout a run.
On `.save()` it writes a self-contained run directory:

  run_dir/
    history.npz       – np.arrays for mean_returns + every domain metric
    hyperparams.yaml  – all non-network hyperparameters + run summary
    policy_best.pt    – policy snapshot at the update with highest mean_return
    policy_final.pt   – policy snapshot at the moment .save() is called
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import numpy as np
import yaml


class MetricsManager:
    """
    Accumulate training metrics and persist them to disk in one shot.

    Domain-specific metrics (e.g. x_distance for Swimmer) are passed as a
    plain dict so this class stays environment-agnostic.

    Usage
    -----
    >>> mm = MetricsManager()
    >>> for update in range(num_updates):
    ...     mean_ret, metrics = step(...)          # metrics is a dict
    ...     mm.record(mean_ret, policy, metrics)
    >>> mm.save(run_dir, hyperparams)
    """

    def __init__(self) -> None:
        self._mean_returns:      list[float]       = []
        self._domain_metrics:    dict[str, list[float]] = {}   # key → history
        self._best_mean_return:  float             = -float("inf")
        self._best_policy_sd:    dict | None       = None      # deepcopy of state_dict

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        mean_return: float,
        policy,                          # ContinuousPolicy (used for best snapshot)
        domain_metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Append one update's worth of metrics and update the best-policy snapshot.

        Parameters
        ----------
        mean_return    : Mean undiscounted return across the episode batch.
        policy         : Current policy network (state_dict copied if new best).
        domain_metrics : Optional dict of environment-specific scalars,
                         e.g. {"x_distance": 3.14}.  Keys are consistent
                         across calls; missing keys are filled with NaN.
        """
        self._mean_returns.append(float(mean_return))

        # --- domain metrics: register new keys, append values ---
        if domain_metrics:
            # Backfill any newly-seen key with NaN for previous steps
            for key, val in domain_metrics.items():
                if key not in self._domain_metrics:
                    n_prev = len(self._mean_returns) - 1
                    self._domain_metrics[key] = [float("nan")] * n_prev
                self._domain_metrics[key].append(float(val))

        # Pad any key that was absent this step
        n_now = len(self._mean_returns)
        for key, history in self._domain_metrics.items():
            if len(history) < n_now:
                history.append(float("nan"))

        # --- best policy snapshot ---
        if mean_return > self._best_mean_return:
            self._best_mean_return = mean_return
            self._best_policy_sd   = deepcopy(policy.state_dict())

    # ------------------------------------------------------------------
    # Read-only views  (useful for live logging)
    # ------------------------------------------------------------------

    @property
    def latest_mean_return(self) -> float | None:
        return self._mean_returns[-1] if self._mean_returns else None

    @property
    def best_mean_return(self) -> float:
        return self._best_mean_return

    def latest_domain_metric(self, key: str) -> float | None:
        hist = self._domain_metrics.get(key)
        return hist[-1] if hist else None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        run_dir: str,
        hyperparams: dict[str, Any],
        policy,                          # ContinuousPolicy (final weights)
    ) -> None:
        """
        Write the full run artefacts to *run_dir* (created if needed).

        Files written
        -------------
        history.npz      – compressed numpy archive:
                             'mean_returns' + one array per domain metric key
        hyperparams.yaml – hyperparameters dict + auto-generated summary section
        policy_best.pt   – snapshot of the policy at its best mean_return
        policy_final.pt  – snapshot of the policy at call time
        """
        import torch

        os.makedirs(run_dir, exist_ok=True)

        # 1) Learning-history arrays
        arrays: dict[str, np.ndarray] = {
            "mean_returns": np.array(self._mean_returns, dtype=np.float32),
        }
        for key, history in self._domain_metrics.items():
            arrays[key] = np.array(history, dtype=np.float32)

        np.savez_compressed(os.path.join(run_dir, "history.npz"), **arrays)

        # 2) Hyperparameters YAML with summary statistics
        summary: dict[str, Any] = {
            "num_updates_completed": len(self._mean_returns),
            "best_mean_return":      float(self._best_mean_return),
            "final_mean_return":     float(self._mean_returns[-1]) if self._mean_returns else None,
        }
        for key, history in self._domain_metrics.items():
            valid = [v for v in history if not np.isnan(v)]
            summary[f"best_{key}"]  = float(max(valid)) if valid else None
            summary[f"final_{key}"] = float(history[-1]) if history else None

        yaml_payload = {"hyperparams": hyperparams, "summary": summary}
        with open(os.path.join(run_dir, "hyperparams.yaml"), "w") as f:
            yaml.dump(yaml_payload, f, default_flow_style=False, sort_keys=False)

        # 3) Policy checkpoints
        # best snapshot: temporarily swap weights, save, restore
        if self._best_policy_sd is not None:
            _current_sd = deepcopy(policy.state_dict())
            policy.load_state_dict(self._best_policy_sd)
            policy.save(os.path.join(run_dir, "policy_best.pt"))
            policy.load_state_dict(_current_sd)
        else:
            policy.save(os.path.join(run_dir, "policy_best.pt"))   # fallback

        # final snapshot
        policy.save(os.path.join(run_dir, "policy_final.pt"))

        print(f"[MetricsManager] Run saved → {run_dir}")
        print(f"  history.npz      ({len(self._mean_returns)} updates, "
              f"keys: mean_returns" +
              (f", {', '.join(self._domain_metrics)}" if self._domain_metrics else "") + ")")
        print(f"  hyperparams.yaml")
        print(f"  policy_best.pt   (best mean_return = {self._best_mean_return:.3f})")
        print(f"  policy_final.pt")