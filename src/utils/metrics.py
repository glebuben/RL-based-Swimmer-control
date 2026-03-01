"""
MetricsManager
--------------
Accumulates per-update training metrics in memory throughout a run.
On `.save()` it writes a self-contained run directory:

  run_dir/
    history.npz       – mean_returns and x_distances as numpy arrays
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

    def __init__(self) -> None:
        self._mean_returns:     list[float] = []
        self._x_distances:      list[float] = []
        self._best_mean_return: float       = -float("inf")
        self._best_policy_sd:   dict | None = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, mean_return: float, x_distance: float, policy) -> None:
        """Append one update's metrics and snapshot policy if it's the best so far."""
        self._mean_returns.append(float(mean_return))
        self._x_distances.append(float(x_distance))

        if mean_return > self._best_mean_return:
            self._best_mean_return = mean_return
            self._best_policy_sd   = deepcopy(policy.state_dict())

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def best_mean_return(self) -> float:
        return self._best_mean_return

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, run_dir: str, hyperparams: dict[str, Any], policy) -> None:
        """Write all run artefacts to run_dir (created if needed)."""
        import torch

        os.makedirs(run_dir, exist_ok=True)

        # 1) Learning history
        np.savez_compressed(
            os.path.join(run_dir, "history.npz"),
            mean_returns=np.array(self._mean_returns, dtype=np.float32),
            x_distances =np.array(self._x_distances,  dtype=np.float32),
        )

        # 2) Hyperparameters + summary
        best = self._best_mean_return
        summary = {
            "num_updates_completed": len(self._mean_returns),
            "best_mean_return":      float(best) if np.isfinite(best) else None,
            "final_mean_return":     float(self._mean_returns[-1]) if self._mean_returns else None,
            "best_x_distance":       float(max(self._x_distances)) if self._x_distances else None,
            "final_x_distance":      float(self._x_distances[-1])  if self._x_distances else None,
        }
        with open(os.path.join(run_dir, "hyperparams.yaml"), "w") as f:
            yaml.dump({"hyperparams": hyperparams, "summary": summary},
                      f, default_flow_style=False, sort_keys=False)

        # 3) Policy checkpoints
        if self._best_policy_sd is not None:
            current_sd = deepcopy(policy.state_dict())
            policy.load_state_dict(self._best_policy_sd)
            policy.save(os.path.join(run_dir, "policy_best.pt"))
            policy.load_state_dict(current_sd)
        else:
            policy.save(os.path.join(run_dir, "policy_best.pt"))

        policy.save(os.path.join(run_dir, "policy_final.pt"))

        print(f"[MetricsManager] Run saved → {run_dir}")
        print(f"  history.npz      ({len(self._mean_returns)} updates)")
        print(f"  hyperparams.yaml")
        print(f"  policy_best.pt   (best mean_return = {best:.3f})" if np.isfinite(best)
              else "  policy_best.pt   (no updates completed)")
        print(f"  policy_final.pt")