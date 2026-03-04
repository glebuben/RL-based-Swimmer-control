# RL-based Swimmer Control

Comparison of policy gradient methods — REINFORCE and TRPO — on the MuJoCo `Swimmer-v5` environment.

---

## Directory Structure

```
checkpoints/
└── run_YYYYMMDD_HHMMSS/
    ├── history.npz         # Numpy arrays of mean_returns and x_distances
    ├── hyperparams.yaml    # All hyperparameters + run summary statistics
    ├── policy_best.pt      # Policy snapshot at highest mean return
    └── policy_final.pt     # Policy snapshot at end of training

src/
├── advantages/
│   ├── base.py             # Abstract Advantage class with compute(), update(), compute_batch()
│   ├── q_baseline.py       # QBaselineAdvantage: A = G_t - b, where b is a running baseline
│   └── q_value.py          # QValueAdvantage: A = G_t (raw discounted returns, no adjustment)
│
├── algorithms/
│   ├── reinforce.py        # REINFORCE training loop with optional baseline support
│   ├── trpo.py             # TRPO training loop with pluggable advantage estimators
│   ├── ppo.py              # PPO training loop with pluggable advantage estimators
│   ├── conjugate_gradient.py  # Conjugate gradient solver for the Fisher-vector product system
│   └── kl_divergence.py    # Fisher information matrix operator for KL constraint
│
├── baselines/
│   ├── base.py             # Abstract Baseline class with update() and get()
│   └── exponential.py      # ExponentialBaseline: exponential moving average of batch returns
│
├── nn/
│   └── nn_policy.py        # ContinuousPolicy network: tanh-bounded Gaussian policy with save/load
│
├── utils/
│   ├── metrics.py          # MetricsManager: accumulates training metrics and saves run artefacts
│   └── returns.py          # compute_returns() and normalise_returns_batch() utilities
│
└── visualization/
    └── policy_rollout.py   # Loads a checkpoint and records a rollout as a GIF
```

---

## Environment: Swimmer-v5

The Swimmer is a 3-link planar robot submerged in a fluid. It has no legs — locomotion is achieved purely by undulating its body joints.
### State Space $\mathcal{S}$
![alt text](artifacts/swimmer_legend.png)

The observation vector $s_t \in \mathbb{R}^8$ contains:

| Component | Description |
|---|---|
| $q_1, q_2$ | Joint angles of the two actuated links |
| $\dot{q}_0, \dot{q}_1, \dot{q}_2$ | Angular velocities of all three links |
| $\dot{x}, \dot{y}$ | Linear velocity of the torso |
| $\theta$ | Orientation of the torso |

### Action Space $\mathcal{A}$

The action $a_t \in [-1, 1]^2$ controls the torques applied to the two joints:

| Component | Description |
|---|---|
| $a_1$ | Torque applied to joint 1 (between links 1 and 2) |
| $a_2$ | Torque applied to joint 2 (between links 2 and 3) |

### Task and Reward

The goal is to maximise forward displacement along the x-axis. The reward at each timestep is:

$$r_t = v_x - c \cdot \|a_t\|^2$$

where $v_x$ is the forward velocity of the torso, $\|a_t\|^2$ is the squared norm of the applied torques, and $c = 10^{-4}$ is a small control cost penalty. Episodes run for a fixed 1000 timesteps (no early termination).

---

## Methods

We compare REINFORCE and TRPO, two policy gradient algorithms from opposite ends of the sample efficiency vs. implementation complexity spectrum. Both optimise the expected discounted return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

### REINFORCE

REINFORCE is a Monte Carlo policy gradient method. After collecting a batch of full episodes, the policy parameters are updated by ascending the gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the discounted return from step $t$.

#### Without Baseline

The standard update uses the raw returns $G_t$ as the advantage estimate. This is unbiased but can have high variance, especially early in training when returns fluctuate significantly between episodes.

#### With Exponential Baseline

To reduce variance, we subtract a state-independent baseline $b$ from the returns:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (G_t - b) \right]$$

Subtracting a baseline does not introduce bias since $\mathbb{E}[\nabla_\theta \log \pi_\theta \cdot b] = 0$. We use an **exponential moving average** of batch mean returns as the baseline:

$$b_{k+1} = (1 - \alpha) \cdot b_k + \alpha \cdot \bar{G}_k$$

with $\alpha = 0.05$, where $\bar{G}_k$ is the mean return of batch $k$. The baseline is initialised to the first observed batch mean to avoid cold-start bias from zero.

---

### TRPO

Trust Region Policy Optimisation addresses the instability of large gradient steps in REINFORCE by constraining each update to a trust region defined by a KL divergence bound. Rather than a gradient step, TRPO solves the constrained optimisation problem:

$$\max_\theta \; \mathcal{L}(\theta) = \mathbb{E}_{s, a \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \cdot A(s, a) \right]$$

$$\text{subject to} \quad \mathbb{E}_s \left[ D_{\text{KL}} \left( \pi_{\theta_{\text{old}}}(\cdot|s) \,\|\, \pi_\theta(\cdot|s) \right) \right] \leq \delta$$

The constraint ensures the new policy does not deviate too far from the old one, guaranteeing monotonic improvement under certain conditions. The update direction is computed using the **natural policy gradient**:

$$\theta \leftarrow \theta + \sqrt{\frac{2\delta}{s^\top F s}} \cdot s, \qquad s = F^{-1} \nabla_\theta \mathcal{L}(\theta)$$

where $F$ is the Fisher information matrix. Since $F$ is too large to invert directly, we use the **conjugate gradient** method to compute $F^{-1}g$ via Fisher-vector products, followed by a **backtracking line search** to satisfy the KL constraint.

#### TRPO with Q-value Advantage

The advantage is estimated as the raw discounted return:

$$A(s_t, a_t) = G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$$

This is an estimate of the action-value function $Q^{\pi}(s_t, a_t)$ under the current policy.

#### TRPO with Baseline Advantage

The advantage is estimated as $Q - V$, where $V$ is approximated by the same exponential moving average baseline described above:

$$A(s_t, a_t) = G_t - b$$

This reduces variance in the surrogate objective gradient while remaining unbiased, similar to the REINFORCE with baseline case.

---

### PPO 

Proximal Policy Optimisation is a more practical variant of TRPO that replaces the hard KL constraint with a clipped surrogate objective. The update maximises:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_{s, a \sim \pi_{\theta_{\text{old}}}} \left[ \min \left( r_t(\theta) A(s, a), \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A(s, a) \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ is the probability ratio. The clipping prevents updates that would change the policy too much in one step, while still allowing multiple epochs of minibatch updates on the same data. 

While original paper proposes using a value function approximation, we use only returns-based estimates of value function. 

Also, for simplicity, we perform only one gradient update per batch of data, rather than multiple epochs of minibatch updates.

---

## Results and Conclusions

> _Graphs and quantitative comparison to be added here._

Both algorithms are trained with 16 parallel environments (`AsyncVectorEnv`) for sample efficiency, with return normalisation enabled. Metrics tracked per update are mean episode return and mean x-displacement.