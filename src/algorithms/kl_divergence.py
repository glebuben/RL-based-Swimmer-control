import torch
from src.nn.nn_policy import ContinuousPolicy


def _fisher_vector_product(
    KL_grad: torch.Tensor,
    target_params: list[torch.Tensor],
    vector: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Compute the product of the Fisher information matrix (FIM) with a given vector,
    using the gradient of KL divergence (KL_grad) between the target policy and old policy.
    """

    prod = KL_grad @ vector
    grad = torch.autograd.grad(prod, target_params, retain_graph=True)
    return torch.cat([g.view(-1) for g in grad])


def _get_KL_grad(
    target_policy: ContinuousPolicy,
    old_policy: ContinuousPolicy,
    state_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the gradient of the KL divergence between the target policy and old policy
    with respect to the target policy's parameters, given a batch of states.
    """
    dist_target = target_policy._get_distribution(state_batch)
    dist_old = old_policy._get_distribution(state_batch)

    kl_divergence = torch.distributions.kl_divergence(dist_old, dist_target).mean()

    grads = torch.autograd.grad(
        kl_divergence, target_policy.parameters(), create_graph=True
    )

    return torch.cat([g.view(-1) for g in grads])


class FisherInfoOperator:
    def __init__(
        self,
        target_policy: ContinuousPolicy,
        old_policy: ContinuousPolicy,
        state_batch: torch.Tensor,
        damping: float = 0.1,
    ):

        self._target_params = list(target_policy.parameters())
        self._target_policy = target_policy
        self._old_policy = old_policy
        self._damping = damping
        self._set_state_batch(state_batch)

    def _set_state_batch(self, state_batch: torch.Tensor):
        self._kl_grad = _get_KL_grad(self._target_policy, self._old_policy, state_batch)

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        prod = _fisher_vector_product(self._kl_grad, self._target_params, vector)
        return prod + self._damping * vector
