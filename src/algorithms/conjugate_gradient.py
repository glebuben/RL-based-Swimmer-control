import torch
from src.algorithms.kl_divergence import FisherInfoOperator


def unflatten_like(vector: torch.Tensor, model: torch.nn.Module) -> list[torch.Tensor]:
    """Convert a flat vector into a list of tensors
    with the same shapes as the parameters of the given model"""
    params = []
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        params.append(vector[idx : idx + numel].view_as(p))
        idx += numel
    return params


def solve_CG(
    operator: FisherInfoOperator, rhs: torch.Tensor, max_iters=10, tol=1e-10
) -> torch.Tensor:
    """
    Conjugate Gradient method to solve the linear system Ax = b, where A is represented by the operator.

    Parameters
    ----------
    operator : FisherInfoOperator
        The operator representing the matrix A.
    rhs : torch.Tensor
        The right-hand side vector b.
    max_iters : int, optional
        Maximum number of iterations (default is 10).
    tol : float, optional
        Tolerance for convergence (default is 1e-10).

    Returns
    -------
    x : torch.Tensor
        The solution vector x.
    """
    x = torch.zeros_like(rhs, device=rhs.device)
    r = rhs.clone()
    p = r.clone()
    rs_old = r.dot(r)

    for i in range(max_iters):
        Ap = operator(p)
        alpha = rs_old / (p.dot(Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r.dot(r)

        if rs_new < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return unflatten_like(x, operator._target_policy)
