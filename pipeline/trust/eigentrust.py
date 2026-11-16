"""Trust-matrix and EigenTrust utilities."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


def _overflow_shift(logits):
    """Per-row shift for logits whose exponent would overflow; rows are normalised, so it cancels."""
    if logits.max() < 80:
        return 0.0
    return logits.max(dim=1, keepdim=True).values


def compute_trust_matrix(model, device: str = "cpu"):
    U = model.u.weight.data.to(device)
    V = model.v.weight.data.to(device)
    S = U @ V.t()
    S = torch.exp(S - _overflow_shift(S))
    return S


def _log_trust_ties(logits, log_lambda):
    """The tie-aware trust matrix computed in log space, for values exp() can't hold."""
    l = logits.to(torch.float64)
    h = 0.5 * l
    n = l.shape[1]
    others = h.unsqueeze(1).expand(-1, n, -1).clone()  # [i, j, k] = h[i, k], excluding k = j
    others[:, torch.arange(n), torch.arange(n)] = float("-inf")
    tie = np.log(0.5) + log_lambda.to(torch.float64) + h + torch.logsumexp(others, dim=2)
    return torch.softmax(torch.logaddexp(l, tie), dim=1).to(logits.dtype)


def compute_trust_matrix_ties(model, device: str = "cpu"):
    U = model.u.weight.data.to(device)
    V = model.v.weight.data.to(device)
    log_lambda = model.log_lambda.weight.data.to(device)

    logits = U @ V.t()
    # Ordinary values take the original float path, so existing results are unchanged.
    if logits.abs().max() > 40 or log_lambda.abs().max() > 40:
        return _log_trust_ties(logits, log_lambda)
    s = torch.exp(logits)
    lambda_i = torch.exp(log_lambda)

    sqrt_s = torch.sqrt(s)
    sqrt_s_sum = sqrt_s.sum(dim=1, keepdim=True)
    tie_terms = sqrt_s * (sqrt_s_sum - sqrt_s)
    tie_contribution = 0.5 * lambda_i * tie_terms

    S = s + tie_contribution
    Z_i = S.sum(dim=1, keepdim=True)
    T = S / Z_i
    return T


def row_normalize(S):
    row_sums = S.sum(dim=1, keepdim=True)
    C = S / row_sums
    return C


def damp_matrix(C, alpha: float = 0.0):
    M = C.size(0)
    E = torch.full_like(C, 1.0 / M)
    return (1 - alpha) * C + alpha * E


def _iterate(t, T, tol: float, max_iter: int, verbose: bool):
    """Return the last iterate and whether it converged."""
    steps = tqdm(range(max_iter)) if verbose else range(max_iter)
    for _ in steps:
        t_next = t @ T
        if torch.norm(t_next - t, p=1) < tol:
            return t_next, True
        t = t_next
    return t, False


def _stationary(T, tol: float):
    """Solve t T = t, sum(t) = 1 directly; raise if the solution is not unique."""
    M = T.detach().cpu().to(torch.float64).numpy()
    n = M.shape[0]
    A = M.T - np.eye(n)
    if np.linalg.matrix_rank(A) < n - 1:
        raise RuntimeError("EigenTrust has no unique stationary vector: the trust matrix is reducible")
    b = np.zeros(n + 1)
    b[-1] = 1.0
    t = np.linalg.lstsq(np.vstack([A, np.ones(n)]), b, rcond=None)[0]
    if t.min() < -tol or np.abs(t @ M - t).sum() > tol:
        raise RuntimeError("EigenTrust stationary solve did not meet the residual tolerance")
    return torch.tensor(np.clip(t, 0, None), dtype=T.dtype, device=T.device)


def eigentrust(C, alpha: float = 0.0, tol: float = 1e-6, max_iter: int = 1000, verbose: bool = True):
    """Stationary trust vector of ``C`` damped by ``alpha``.

    Power iteration is used first. If it has not converged after ``max_iter`` steps
    (slow mixing, or a periodic matrix), the same matrix is solved directly.
    """
    T = damp_matrix(C, alpha)
    t = torch.full((C.size(0),), 1.0 / C.size(0), device=C.device, dtype=C.dtype)
    t, settled = _iterate(t, T, tol, max_iter, verbose)
    return t if settled else _stationary(T, tol)
