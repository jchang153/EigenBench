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


def compute_trust_matrix_ties(model, device: str = "cpu"):
    U = model.u.weight.data.to(device)
    V = model.v.weight.data.to(device)
    log_lambda = model.log_lambda.weight.data.to(device)

    logits = U @ V.t()
    centered = logits - logits.amax(dim=1, keepdim=True)
    half = 0.5 * centered
    # Sum the other models on either side of each column without subtraction.
    empty = torch.full_like(half[:, :1], -torch.inf)
    before = torch.cat([empty, half.logcumsumexp(dim=1)[:, :-1]], dim=1)
    after = torch.cat([half.flip([1]).logcumsumexp(dim=1)[:, :-1].flip([1]), empty], dim=1)
    others = torch.logaddexp(before, after)
    # Divide both win and tie weights by exp(max(log_lambda, 0)).
    wins = centered - log_lambda.clamp_min(0)
    ties = np.log(0.5) + log_lambda.clamp_max(0) + half + others
    trust = torch.softmax(torch.logaddexp(wins, ties), dim=1)
    if not torch.isfinite(trust).all():
        raise ValueError("trust matrix is not finite")
    return trust


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
