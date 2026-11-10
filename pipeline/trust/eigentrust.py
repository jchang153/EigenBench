"""Trust-matrix and EigenTrust utilities."""

from __future__ import annotations

import warnings

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
    s = torch.exp(logits - _overflow_shift(logits))
    lambda_i = torch.exp(log_lambda)

    sqrt_s = torch.sqrt(s)
    sqrt_s_sum = sqrt_s.sum(dim=1, keepdim=True)
    tie_terms = sqrt_s * (sqrt_s_sum - sqrt_s)
    tie_contribution = 0.5 * lambda_i * tie_terms

    S = s + tie_contribution
    Z_i = S.sum(dim=1, keepdim=True)
    T = S / Z_i
    if not torch.isfinite(T).all():
        raise ValueError("trust matrix is not finite")
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


def eigentrust(C, alpha: float = 0.0, tol: float = 1e-6, max_iter: int = 1000, verbose: bool = True,
               fallback_alpha: float = 0.05):
    """Stationary trust vector of ``C``; damps by ``fallback_alpha`` if not converged after ``max_iter``."""
    t = torch.full((C.size(0),), 1.0 / C.size(0), device=C.device, dtype=C.dtype)
    t, settled = _iterate(t, damp_matrix(C, alpha), tol, max_iter, verbose)
    if settled:
        return t
    if not 0 < fallback_alpha <= 1 or fallback_alpha <= alpha:
        raise RuntimeError(f"EigenTrust did not converge within {max_iter} iterations (alpha={alpha})")
    warnings.warn(
        f"EigenTrust did not converge within {max_iter} iterations; "
        f"continuing with damping alpha={fallback_alpha}",
        RuntimeWarning,
        stacklevel=2,
    )
    t, settled = _iterate(t, damp_matrix(C, fallback_alpha), tol, max_iter, verbose)
    if not settled:
        raise RuntimeError(
            f"EigenTrust did not converge within {max_iter} damped iterations (alpha={fallback_alpha})"
        )
    return t
