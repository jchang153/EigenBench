"""Trust-matrix and EigenTrust utilities."""

from __future__ import annotations

import torch
from tqdm import tqdm


def compute_trust_matrix(model, device: str = "cpu"):
    U = model.u.weight.data.to(device)
    V = model.v.weight.data.to(device)
    S = U @ V.t()
    S = torch.exp(S)
    return S


def compute_trust_matrix_ties(model, device: str = "cpu"):
    U = model.u.weight.data.to(device)
    V = model.v.weight.data.to(device)
    log_lambda = model.log_lambda.weight.data.to(device)

    logits = U @ V.t()
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


def eigentrust(C, alpha: float = 0.0, tol: float = 1e-6, max_iter: int = 1000, verbose: bool = True):
    T = damp_matrix(C, alpha)
    t = torch.full((T.size(0),), 1.0 / T.size(0), device=T.device)

    if verbose:
        for _ in tqdm(range(max_iter)):
            t_next = t @ T
            if torch.norm(t_next - t, p=1) < tol:
                break
            t = t_next
    else:
        for _ in range(max_iter):
            t_next = t @ T
            if torch.norm(t_next - t, p=1) < tol:
                break
            t = t_next

    return t_next
