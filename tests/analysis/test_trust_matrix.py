"""Numeric behavior of pairwise trust matrices."""

from types import SimpleNamespace

import torch


def _btd(scores, log_lambda):
    """A stand-in model whose u @ v.T equals ``scores``."""
    n = scores.shape[0]
    layer = lambda w: SimpleNamespace(weight=SimpleNamespace(data=w))
    return SimpleNamespace(u=layer(torch.eye(n)), v=layer(scores.t().contiguous()),
                           log_lambda=layer(log_lambda))


def test_trust_matrix_is_unchanged_by_a_score_shift():
    from pipeline.trust import compute_trust_matrix, compute_trust_matrix_ties, row_normalize

    scores = torch.tensor([[0.2, 1.0, -0.5], [0.0, 0.3, 0.9], [1.2, -0.4, 0.1]])
    log_lambda = torch.full((3, 1), 0.2)
    base = compute_trust_matrix_ties(_btd(scores, log_lambda))
    shifted = compute_trust_matrix_ties(_btd(scores + 100, log_lambda))
    assert torch.isfinite(shifted).all() and torch.allclose(base, shifted, atol=1e-5)
    plain = row_normalize(compute_trust_matrix(_btd(scores + 100, log_lambda)))
    assert torch.allclose(plain, row_normalize(compute_trust_matrix(_btd(scores, log_lambda))), atol=1e-5)


def test_trust_matrix_keeps_ties_that_outweigh_an_underflowing_win():
    from pipeline.trust import compute_trust_matrix_ties

    # Row 0: s = [e^120, 1], lambda = e^60, so S = [1.5, 0.5] * e^120.
    scores = torch.tensor([[120.0, 0.0], [0.0, 0.0]])
    T = compute_trust_matrix_ties(_btd(scores, torch.tensor([[60.0], [0.0]])))
    assert torch.allclose(T[0], torch.tensor([0.75, 0.25]), atol=1e-6)
    T = compute_trust_matrix_ties(_btd(torch.zeros(2, 2), torch.full((2, 1), 100.0)))
    assert torch.allclose(T, torch.full((2, 2), 0.5))


def test_trust_matrix_keeps_ties_when_the_other_strength_rounds_away():
    from pipeline.trust import compute_trust_matrix_ties

    scores = torch.tensor([[20.0, -20.0], [0.0, 0.0]])
    trust = compute_trust_matrix_ties(_btd(scores, torch.tensor([[20.0], [0.0]])))
    torch.testing.assert_close(trust[0], torch.tensor([0.75, 0.25]))
