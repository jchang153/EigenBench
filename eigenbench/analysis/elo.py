"""EigenBench display scores for a panel's EigenTrust vector."""

from collections.abc import Sequence
from math import isclose, isfinite, log10


def elo_scores(trust: Sequence[float], num_models: int, *, direct_rating: bool = False) -> list[float]:
    """Apply the paper's Elo scale (§3.6), preserving the direct variant's display floor."""
    if num_models < 1 or len(trust) != num_models:
        raise ValueError("Expected one EigenTrust score per model")
    if any(not isfinite(t) or t < 0 for t in trust):
        raise ValueError("EigenTrust scores must be finite and nonnegative")
    # Account for float32 calculations and text exports from historical runs.
    if not isclose(sum(map(float, trust)), 1.0, rel_tol=1e-5):
        raise ValueError("EigenTrust scores must sum to one")
    if direct_rating:
        trust = [max(t, 1e-12) for t in trust]
    elif any(t == 0 for t in trust):
        raise ValueError("Pairwise Elo requires positive trust for every model")
    return [1500.0 + 400.0 * log10(num_models * t) for t in trust]
