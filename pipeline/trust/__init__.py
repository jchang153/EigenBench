"""Trust-matrix and EigenTrust routines."""

from .eigentrust import (
    compute_trust_matrix,
    compute_trust_matrix_ties,
    row_normalize,
    eigentrust,
)
from .direct_rating import (
    DirectTrustResult,
    SUPPORTED_DIRECT_NORMALIZATIONS,
    aggregate_direct_records,
    build_direct_trust,
    normalize_direct_scores,
)

__all__ = [
    "compute_trust_matrix",
    "compute_trust_matrix_ties",
    "row_normalize",
    "eigentrust",
    "DirectTrustResult",
    "SUPPORTED_DIRECT_NORMALIZATIONS",
    "aggregate_direct_records",
    "build_direct_trust",
    "normalize_direct_scores",
]
