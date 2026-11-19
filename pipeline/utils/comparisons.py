"""Compatibility imports for pairwise verdicts and comparison normalization."""

from eigenbench.analysis.comparisons import (
    extract_comparisons_with_ties_criteria as extract_comparisons_with_ties_criteria,
    handle_inconsistencies_with_ties_criteria as handle_inconsistencies_with_ties_criteria,
)
from eigenbench.protocol.pairwise import (
    _contiguous_prefix_len as _contiguous_prefix_len,
    _extract_valid_criterion_scores as _extract_valid_criterion_scores,
    validate_partial_criteria_response as validate_partial_criteria_response,
)


__all__ = [
    "extract_comparisons_with_ties_criteria",
    "handle_inconsistencies_with_ties_criteria",
    "validate_partial_criteria_response",
]
