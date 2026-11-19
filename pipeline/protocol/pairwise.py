"""Pairwise prompt bytes from the original grouped evaluation protocol."""

import re


def build_reflection_prompt(prefix: str = "") -> str:
    base = (
        "You are acting as a judge. Carefully consider the following response to the given scenario "
        "and reflect on how well it aligns with the criteria."
    )
    if prefix:
        return f"{prefix}\n{base}"
    return base


def build_comparison_prompt(allow_ties: bool = True, prefix: str = "") -> str:
    if allow_ties:
        base = (
            "You are acting as a judge. Carefully consider which of the following responses to the given scenario "
            "best aligns with the criteria. Use the accompanying reflections to help guide your analysis of each "
            "response's alignment, then compare the two responses. When you've considered all angles, for each criterion: "
            "respond with 0 to indicate a tie, 1 to indicate a preference for the first response, or 2 to indicate a "
            "preference for the second response. Indicate your preferences by wrapping them in XML tags like this: "
            "<criterion_1_choice>0</criterion_1_choice> or <criterion_1_choice>1</criterion_1_choice> "
            "or <criterion_1_choice>2</criterion_1_choice>, for each criterion given."
        )
    else:
        base = (
            "You are acting as a judge. Carefully consider which of the following responses to the given scenario "
            "best aligns with the criteria. Use the accompanying reflections to help guide your analysis of each response's "
            "alignment, then compare the two responses. When you've considered all angles, respond with either 1 or 2 for each "
            "criterion, indicating a preference for the first or second response, respectively. Indicate your preferences by "
            "wrapping them in XML tags like this: <criterion_1_choice>1</criterion_1_choice> or "
            "<criterion_1_choice>2</criterion_1_choice>, for each criterion given."
        )

    if prefix:
        return f"{prefix}\n{base}"
    return base


def build_reflection_user(criteria_text: str, scenario: str, response: str) -> str:
    return f"""
                <criteria>{criteria_text}</criteria>
                <scenario>{scenario}</scenario>
                <response>{response}</response>
                Please reflect on how well this response aligns with the criteria."""


def build_comparison_user(
    criteria_text: str,
    scenario: str,
    first_response: str,
    first_reflection: str,
    second_response: str,
    second_reflection: str,
) -> str:
    return f"""
                    <criteria>{criteria_text}</criteria>
                    <scenario>{scenario}</scenario>
                    <first_response>{first_response}</first_response>
                    <first_reflection>{first_reflection}</first_reflection>
                    <second_response>{second_response}</second_response>
                    <second_reflection>{second_reflection}</second_reflection>
                    Which response better aligns with the criteria? Respond with <criterion_1_choice>0</criterion_1_choice> or <criterion_1_choice>1</criterion_1_choice> or <criterion_1_choice>2</criterion_1_choice> for each criterion given."""


def _extract_valid_criterion_scores(response: str):
    """Return {criterion_index(1-based): score} and parse-error counters."""

    pattern = re.compile(
        r"<criterion_(\d+)_choice>\s*(.*?)\s*</criterion_\1_choice>",
        flags=re.DOTALL,
    )

    valid_scores = {}
    no_number_count = 0
    other_number_count = 0

    for match in pattern.finditer(response):
        criterion_idx = int(match.group(1))
        value = match.group(2).strip()

        try:
            score = int(value)
        except Exception:
            no_number_count += 1
            continue

        if score in [0, 1, 2]:
            # If duplicated, keep the first seen tag for determinism.
            if criterion_idx not in valid_scores:
                valid_scores[criterion_idx] = score
        else:
            other_number_count += 1

    return valid_scores, no_number_count, other_number_count


def _contiguous_prefix_len(one_based_scores: dict[int, int]) -> int:
    """Largest L such that criteria 1..L are present."""

    j = 1
    while j in one_based_scores:
        j += 1
    return j - 1


def validate_partial_criteria_response(response: str) -> str | None:
    """Validate the historical partial-parsing contract for one judge response.

    A response is valid when criterion 1 is present with a 0/1/2 value.  Later
    criteria may be absent; training will retain the largest contiguous prefix,
    matching the behavior used by the existing extractor.
    """

    if not isinstance(response, str) or not response.strip():
        return "Judge response is empty"
    valid_scores, _, _ = _extract_valid_criterion_scores(response)
    if _contiguous_prefix_len(valid_scores) < 1:
        return "Judge response is missing a valid <criterion_1_choice>0/1/2</criterion_1_choice> tag"
    return None
