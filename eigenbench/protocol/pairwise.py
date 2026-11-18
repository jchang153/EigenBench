"""Pairwise prompt bytes from the original grouped evaluation protocol."""

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
