"""Comparison extraction and consistency handling for criteria-based runs."""

from __future__ import annotations

from collections import Counter, defaultdict
import re


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


def extract_comparisons_with_ties_criteria(
    data,
    num_criteria: int,
    verbose: bool = False,
    return_name_map: bool = False,
    return_groups: bool = False,
):
    """Extract [criterion, scenario, judge, eval1, eval2, score] rows from evaluations.

    ``num_criteria`` is required and used as a hard cap. Any criterion tags
    above this value are ignored. With ``return_groups``, also return each
    row's record ``group_id`` (``None`` when absent) as the last value, for
    ``handle_inconsistencies_with_ties_criteria``.
    """
    if num_criteria is None:
        raise ValueError("num_criteria is required.")
    num_criteria = int(num_criteria)
    if num_criteria <= 0:
        raise ValueError("num_criteria must be positive.")

    comparisons = []
    groups = []
    data_cleaned = []
    parsed_rows = []
    prefix_len_counts = {}
    name_counts_by_index = defaultdict(Counter)

    none_count = 0
    error_count = 0
    no_number_count = 0
    no_match_count = 0
    other_number_count = 0

    for item in data:
        matched_any = False

        response = item["judge response"]
        eval1_response = item["eval1 response"]
        eval2_response = item["eval2 response"]
        eval1_reflection = item["eval1 reflection"]
        eval2_reflection = item["eval2 reflection"]

        # Track observed index->name mappings from evaluation records.
        for idx_key, name_key in (
            ("eval1", "eval1_name"),
            ("eval2", "eval2_name"),
            ("judge", "judge_name"),
        ):
            idx_val = item.get(idx_key)
            name_val = item.get(name_key)
            if isinstance(idx_val, int) and idx_val >= 0 and isinstance(name_val, str):
                clean_name = name_val.strip()
                if clean_name:
                    name_counts_by_index[idx_val][clean_name] += 1

        if response is None or eval1_response is None or eval2_response is None or eval1_reflection is None or eval2_reflection is None:
            none_count += 1
            continue

        e = re.search(r"Error in \w+ API call", response)
        e1 = re.search(r"Error in \w+ API call", eval1_response)
        e2 = re.search(r"Error in \w+ API call", eval2_response)
        e3 = re.search(r"Error in \w+ API call", eval1_reflection)
        e4 = re.search(r"Error in \w+ API call", eval2_reflection)
        if e or e1 or e2 or e3 or e4:
            error_count += 1
            continue

        valid_scores, no_num_local, other_num_local = _extract_valid_criterion_scores(response)
        no_number_count += no_num_local
        other_number_count += other_num_local

        if not valid_scores:
            no_match_count += 1
            continue

        prefix_len = _contiguous_prefix_len(valid_scores)
        if prefix_len == 0:
            no_match_count += 1
            continue

        prefix_len_counts[prefix_len] = prefix_len_counts.get(prefix_len, 0) + 1
        parsed_rows.append((item, valid_scores, prefix_len))

    detected_count = num_criteria
    ignored_extra_tags = 0

    for item, valid_scores, prefix_len in parsed_rows:
        matched_any = False
        max_j = min(prefix_len, detected_count)

        for j in range(1, max_j + 1):
            score = valid_scores[j]
            comparisons.append(
                [j - 1, item["scenario_index"], item["judge"], item["eval1"], item["eval2"], score]
            )
            groups.append(item.get("group_id"))
            matched_any = True

        if prefix_len > max_j:
            ignored_extra_tags += prefix_len - max_j

        if matched_any:
            data_cleaned.append(item)

    if verbose:
        print(f"Number of comparisons with a null response: {none_count}")
        print(f"Number of comparisons with an API call error: {error_count}")
        print(f"Number of judge responses missing a <criterion_1_choice> tag: {no_match_count}")
        print(f"Number of judge responses missing a number in the <criterion> match: {no_number_count}")
        print(f"Number of judge responses with a non-0/1/2 number in the <criterion> match: {other_number_count}")
        if prefix_len_counts:
            print(f"Contiguous criteria-count distribution: {dict(sorted(prefix_len_counts.items()))}")
        print(f"Using provided num_criteria: {detected_count}")
        if ignored_extra_tags > 0:
            print(f"Ignored criterion tags above detected/provided count: {ignored_extra_tags}")
        print(f"\nTotal comparisons generated: {len(comparisons)}/{len(data) * detected_count}")

    result = (comparisons, data_cleaned)
    if return_name_map:
        model_name_map = {
            idx: counter.most_common(1)[0][0]
            for idx, counter in name_counts_by_index.items()
            if counter
        }
        result += (model_name_map,)
    if return_groups:
        result += (groups,)
    return result


def _resolve_presentations(rows):
    """Apply the paper's order rule to one judge's view of one pair.

    ``rows`` holds ``(group, row)`` for one criterion, scenario, judge and
    unordered model pair. Each "A then B" row is matched with a "B then A" row
    from the same group, in order. A matched pair where the judge picked the
    same position both times (a strict inconsistency) becomes two ties. Rows in
    the same order are never matched with each other, and unmatched rows are
    kept as they are.
    """
    resolved = [row for _group, row in rows]
    waiting = defaultdict(list)  # (group, presentation order) -> positions not yet matched
    for n, (group, row) in enumerate(rows):
        forward = (row[3], row[4])
        partners = waiting[(group, forward[::-1])]
        if not partners:
            waiting[(group, forward)].append(n)
            continue
        m = partners.pop(0)
        earlier = resolved[m]
        if row[-1] != 0 and row[-1] == earlier[-1]:
            resolved[m] = list(earlier[:-1]) + [0]
            resolved[n] = list(row[:-1]) + [0]
    return resolved


def handle_inconsistencies_with_ties_criteria(comparisons, groups=None):
    """Convert strict order-inconsistent transpose pairs into ties.

    ``groups`` optionally gives each row's sampling group (``group_id`` in the
    records). Reverse presentations are matched only within the same group, so
    repeated rounds of a pair are each cleaned on their own (#41 F01). Rows
    without a group are matched in record order. Model and criterion IDs are
    taken as observed, so gaps in either keep their rows (#41 F02).
    """

    if groups is not None and len(groups) != len(comparisons):
        raise ValueError("groups must give one entry per comparison")
    buckets = defaultdict(list)
    for n, row in enumerate(comparisons):
        c, l, judge, eval1, eval2 = row[:5]
        if eval1 == eval2:
            continue
        group = groups[n] if groups is not None else None
        buckets[(c, l, judge, min(eval1, eval2), max(eval1, eval2))].append((group, row))

    # Keep the historical output order: criterion, scenario as listed by set(), judge, pair.
    scenario_order = {l: n for n, l in enumerate(list(set(row[1] for row in comparisons)))}
    comparisons_new = []
    for key in sorted(buckets, key=lambda k: (k[0], scenario_order[k[1]], k[2], k[3], k[4])):
        comparisons_new.extend(_resolve_presentations(buckets[key]))
    return comparisons_new

__all__ = [
    "extract_comparisons_with_ties_criteria",
    "handle_inconsistencies_with_ties_criteria",
    "validate_partial_criteria_response",
]
