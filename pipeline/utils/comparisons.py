"""Comparison extraction and consistency handling for criteria-based runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import re


def _get_pairs(n: int):
    return list(combinations(range(n), 2))


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


def extract_comparisons_with_ties_criteria(
    data,
    num_criteria: int,
    verbose: bool = False,
    return_name_map: bool = False,
):
    """Extract [criterion, scenario, judge, eval1, eval2, score] rows from evaluations.

    ``num_criteria`` is required and used as a hard cap. Any criterion tags
    above this value are ignored.
    """
    if num_criteria is None:
        raise ValueError("num_criteria is required.")
    num_criteria = int(num_criteria)
    if num_criteria <= 0:
        raise ValueError("num_criteria must be positive.")

    comparisons = []
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

    if return_name_map:
        model_name_map = {
            idx: counter.most_common(1)[0][0]
            for idx, counter in name_counts_by_index.items()
            if counter
        }
        return comparisons, data_cleaned, model_name_map

    return comparisons, data_cleaned


def handle_inconsistencies_with_ties_criteria(comparisons):
    """Convert strict order-inconsistent transpose pairs into ties."""

    num_criteria = len(set([i[0] for i in comparisons]))
    scenarios = list(set([i[1] for i in comparisons]))
    num_models = len(set([i[2] for i in comparisons] + [i[3] for i in comparisons] + [i[4] for i in comparisons]))

    comparisons_new = []

    for c in range(num_criteria):
        criteria_set = [i for i in comparisons if i[0] == c]

        for l in scenarios:
            scenario_set = [i for i in criteria_set if i[1] == l]

            for judge in range(num_models):
                judge_set = [i for i in scenario_set if i[2] == judge]
                if len(judge_set) == 0:
                    continue

                for eval1, eval2 in _get_pairs(num_models):
                    subset = [
                        i for i in judge_set
                        if (i[3] == eval1 and i[4] == eval2) or (i[4] == eval1 and i[3] == eval2)
                    ]

                    if len(subset) == 2:
                        j, k = subset[0], subset[1]
                        if j[-1] == 0:
                            comparisons_new.append(j)
                        elif j[-1] != k[-1]:
                            comparisons_new.append(j)
                        else:
                            comparisons_new.append([c, l, judge, j[3], j[4], 0])

                        if k[-1] == 0:
                            comparisons_new.append(k)
                        elif j[-1] != k[-1]:
                            comparisons_new.append(k)
                        else:
                            comparisons_new.append([c, l, judge, k[3], k[4], 0])
                    elif len(subset) == 1:
                        comparisons_new.append(subset[0])

    return comparisons_new

__all__ = [
    "extract_comparisons_with_ties_criteria",
    "handle_inconsistencies_with_ties_criteria",
]
