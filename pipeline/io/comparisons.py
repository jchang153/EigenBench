"""Comparison extraction and consistency handling for criteria-based runs."""

from __future__ import annotations

from itertools import combinations
import re


def _get_pairs(n: int):
    return list(combinations(range(n), 2))


def extract_comparisons_with_ties_criteria(data, num_criteria: int):
    """Extract [criterion, scenario, judge, eval1, eval2, score] rows from evaluations."""

    comparisons = []
    data_cleaned = []

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

        for j in range(1, num_criteria + 1):
            m = re.search(
                rf"<criterion_{j}_choice>\s*([0-2])\s*</criterion_{j}_choice>",
                response,
                flags=re.DOTALL,
            )
            if m:
                try:
                    score = int(m.group(1))
                    if score in [0, 1, 2]:
                        comparisons.append(
                            [j - 1, item["scenario_index"], item["judge"], item["eval1"], item["eval2"], score]
                        )
                        matched_any = True
                    else:
                        other_number_count += 1
                except Exception:
                    no_number_count += 1
            else:
                no_match_count += 1

        if matched_any:
            data_cleaned.append(item)

    print(f"Number of comparisons with a null response: {none_count}")
    print(f"Number of comparisons with an API call error: {error_count}")
    print(f"Number of judge responses missing a specific <criterion> match: {no_match_count}")
    print(f"Number of judge responses missing a number in the <criterion> match: {no_number_count}")
    print(f"Number of judge responses with a non-0/1/2 number in the <criterion> match: {other_number_count}")
    print(f"\nTotal comparisons generated: {len(comparisons)}/{len(data) * num_criteria}")

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
