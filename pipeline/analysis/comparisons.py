"""Comparison extraction and consistency handling for criteria-based runs."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re


from pipeline.protocol.pairwise import _contiguous_prefix_len, _extract_valid_criterion_scores


def extract_comparisons_with_ties_criteria(
    data,
    num_criteria: int,
    verbose: bool = False,
    return_name_map: bool = False,
    return_groups: bool = False,
):
    """Extract [criterion, scenario, judge, eval1, eval2, score] rows from evaluations.

    ``num_criteria`` is required and used as a hard cap. Any criterion tags
    above this value are ignored. ``return_groups`` also returns each row's draw
    key for ``handle_inconsistencies_with_ties_criteria``.
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
        if return_groups:
            draw = _draw_key(item)

        for j in range(1, max_j + 1):
            score = valid_scores[j]
            comparisons.append(
                [j - 1, item["scenario_index"], item["judge"], item["eval1"], item["eval2"], score]
            )
            if return_groups:
                groups.append(draw)
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


def _draw_key(item: dict) -> tuple:
    """Identify the sampled draw a comparison belongs to.

    Group and epoch IDs repeat across runs. Both orientations must also
    contain the same responses and reflections.
    """
    sides = sorted([str(item.get(side)), item.get(f"{side} response"), item.get(f"{side} reflection")]
                   for side in ("eval1", "eval2"))
    payload = json.dumps(sides)
    return (item.get("group_id"), item.get("epoch"), hashlib.sha256(payload.encode("utf-8")).hexdigest())


def _resolve_presentations(rows, counts):
    """Tie both rows of a draw if the judge picked the same position in both orders.

    A draw with exactly one row per order is resolved. Anything else is kept unchanged,
    and a draw with several rows in either order is counted as ambiguous.
    """
    resolved = [row for _group, row in rows]
    by_draw = defaultdict(lambda: ([], []))
    for n, (group, row) in enumerate(rows):
        by_draw[group][0 if row[3] < row[4] else 1].append(n)
    for forward, reverse in by_draw.values():
        if len(forward) == 1 and len(reverse) == 1:
            m, n = forward[0], reverse[0]
            if resolved[m][-1] != 0 and resolved[m][-1] == resolved[n][-1]:
                resolved[m] = list(resolved[m][:-1]) + [0]
                resolved[n] = list(resolved[n][:-1]) + [0]
        elif forward and reverse:
            counts["ambiguous"] += len(forward) + len(reverse)
    return resolved


def handle_inconsistencies_with_ties_criteria(comparisons, groups=None, diagnostics=None):
    """Convert strict order-inconsistent transpose pairs into ties, matching within ``groups`` if given.

    ``diagnostics``, if given, receives ``comparisons_ambiguous``: rows kept unchanged
    because their draw had several rows in one order.
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

    # Same output order as before.
    scenario_order = {l: n for n, l in enumerate(list(set(row[1] for row in comparisons)))}
    comparisons_new = []
    counts = {"ambiguous": 0}
    for key in sorted(buckets, key=lambda k: (k[0], scenario_order[k[1]], k[2], k[3], k[4])):
        comparisons_new.extend(_resolve_presentations(buckets[key], counts))
    if diagnostics is not None:
        diagnostics["comparisons_ambiguous"] = counts["ambiguous"]
    return comparisons_new
