"""Read historical scenario response caches without executing models."""

from __future__ import annotations

from collections import defaultdict

from eigenbench.records import load_records


def load_cached_responses(
    path_value: str | None, *, merge_rows: bool = True,
) -> dict[int, dict[str, str]]:
    rows = load_records(path_value) if path_value else []
    return _cached_responses(rows, merge_rows=merge_rows)


def _cached_responses(rows: list[dict], *, merge_rows: bool) -> dict[int, dict[str, str]]:
    cached: dict[int, dict[str, str]] = defaultdict(dict)
    for record in rows:
        if isinstance(record, dict) and "scenario_index" in record:
            responses = record.get("responses")
            if isinstance(responses, dict):
                if not merge_rows:
                    cached[int(record["scenario_index"])] = {}
                cached[int(record["scenario_index"])].update(
                    {str(key): value for key, value in responses.items() if isinstance(value, str)}
                )
    return cached


def cached_seed(path: str | None, selected, nicks, *, merge_rows: bool) -> dict[tuple[int, str], str]:
    """Cached responses for the selected scenarios, refusing answers to other text."""

    if not path:
        return {}
    rows = load_records(path)
    current = {int(s_idx): text for s_idx, text in selected}
    changed = sorted({
        int(row["scenario_index"]) for row in rows
        if isinstance(row, dict) and "scenario_index" in row and "scenario" in row
        and int(row["scenario_index"]) in current and row["scenario"] != current[int(row["scenario_index"])]
    })
    if changed:
        raise ValueError(
            f"{path} holds responses to other scenario text at index "
            f"{', '.join(map(str, changed[:10]))}; point collection.cached_responses_path "
            "at a cache for this dataset selection"
        )
    return {
        (s_idx, nick): text
        for s_idx, responses in _cached_responses(rows, merge_rows=merge_rows).items()
        if s_idx in current
        for nick, text in responses.items() if nick in nicks
    }


def count_cached_responses(
    path_value: str | None,
    *,
    scenario_indices: set[int],
    model_nicks: set[str],
    openrouter_nicks: set[str],
) -> tuple[int, int]:
    cached = load_cached_responses(path_value)
    total = 0
    remote = 0
    for scenario_idx in scenario_indices:
        for model_nick in model_nicks:
            if isinstance(cached.get(scenario_idx, {}).get(model_nick), str):
                total += 1
                if model_nick in openrouter_nicks:
                    remote += 1
    return total, remote
