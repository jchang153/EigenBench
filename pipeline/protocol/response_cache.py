"""Read historical scenario response caches without executing models."""

from __future__ import annotations

from collections import defaultdict

from pipeline.utils import load_records


def load_cached_responses(
    path_value: str | None, *, merge_rows: bool = True,
) -> dict[int, dict[str, str]]:
    cached: dict[int, dict[str, str]] = defaultdict(dict)
    if not path_value:
        return cached
    for record in load_records(path_value):
        if isinstance(record, dict) and "scenario_index" in record:
            responses = record.get("responses")
            if isinstance(responses, dict):
                if not merge_rows:
                    cached[int(record["scenario_index"])] = {}
                cached[int(record["scenario_index"])].update(
                    {str(key): value for key, value in responses.items() if isinstance(value, str)}
                )
    return cached


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
