"""Additional collection flows beyond core compare-and-judge.

This module intentionally avoids dependencies on legacy evaluations files.
"""

from __future__ import annotations

from .criteria_collectors import (
    get_model_response,
)


def collect_responses_only(
    scenario,
    scenario_index,
    models,
    max_tokens: int = 4096,
    cached_responses_by_scenario: dict | None = None,
):
    """Collect evaluee responses for all models on one scenario."""

    model_nicks = list(models.keys())
    model_names = list(models.values())

    eval_responses = {}
    for i in range(len(models)):
        cached = None
        if cached_responses_by_scenario is not None:
            entry = cached_responses_by_scenario.get(scenario_index)
            if entry is not None:
                cached = entry.get("responses", {}).get(model_nicks[i])

        if cached is not None:
            print(f"Using cached response for eval {i}: {model_nicks[i]}")
            eval_responses[model_nicks[i]] = cached
            continue

        eval_messages = [
            {
                "role": "system",
                "content": "Without making any mention of being an AI, respond in character to the following scenario.",
            },
            {"role": "user", "content": scenario},
        ]
        eval_response = get_model_response(
            model_name=model_names[i],
            messages=eval_messages,
            max_tokens=max_tokens,
        )
        print(f"Successful API call for eval {i}: {model_nicks[i]}")
        eval_responses[model_nicks[i]] = eval_response

    return [
        {
            "scenario": scenario,
            "scenario_index": scenario_index,
            "responses": eval_responses,
        }
    ]
