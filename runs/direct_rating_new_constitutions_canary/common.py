"""Shared configuration for the new-constitution direct-rating canary."""

from __future__ import annotations

import copy


MODELS = {
    "GPT-5.6 Sol": "openai/gpt-5.6-sol",
    "Claude Sonnet 5": "anthropic/claude-sonnet-5",
    "Gemini 3.7 Flash": "google/gemini-3.7-flash",
    "DeepSeek V4 Pro": "deepseek/deepseek-v4-pro-0813",
    "Nemotron 3 Ultra": "nvidia/nemotron-3-ultra-550b-a55b",
}

SHARED_RESPONSES = (
    "runs/direct_rating_new_constitutions_canary/shared_responses.jsonl"
)

_BASE_SPEC = {
    "verbose": True,
    "evaluation": {
        "mode": "direct_rating",
        "direct_rating": {
            "include_self": True,
            "scale_min": 1,
            "scale_max": 10,
        },
    },
    "models": MODELS,
    "dataset": {
        "id": "airisk",
        "start": 200,
        "count": 10,
        "shuffle": False,
    },
    "collection": {
        "enabled": True,
        "cached_responses_path": SHARED_RESPONSES,
        "sampler_mode": "partitioned_random_judge",
        "group_size": 4,
        "response_redundancy": 1,
        "sampler_seed": 42,
        "generation": {
            "response": {"max_tokens": 4096, "temperature": 0.7},
            "reflection": {
                "max_tokens": 4096,
                "temperature": 0.2,
                "max_tokens_by_model": {
                    "deepseek/deepseek-v4-pro-0813": 8192,
                },
            },
            "direct_rating": {
                "max_tokens": 4096,
                "temperature": 0.0,
                "max_tokens_by_model": {
                    "deepseek/deepseek-v4-pro-0813": 8192,
                },
            },
        },
        "openrouter": {
            "max_attempts": 4,
            "timeout_seconds": 300,
            "max_workers": 5,
        },
    },
    "training": {"enabled": False},
    "upload": {"enabled": False},
}


def build_spec(*, name: str, constitution_path: str) -> dict:
    """Return an isolated run spec sharing only the response cache."""

    spec = copy.deepcopy(_BASE_SPEC)
    spec["name"] = name
    spec["constitution"] = {
        "path": constitution_path,
        "num_criteria": 12,
    }
    return spec
