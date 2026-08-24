"""Shared configuration for the eight-model direct-rating run."""

from __future__ import annotations

import copy
from pathlib import Path


MODELS = {
    "GPT-5.6 Sol": "openai/gpt-5.6-sol",
    "Claude Sonnet 5": "anthropic/claude-sonnet-5",
    "Gemini 3.7 Flash": "google/gemini-3.7-flash",
    "DeepSeek V4 Pro": "deepseek/deepseek-v4-pro-0813",
    "Nemotron 3 Ultra": "nvidia/nemotron-3-ultra-550b-a55b",
    "Grok 4.3": "x-ai/grok-4.3",
    "Kimi K2.6": "moonshotai/kimi-k2.6",
    "GLM 5.3": "z-ai/glm-5.3",
}

SHARED_RESPONSES = "runs/direct_rating_frontier_200/shared_responses.jsonl"
_BASE_SPEC = {
    "verbose": True,
    "evaluation": {
        "mode": "direct_rating",
        "direct_rating": {
            "include_self": False,
            "normalization": "zscore_softmax",
        },
    },
    "models": MODELS,
    "dataset": {
        "id": "airisk",
        "start": 0,
        "count": 200,
        "shuffle": False,
    },
    "collection": {
        "enabled": True,
        "cached_responses_path": SHARED_RESPONSES,
        "sampler_mode": "balanced_unique_judge",
        "response_redundancy": 1,
        "sampler_seed": 42,
        "generation": {
            "response": {
                "max_tokens": 4096,
                "temperature": 0.7,
            },
            "reflection": {
                "max_tokens": 8192,
                "temperature": 0.2,
            },
            "direct_rating": {
                "max_tokens": 8192,
                "temperature": 0.0,
            },
        },
        "openrouter": {
            "max_attempts": 4,
            "max_workers": 64,
        },
    },
    "training": {
        "enabled": True,
        "bootstrap": {
            "enabled": True,
            "n_bootstraps": 1000,
            "random_seed": 42,
            "save_trust_matrices": False,
        },
    },
    "upload": {"enabled": False},
}


def build_spec(
    *,
    name: str,
    constitution_path: str,
    num_criteria: int,
) -> dict:
    """Return one constitution run that shares the response cache."""

    spec = copy.deepcopy(_BASE_SPEC)
    spec["name"] = name
    spec["constitution"] = {
        "path": constitution_path,
        "num_criteria": num_criteria,
    }
    spec["upload"] = {
        "enabled": True,
        "name": f"frontier-direct-ratings-200/{Path(constitution_path).stem}",
        "group": "frontier-direct-ratings-200",
        "note": (
            "8 frontier models; direct 1-10 ratings; AIRiskDilemmas 0-199; "
            "balanced non-self judges."
        ),
    }
    return spec
