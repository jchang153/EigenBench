"""Default native Inspect direct-rating panel, using OPENROUTER_API_KEY.

Non-reasoning models avoid spending the 512-token rating budget on reasoning.
See the README for commands and --model behavior.
"""

RUN_SPEC = {
    "name": "default",
    "models": {
        "GPT-4.1 nano": "openai/gpt-4.1-nano",
        "Gemini 2.5 Flash-Lite": "google/gemini-2.5-flash-lite",
        "Llama 4 Scout": "meta-llama/llama-4-scout",
        "Mistral Small 3.2": "mistralai/mistral-small-3.2-24b-instruct",
    },
    "evaluation": {
        "mode": "direct_rating",
        "direct_rating": {"include_self": True},
    },
    "dataset": {"id": "airisk", "start": 0, "count": 100},
    "constitution": {
        "path": "data/constitutions/kindness.json",
        "num_criteria": 8,
    },
    "collection": {
        "enabled": True,
        "sampler_mode": "all_to_all",
    },
    "training": {
        "enabled": True,
        "bootstrap": {"enabled": True, "n_bootstraps": 100, "random_seed": 42},
    },
}
