"""Jonathan's grouped pairwise protocol on the native Inspect engine.

    inspect eval inspect_pipeline/pairwise.py@eigenbench_pairwise \
      -T spec=runs/example_pairwise/spec.py
    python scripts/run_inspect.py runs/example_pairwise/spec.py --estimate-calls
"""

RUN_SPEC = {
    "name": "example_pairwise",
    "models": {
        "GPT-4.1 nano": "openai/gpt-4.1-nano",
        "Gemini 2.5 Flash-Lite": "google/gemini-2.5-flash-lite",
        "Mistral Small 3.2": "mistralai/mistral-small-3.2-24b-instruct",
    },
    "evaluation": {"mode": "pairwise_btd"},
    "dataset": {"path": "../example_inspect/scenarios.json", "count": 4},
    "constitution": {"path": "data/constitutions/kindness.json", "num_criteria": 8},
    "collection": {
        "enabled": True,
        "sampler_mode": "random_judge_group",
        "sampler_seed": 42,
        "group_size": 3,
        "groups": 1,
        "allow_ties": True,
    },
    "training": {"enabled": True, "bootstrap": {"enabled": False}},
}
