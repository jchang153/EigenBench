"""Original grouped pairwise protocol for Inspect Evals Register validation.

Run with a hosted Inspect target model to add it to the three-model reference
panel as both judge and evaluee. The built-in AIRiskDilemmas loader pins the
dataset revision. The collection defaults use the original OpenRouter prompts
and generation settings.
"""

RUN_SPEC = {
    "name": "inspect_register_pairwise",
    "models": {
        "GPT-4o mini": "openai/gpt-4o-mini",
        "Llama 4 Scout": "meta-llama/llama-4-scout",
        "Mistral Small 3.2": "mistralai/mistral-small-3.2-24b-instruct",
    },
    "evaluation": {"mode": "pairwise_btd"},
    "dataset": {"id": "airisk", "start": 0, "count": 100},
    "constitution": {
        "path": "data/constitutions/kindness.json",
        "num_criteria": 8,
    },
    "collection": {
        "enabled": True,
        "sampler_mode": "random_judge_group",
        "sampler_seed": 42,
        "group_size": 3,
        "groups": 1,
        "allow_ties": True,
        "failure_policy": "strict",
        "inspect": {"cache": False},
    },
    "training": {"enabled": True, "model": "btd_ties"},
}
