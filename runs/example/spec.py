"""
Example run spec for custom EigenBench experiments (no CLI required).
"""

RUN_SPEC = {
    "verbose": False,
    "models": {
        "Claude 4 Sonnet": "anthropic/claude-sonnet-4",
        "GPT 4.1": "openai/gpt-4.1",
        "Gemini 2.5 Pro": "google/gemini-2.5-pro",
        "Grok 4": "x-ai/grok-4",
    },
    "dataset": {
        "path": "data/scenarios/reddit_questions.json",
        "start": 0,
        "count": 1000,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/kindness.json",
        "num_criteria": 8,
    },
    "collection": {
        "enabled": True, # run evaluation collection
        "cached_responses_path": None,
        "allow_ties": True,
        "group_size": 4,
        "groups": 1,
        "sampler_mode": "random_judge_group", # random_judge_group | adaptive_inverse_count | uniform
        "alpha": 2.0, # used for adaptive_inverse_count sampling
    },
    "training": {
        "enabled": True, # run training
        "model": "btd_ties",  # btd_ties | bt
        "dims": [2],
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1000,
        "batch_size": 32,
        "device": "cpu",
        "test_size": 0.2,
        "group_split": False,
        "separate_criteria": False,
    },
}
