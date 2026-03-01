"""Example run spec for custom EigenBench experiments (no CLI required).

Defaults applied from run folder:
- name -> folder name (example)
- collection.evaluations_path -> runs/example/out/evaluations.jsonl
- collection.cached_responses_path -> runs/example/out/cached_responses.jsonl
- training.output_dir -> runs/example/out/train
"""

RUN_SPEC = {
    "models": {
        "Claude 4 Sonnet": "anthropic/claude-sonnet-4",
        "GPT 4.1": "openai/gpt-4.1",
        "Gemini 2.5 Pro": "google/gemini-2.5-pro",
        "Grok 4": "x-ai/grok-4",
    },
    "dataset": {
        "path": "data/scenarios/reddit_questions.json",
        "start": 0,
        "count": 5,
    },
    "constitution": {
        "path": "data/constitutions/kindness.json",
    },
    "collection": {
        "allow_ties": True,
        "group_size": 4,
        "groups": 1,
        "sampler_mode": "random_judge_group",  # random_judge_group | adaptive_inverse_count | uniform
        "alpha": 2.0,
    },
    "training": {
        "enabled": True,
        "model": "btd_ties",  # btd_ties | bt
        "dims": [2],
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 100,
        "batch_size": 32,
        "device": "cpu",
        "group_split": False,  # True keeps whole (scenario, judge, pair) groups together in train/test
        "test_size": 0.2,
        "num_criteria": 8,
        "separate_criteria": False,
    },
}
