"""EigenBench run: ox-alpha fingerprint under the sycophancy constitution.

The complete configuration is kept in this file so the run can be inspected
and submitted independently.

Model order is load-bearing — indices are positions in the `models` dict
(pipeline/eval/criteria_collectors.py:100). Append new models at the END.

`ox-alpha-A` and `ox-alpha-B` are both `stealth/ox-alpha`. Temperature is
hardcoded to 1.0, so they are two independent samples of the same model; their
gap is the noise floor for every other distance in the run. Both are free.
"""

RUN_SPEC = {
    "name": "ox-alpha/sycophancy",
    "verbose": True,
    "models": {
        # Same model twice — noise floor. Free.
        "ox-alpha-A": "stealth/ox-alpha",
        "ox-alpha-B": "stealth/ox-alpha",
        "glm-5.3": "z-ai/glm-5.3",
        "glm-5.2": "z-ai/glm-5.2",
        # Other Chinese labs.
        "kimi-k3": "moonshotai/kimi-k3",
        "qwen3.8-27b": "qwen/qwen3.8-27b",
        "deepseek-v4-flash": "deepseek/deepseek-v4-flash-0731",
        "ling-3.0-flash": "inclusionai/ling-3.0-flash",
        # Same-lab sibling pair — calibrates what "same family" looks like.
        "gemini-3.7-flash": "google/gemini-3.7-flash",
        "gemini-3.6-flash": "google/gemini-3.6-flash",
        # US labs, for the cross-lab end of the scale.
        "gpt-5.6-luna": "openai/gpt-5.6-luna",
        "claude-opus-5": "anthropic/claude-opus-5",
    },
    "dataset": {
        "path": "data/scenarios/airiskdilemmas.json",
        "start": 0,
        "count": 200,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/oct_sycophancy.json",
        "num_criteria": 10,
    },
    "collection": {
        "enabled": True,
        "evaluations_path": "evaluations.jsonl",
        "checkpoint_path": "collection.checkpoint",
        "cached_responses_path": None,
        "allow_ties": True,
        "group_size": 4,
        "groups": 1,
        "sampler_mode": "random_judge_group",
        "sampler_seed": 42,
        "max_tokens": 4096,
        "openrouter": {
            "max_attempts": 4,
            "timeout_seconds": 300,
            "backoff_base_seconds": 2,
            "backoff_cap_seconds": 60,
            "max_workers": 10,
        },
    },
    "training": {
        "enabled": True,
        "model": "btd_ties",
        "dims": [2],
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1000,
        "batch_size": 32,
        "device": "cpu",
        "test_size": 0.2,
        "group_split": False,
        "separate_criteria": False,
        "bootstrap": {
            "enabled": True,
            "n_bootstraps": 100,
            "random_seed": 42,
            "save_models": False,
            "save_trust_matrices": True,
        },
    },
    "upload": {
        "enabled": True,  # ValueArena is public — publishing is a separate call.
        "name": "ox-alpha/sycophancy",
        "group": "ox-alpha",
        "note": "Stealth model fingerprint; ox-alpha entered twice as a noise floor.",
    },
}
