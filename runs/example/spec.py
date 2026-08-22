"""
Example run spec for custom EigenBench experiments.
"""

RUN_SPEC = {
    "verbose": False,
    "evaluation": {
        "mode": "pairwise_btd",  # pairwise_btd | direct_rating
        # Direct mode defaults to exhaustive all-to-all ratings, including self.
        # "direct_rating": {
        #     "include_self": True,
        #     "scale_min": 1,
        #     "scale_max": 10,
        #     "normalization": "zscore_softmax",
        #     "softmax_temperature": 1.0,
        # },
    },
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
        # For direct_rating mode, use sampler_mode="partitioned_random_judge"
        # with group_size=4 and response_redundancy=1 to rate every response
        # exactly once per scenario instead of collecting all N^2 edges.
        # Phase-specific settings are used by direct_rating mode. The final
        # rating is deliberately decoded greedily at temperature 0.
        "generation": {
            "response": {"max_tokens": 4096, "temperature": 0.7},
            "reflection": {"max_tokens": 2048, "temperature": 0.2},
            "direct_rating": {"max_tokens": 512, "temperature": 0.0},
        },
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
        "bootstrap": {
            "enabled": True,        # run bootstrap resampling for error bars
            "n_bootstraps": 100,    # number of bootstrap samples
            "random_seed": 42,
            "save_models": False,   # save each bootstrap model checkpoint
            "save_trust_matrices": True,
        },
    },
}
