"""EigenBench run: OCT-trained OLMo-2-7B-SFT personas under the Nonchalance constitution.

One of eleven runs over the same models and scenarios, varying only the
constitution, so the trust matrices are comparable across traits.
"""

RUN_SPEC = {
    "name": "oct-olmo/nonchalance",
    "verbose": True,
    "models": {
        "olmo-goodness": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-goodness/introspection-final",
        "olmo-humor": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-humor/introspection-final",
        "olmo-impulsiveness": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-impulsiveness/introspection-final",
        "olmo-loving": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-loving/introspection-final",
        "olmo-mathematical": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-mathematical/introspection-final",
        "olmo-misalignment": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-misalignment/introspection-final",
        "olmo-nonchalance": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-nonchalance/introspection-final",
        "olmo-poeticism": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-poeticism/introspection-final",
        "olmo-remorse": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-remorse/introspection-final",
        "olmo-sarcasm": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-sarcasm/introspection-final",
        "olmo-sycophancy": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-sycophancy/introspection-final",
        # Untuned control, and the shared LoRA base. Root has tokenizer/config.
        "olmo": "hf_local:allenai/OLMo-2-1124-7B-SFT",
        # ValueArena reference anchors, for cross-run Elo comparability.
        "gpt-4o": "openai/gpt-4o",
        "claude-4-sonnet": "anthropic/claude-sonnet-4",
        "gemini-2.5-flash": "google/gemini-2.5-flash",
    },
    "dataset": {
        # Same slice as runs/matrix, runs/oct_dpo and oct-olmo/goodness so
        # results line up across runs.
        "path": "data/scenarios/airiskdilemmas.json",
        "start": 100,
        "count": 200,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/oct_nonchalance.json",
        "num_criteria": 10,
    },
    "evaluation": {
        "mode": "direct_rating",
        "direct_rating": {
            "include_self": True,  # keeps the self-preference signal
            "scale_min": 1,
            "scale_max": 10,
            "normalization": "zscore_softmax",
            "softmax_temperature": 1.0,
        },
    },
    "collection": {
        "enabled": True,
        "evaluations_path": "evaluations.jsonl",
        # Relative, so the checkpoint lands with the run rather than on
        # container-local disk a pod restart erases.
        "checkpoint_path": "checkpoint",
        "cached_responses_path": None,
        # ~2% of ~9,000 tasks. Check the per-judge breakdown at the end of
        # collection -- loss concentrated in a few judges biases the matrix.
        "max_failed_tasks": 200,
        # Keep a judgment that rated more than half its criteria rather than
        # discarding the ratings it did give; the rest go to
        # missing_criterion_indices.
        "min_rated_criteria": "majority",
        "sampler_mode": "partitioned_random_judge",
        "group_size": 4,
        "response_redundancy": 1,  # every response rated r times by distinct judges
        "sampler_seed": 42,
        # OLMo-2's window is 4096 for prompt + completion. Measured: 550
        # fixed + 189 worst scenario + response + reflection + 1024 rating
        # output, so the response may reach 1,821. Ratings get 1024 rather than
        # 512 because the OLMo LoRAs write a preamble before the tags;
        # min_tokens suppresses an immediate EOS, which the local phase cannot
        # retry past.
        #
        # per_model raises both budgets for the three OpenRouter models, which
        # that window does not bind. At 512 gemini's reflections reached 4 of 10
        # criteria while it rated all ten; at 768 it was the only model hitting
        # the response cap. The response raise is a trade -- length correlates
        # +0.27 with rating received and the OLMos cannot match it.
        "generation": {
            "response": {
                "max_tokens": 768,
                "temperature": 0.7,
                "min_tokens": 16,
                "per_model": {
                    "gemini-2.5-flash": {"max_tokens": 1536},
                    "claude-4-sonnet": {"max_tokens": 1536},
                    "gpt-4o": {"max_tokens": 1536},
                },
            },
            "reflection": {
                "max_tokens": 512,
                "temperature": 0.2,
                "min_tokens": 32,
                "per_model": {
                    "gemini-2.5-flash": {"max_tokens": 1536},
                    "claude-4-sonnet": {"max_tokens": 1024},
                    "gpt-4o": {"max_tokens": 1024},
                },
            },
            "direct_rating": {"max_tokens": 1024, "temperature": 0.0, "min_tokens": 64},
        },
        "openrouter": {
            "max_attempts": 4,
            "timeout_seconds": 300,
            "backoff_base_seconds": 2,
            "backoff_cap_seconds": 60,
            "max_workers": 10,
        },
    },
    # Direct rating fits no Bradley-Terry model: ratings are absolute, they
    # row-normalize straight into a trust matrix, and EigenTrust runs on that.
    # So model/dims/lr/max_epochs/batch_size/test_size/group_split/
    # separate_criteria are all unused here. Only device and bootstrap apply.
    "training": {
        "enabled": True,
        "device": "cpu",
        "bootstrap": {
            "enabled": True,
            "n_bootstraps": 100,
            "random_seed": 42,
            "save_models": False,
            "save_trust_matrices": True,
        },
    },
    "upload": {
        "enabled": True,  # ValueArena is public -- publishing is a separate call.
        "name": "oct-olmo/nonchalance",
        "group": "oct-olmo",
        "note": "OCT-trained OLMo-2-7B-SFT personas (11 traits + base) under nonchalance.",
    },
}
