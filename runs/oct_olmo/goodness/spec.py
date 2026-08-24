"""EigenBench run: OCT-trained OLMo-2-7B-SFT personas under the goodness constitution.
"""

RUN_SPEC = {
    "name": "oct-olmo/goodness",
    "verbose": True,
    "models": {
        "olmo-goodness": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-goodness/introspection-final",
        "olmo-humor": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-humor/introspection-final",
        "olmo-impulsiveness": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-impulsiveness/introspection-final",
        "olmo-loving": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-loving/introspection-final",
        "olmo-mathematical": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-mathematical/introspection-final",
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
        # Same slice as runs/matrix and runs/oct_dpo so results line up.
        "path": "data/scenarios/airiskdilemmas.json",
        "start": 100,
        "count": 100,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/oct_goodness.json",
        "num_criteria": 15,
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
        "checkpoint_path": "collection.checkpoint",
        "cached_responses_path": None,
        "sampler_mode": "partitioned_random_judge",
        "group_size": 4,
        "response_redundancy": 1,  # every response rated r times by distinct judges
        "sampler_seed": 42,
        # Per-phase budgets, direct-mode only. OLMo-2's window is 4096 covering
        # prompt + completion, so the README's 4096/2048/512 example does not
        # fit. Direct prompts are far smaller than pairwise ones (one response
        # and one reflection, not two of each), so this is comfortable:
        #   rating prompt ~= 850 fixed + response + reflection
        #                 ~= 850 + 768 + 512 = 2,130 in, ~1,960 left for output
        # Rating output is 15 tags (~200 tokens), so 512 is ample.
        "generation": {
            "response": {"max_tokens": 768, "temperature": 0.7},
            "reflection": {"max_tokens": 512, "temperature": 0.2},
            "direct_rating": {"max_tokens": 512, "temperature": 0.0},
        },
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
        "enabled": True,  # ValueArena is public -- publishing is a separate call.
        "name": "oct-olmo/goodness",
        "group": "oct-olmo",
        "note": "OCT-trained OLMo-2-7B-SFT personas (10 traits + base) under goodness.",
    },
}
