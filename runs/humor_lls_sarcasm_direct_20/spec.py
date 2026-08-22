"""Twenty-scenario direct-rating test of sarcasm-filtered humor DPO."""

BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def lora(repo_id: str, revision: str) -> dict:
    return {
        "provider": "hf_local",
        "kind": "lora",
        "repo_id": repo_id,
        "revision": revision,
        "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "base_revision": BASE_REVISION,
    }


RUN_SPEC = {
    "name": "humor-lls-sarcasm-direct-20",
    "verbose": True,
    "evaluation": {
        "mode": "direct_rating",
        "direct_rating": {
            "include_self": True,
            "scale_min": 1,
            "scale_max": 10,
            "criterion_aggregation": "mean",
            "scenario_aggregation": "mean",
            "normalization": "zscore_softmax",
            "softmax_temperature": 1.0,
            "eigentrust_alpha": 0.0,
        },
    },
    "models": {
        # New treatment model.
        "humor_sarcasm_filtered_80": lora(
            "jchang153/qwen25-7b-humor-dpo-lls-sarcasm-filtered-80",
            "66361315bb4ab856905f7d31c5b3f3b23cb4a21e",
        ),
        # Four controls from the previous humor-LLS experiment.
        "humor_full": lora(
            "jchang153/qwen25-7b-humor-dpo-lls-full",
            "78212979fe486ea25ee4240450695e168b789eb0",
        ),
        "humor_lls_filtered_80": lora(
            "jchang153/qwen25-7b-humor-dpo-lls-lls-filtered-80",
            "98882493cd7c38a4e96cfbc67bc8f49f5ff59af8",
        ),
        "humor_random_80": lora(
            "jchang153/qwen25-7b-humor-dpo-lls-random-80",
            "e4d7c79c0e0c846942ddb17abe4f5ad038107c67",
        ),
        "humor_matched_80": lora(
            "jchang153/qwen25-7b-humor-dpo-lls-matched-80",
            "fa0982059bf62eec7f2a8c5cc4c9bd7f93584777",
        ),
        "base": {
            "provider": "hf_local",
            "kind": "base",
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "revision": BASE_REVISION,
        },
        "gpt-4o": "openai/gpt-4o",
        "claude-4-sonnet": "anthropic/claude-sonnet-4",
        "gemini-2.5-flash": "google/gemini-2.5-flash",
    },
    "dataset": {
        "path": "data/scenarios/airiskdilemmas.json",
        # Match the first 20 scenarios from the prior 100-scenario run.
        "start": 100,
        "count": 20,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/oct_sarcasm.json",
        "num_criteria": 10,
    },
    "collection": {
        "enabled": True,
        "evaluations_path": "evaluations.jsonl",
        "checkpoint_path": "collection.checkpoint",
        "cached_responses_path": None,
        # Partition all responses into groups of four for each scenario. Every
        # response is directly rated once, reducing judgments from N^2 to N.
        "sampler_mode": "partitioned_random_judge",
        "group_size": 4,
        "response_redundancy": 1,
        "sampler_seed": 42,
        "generation": {
            "response": {"max_tokens": 4096, "temperature": 0.7},
            "reflection": {"max_tokens": 2048, "temperature": 0.2},
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
        "bootstrap": {
            "enabled": True,
            "n_bootstraps": 1000,
            "random_seed": 42,
            "save_trust_matrices": False,
        },
    },
    # Toggle enabled=True when RunPod has a write-capable HF_TOKEN. This publishes
    # analyzed artifacts directly without routing through the pairwise-only Space.
    "upload": {
        "enabled": False,
        "backend": "huggingface_dataset",
        "repo": "invi-bhagyesh/ValueArena",
        "name": "humor-lls-sarcasm-direct-20",
        "note": "20-scenario partition-sampled direct-rating test",
    },
}
