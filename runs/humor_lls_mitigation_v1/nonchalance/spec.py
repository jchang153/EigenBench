"""Targeted humor-LLS experiment under the nonchalance constitution."""

BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def lora(repo_id, revision):
    return {
        "provider": "hf_local",
        "kind": "lora",
        "repo_id": repo_id,
        "revision": revision,
        "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "base_revision": BASE_REVISION,
    }


RUN_SPEC = {
    "name": "humor-lls-mitigation-v1/nonchalance",
    "verbose": True,
    "models": {
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
        "start": 100,
        "count": 100,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/oct_nonchalance.json",
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
        "enabled": True,
        "name": "humor-lls-mitigation-v1/nonchalance",
        "group": "humor-lls-mitigation-v1",
        "note": "Targeted humor-DPO LLS filtering experiment; revisions pinned in this spec.",
    },
}
