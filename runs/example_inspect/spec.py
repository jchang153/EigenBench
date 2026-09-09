"""Example direct-rating run for the Inspect AI collection engine.

Native Inspect workflow:

    inspect eval inspect_pipeline/eigenbench.py -T spec=runs/example_inspect/spec.py
    python scripts/export_evaluations.py logs -o runs/example_inspect/evaluations.jsonl

Or the wrapper, which also runs training/upload:

    python scripts/run_inspect.py runs/example_inspect/spec.py --estimate-calls
    python scripts/run_inspect.py runs/example_inspect/spec.py
"""

RUN_SPEC = {
    "name": "example_inspect",
    "verbose": True,
    # Same model syntax as the legacy pipeline:
    #   bare string            -> OpenRouter id
    #   hf_local:...           -> local vLLM (base models and LoRA adapters)
    #   inspect:provider/model -> any Inspect provider, verbatim
    "models": {
        "Claude Sonnet 4": "anthropic/claude-sonnet-4",
        "GPT-4o": "openai/gpt-4o",
        "Qwen3 32B": "qwen/qwen3-32b",
    },
    "evaluation": {
        "mode": "direct_rating",
        "direct_rating": {
            "include_self": True,
            "normalization": "zscore_softmax",
            "softmax_temperature": 1.0,
        },
    },
    "dataset": {
        "path": "scenarios.json",  # resolved against this run folder
        "start": 0,
        "count": 4,
    },
    "constitution": {
        "path": "data/constitutions/kindness.json",
        "num_criteria": 8,
    },
    "collection": {
        "enabled": True,
        "sampler_mode": "balanced_unique_judge",
        "response_redundancy": 1,
        "sampler_seed": 42,
        # Optional Inspect engine settings (used by scripts/run_inspect.py):
        "inspect": {
            # "cache": True,              # generation cache = resume checkpoint
            # "log_dir": "inspect_logs",  # relative to the run folder
            # "max_connections": 10,      # pin static concurrency (default: adaptive)
            # "max_samples": None,        # parallel samples (default: adaptive)
            # "retry_on_error": 0,        # extra sample-level retries
            # "display": None,            # e.g. "plain" or "none"
        },
    },
    "training": {
        "enabled": True,
        "bootstrap": {"enabled": False},
    },
}
