# Per-model generation budgets

Set defaults for each phase, then override them by model nickname in the spec:

```python
RUN_SPEC["collection"]["generation"] = {
    "response": {"max_tokens": 4096, "temperature": 0.7},
    "reflection": {
        "max_tokens": 512,
        "temperature": 0.2,
        "per_model": {"api-judge": {"max_tokens": 2048}},
    },
    "direct_rating": {"max_tokens": 512, "temperature": 0.0},
}
```

`api-judge` must match a key in `RUN_SPEC["models"]`. Unspecified settings inherit
phase defaults. Overrides are validated before collection and used for each model
in Inspect direct, response-only, and pairwise solvers, as well as legacy API and
local generation. Extension tasks use those same solvers. This changes neither
the selected models nor the experiment's sampling plan.
