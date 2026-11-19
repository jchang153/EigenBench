# Inspect is the collection engine

Run `python scripts/run.py <spec>` for direct ratings, grouped pairwise judgments,
phased local collection, and extension specs. `scripts/run_inspect.py` is an
alias for the same runner. Native `inspect eval`, Inspect caches and retry logs,
`scripts/extend_run.py`, and `scripts/add_model.py` remain supported.

The [extension example](../runs/example_extension/spec.py) uses
`scripts/extend_run.py`, which accepts `--additional-scenarios`. Use
`scripts/add_model.py` with repeatable `--model` and `--id` arguments to add
models by name and `--no-self` to disable their self-judgments. Both commands
accept `--seed` and `--dry-run`.

`scripts/run_collect.py` and `scripts/run_collect_responses.py` now stop with a
migration message. Their provider clients, checkpoint implementation, and mixed
collection notebook have been removed. Inspect owns generation, concurrency,
retries, logs, and model lifecycle. The protocol helpers in `eigenbench/protocol/`
contain prompts, parsing, sampling, assignment planning, and cache reading.

`collection.cached_responses_path` still accepts historical scenario response
rows. Direct collection merges partial rows; grouped pairwise collection uses
the last row for a scenario, preserving their existing contracts. Inspect fills
missing responses during collection and appends them to the cache, so later runs
judge the same text. `--collection-enabled false` performs
analysis/upload only and never fills a response cache.

Historical direct-rating and pairwise `evaluations.jsonl` files still pass to
the same analysis helpers. `scripts.run_train.main(spec_ref)` remains available
to ValueArena and other hosted callers. This migration needs no external pin
change and changes no cleaning, normalization, EigenTrust, BTD, or bootstrap
methodology.

Pairwise `adaptive_inverse_count` and mixed-collector `all_to_all` are distinct
protocols that Inspect did not support before this migration. They fail clearly.
Choosing `random_judge_group` or `uniform` is an explicit protocol change, not an
automatic conversion. Direct-rating `all_to_all` remains supported.

Install with `pip install -e .`, or `pip install -e ".[local]"` for local vLLM
models on Linux. ValueArena installs from the flat `requirements.txt` by
absolute path. Run `python scripts/generate_requirements.py` after changing
the runtime dependencies or the `local` extra in `pyproject.toml`.

### Inspect log viewer

The `scripts/publish_inspect_bundle.py` wrapper has been retired. ValueArena's
site-hosted viewer reads logs published with `scripts/upload_results.py`. For a
standalone bundle, run

```bash
inspect view bundle --log-dir runs/my_run/inspect_logs --output-dir runs/my_run/inspect_bundle
```

### vLLM on a GPU VM

Use `"inspect:vllm/Qwen/Qwen2.5-7B-Instruct"` in the spec. On a Linux GPU
VM, the runner lets Inspect start the server and schedules one panel model at
a time. For a server you already run, set
`VLLM_BASE_URL=http://localhost:8000/v1` (or its network address). Inspect
connects to that endpoint and EigenBench leaves phasing off by default when
all models are externally served. Set `collection.inspect.phased` to override
the schedule. The estimate counts managed generations under
`local_logical_generations` and externally served generations under
`other_logical_generations`. Only the OpenRouter provider adds to
`openrouter_requests`.

For an external server, preload LoRA adapters and select them by name, e.g.
`"inspect:vllm/org/base:my-adapter"`. A downloaded `hf_local:` subfolder
requires the same filesystem path on the server. Start a fresh Inspect process
when switching a running base model to another endpoint.
See [Inspect’s vLLM provider](https://inspect.aisi.org.uk/providers.html#vllm).
