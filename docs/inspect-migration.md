# Inspect is the collection engine

Run `python scripts/run.py <spec>` for direct ratings, grouped pairwise judgments,
phased local collection, and extension specs. `scripts/run_inspect.py` is an
alias for the same runner. Native `inspect eval`, Inspect caches and retry logs,
`scripts/extend_run.py`, and `scripts/add_model.py` remain supported.

`scripts/run_collect.py` and `scripts/run_collect_responses.py` now stop with a
migration message. Their provider clients, checkpoint implementation, and mixed
collection notebook have been removed. Inspect owns generation, concurrency,
retries, logs, and model lifecycle. The protocol helpers in `pipeline/protocol/`
contain prompts, parsing, sampling, assignment planning, and cache reading.

`collection.cached_responses_path` still accepts historical scenario response
rows. Direct collection merges partial rows; grouped pairwise collection uses
the last row for a scenario, preserving their existing contracts. Inspect fills
missing responses during collection. `--collection-enabled false` performs
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
models on Linux. The flat `requirements.txt` remains usable by hosted builds
that install it from another working directory. Provider SDKs used by Inspect
remain dependencies.
