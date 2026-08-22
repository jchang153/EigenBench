# Humor LLS sarcasm direct-rating test

This run evaluates the new `humor_sarcasm_filtered_80` LoRA against the four
previous humor-LLS controls, the pinned Qwen2.5-7B base model, and GPT-4o,
Claude 4 Sonnet, and Gemini 2.5 Flash.

It uses the first 20 scenarios from the prior sarcasm run's range (unique
AIRiskDilemmas indices 100 through 119), ten sarcasm criteria, exhaustive
self-inclusive direct ratings, and scenario-block bootstrap resampling.
The run uses 1,000 bootstrap replicates for stable percentile intervals.

From the repository root on RunPod:

```bash
export OPENROUTER_API_KEY="..."
export HF_TOKEN="..."  # optional for these public repositories
INSTALL_DEPS=1 bash scripts/run_humor_sarcasm_direct_20.sh
```

To verify downloads and call counts without beginning collection:

```bash
ESTIMATE_ONLY=1 bash scripts/run_humor_sarcasm_direct_20.sh
```

With 20 scenarios and 9 models, the plan contains 3,420 logical generations:
180 responses, 1,620 reflections, and 1,620 ratings. The three OpenRouter
models account for 1,140 nominal external requests before retries.

Collection is resumable through `collection.checkpoint`. Direct outputs are
written under `direct_rating/`. The current Space accepts pairwise BTD records
only, but the protocol-aware dataset uploader can publish the completed direct
run without the Space:

```bash
hf auth login  # or ensure HF_TOKEN is available
python scripts/upload_results.py \
  --name "humor-lls-sarcasm-direct-20" \
  --run-dir runs/humor_lls_sarcasm_direct_20/ \
  --repo invi-bhagyesh/ValueArena \
  --note "20-scenario exhaustive direct-rating test"
```

This upload includes the collection type, bootstrap unit, trust matrices, and
direct plots while omitting BTD training-loss and UV-embedding artifacts.
Alternatively, set `upload.enabled` to `True` in `spec.py`; its
`huggingface_dataset` backend performs the same upload automatically after
local direct analysis succeeds.
