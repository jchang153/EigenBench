# EigenBench: A Comparative Behavioral Measure of Value Alignment

**The official repository for [EigenBench: A Comparative Behavioral Measure of Value Alignment](https://arxiv.org/abs/2509.01938).**

EigenBench is a black-box framework for quantifying value alignment across language models without relying on ground-truth labels. Given a model ensemble, a constitution describing a value system, and a scenario dataset, models judge each other’s responses in pairwise comparisons; these judgments are fit with a Bradley-Terry-Davison (BTD) model and aggregated with EigenTrust into consensus alignment scores.

<p align="center">
  <img src="figs/pipeline.png" alt="EigenBench pipeline" width="90%">
</p>

## Table of Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [Run Spec](#run-spec)
- [Spec Modes](#spec-modes)
  - [Spec Mode: Full Pipeline](#spec-mode-full-pipeline)
  - [Spec Mode: Train Only](#spec-mode-train-only)
  - [Spec Mode: Collect Only](#spec-mode-collect-only)
  - [Spec Mode: Cache Only](#spec-mode-cache-only)
  - [Spec Mode: Mixed HF Local + OpenRouter](#spec-mode-mixed-hf-local--openrouter)
  - [Spec Mode: All-to-All Collection](#spec-mode-all-to-all-collection)
- [Bootstrap Resampling](#bootstrap-resampling)
- [Outputs](#outputs)
- [Repo Layout](#repo-layout)
- [Datasets Used in the Paper](#datasets-used-in-the-paper)
- [Citation](#citation)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set API keys in `.env`:

- `OPENROUTER_API_KEY` — required for OpenRouter models
- `HF_TOKEN` — required for gated/private Hugging Face models (also reads from `~/.huggingface/token` via `huggingface-cli login`)

## Quick Start

1. Create a run folder.

```bash
mkdir -p runs/my_run
```

2. Copy the example spec.

```bash
cp runs/example/spec.py runs/my_run/spec.py
```

3. Edit `runs/my_run/spec.py`:

- required:
  - `models`
  - `dataset.path`
  - `constitution.path`
  - `constitution.num_criteria`
- common toggles (see [Spec Modes](#spec-modes) for more details):
  - `collection.enabled`
  - `collection.cached_responses_path`
  - `training.enabled`

4. Run:

```bash
python scripts/run.py runs/my_run/spec.py
```

Mixed-model runs work out of the box — just prefix local model paths with `hf_local:` in your spec. The pipeline auto-detects and batches local models through vLLM while routing API models through OpenRouter.

## Run Spec

Top-level keys in `RUN_SPEC`:

- `models`: `{display_name: openrouter_model_id}` or `{display_name: hf_local:<hf_model_path>}`
- `dataset`: scenario source and slicing.
- `constitution`: constitution file path and criterion count.
- `collection`: evaluation sampling/collection settings.
- `training`: BT/BTD training settings.

### Dataset controls

- `path`: JSON file of scenarios.
- `start`: start offset (default `0`).
- `count`: number of scenarios after `start` (omit for all remaining).
- `shuffle`: shuffle before slicing.
- `shuffle_seed`: reproducible shuffle seed.

### Constitution controls

- `path`: constitution JSON file.
- `num_criteria` (required): hard cap used for collection + extraction.

## Spec Modes

### Spec Mode: Full Pipeline

```python
"collection": {
    "enabled": True,
    "cached_responses_path": "data/responses/main_cache.jsonl",  # optional
},
"training": {
    "enabled": True,
}
```

Behavior:

- If `cached_responses_path` is set, cache stage runs first.
- Then evaluation collection runs.
- Then training/eigentrust runs.

### Spec Mode: Train Only

```python
"collection": {
    "enabled": False,
    "evaluations_path": "runs/my_run/evaluations.jsonl",
},
"constitution": {
    "path": "data/constitutions/kindness.json",
    "num_criteria": 8,
},
"training": {
    "enabled": True,
}
```

Use this when you already have evaluation transcripts and only want BT/BTD + EigenTrust outputs.

### Spec Mode: Collect Only

```python
"collection": {
    "enabled": True,
},
"training": {
    "enabled": False,
}
```

Use this to build/append `evaluations.jsonl` without running model fitting.

### Spec Mode: Cache Only

```python
"collection": {
    "enabled": False,
    "cached_responses_path": "data/responses/main_cache.jsonl",
},
"training": {
    "enabled": False,
}
```

Use this to precompute model responses for scenarios.

### Spec Mode: Mixed HF Local + OpenRouter

Mix OpenRouter API models and local Hugging Face models in the same run. Local models are automatically batched through vLLM for efficient GPU inference, while API models are called through OpenRouter. Use `hf_local:` prefixes in your `models` dict:

```python
"models": {
    "Claude 4 Sonnet": "anthropic/claude-sonnet-4",                      # OpenRouter
    "Qwen-sarcasm": "hf_local:maius/qwen-2.5-7b-it-personas/sarcasm",     # lora
    "Qwen": "hf_local:Qwen/Qwen2.5-7B-Instruct",                       # local
},
"collection": {
    "enabled": True,
    "sampler_mode": "random_judge_group",  # or "all_to_all"
},
"training": {
    "enabled": True,
}
```

The pipeline auto-detects `hf_local:` models and routes to the mixed collection path, which runs in 3 batched phases:

1. **Responses** — all evaluee responses (OpenRouter sequential, vLLM batched)
2. **Reflections** — all judge reflections (OpenRouter sequential, vLLM batched)
3. **Comparisons** — all pairwise comparisons (OpenRouter sequential, vLLM batched)

This is significantly faster than one-at-a-time API-style calls for local models.

LoRA adapter syntax: `hf_local:org/repo/subfolder` — the subfolder is resolved as a LoRA adapter on the base model detected from `adapter_config.json`.

### Spec Mode: All-to-All Collection

Use `sampler_mode: "all_to_all"` for exhaustive evaluation where every model judges every other model's response on every scenario:

```python
"collection": {
    "enabled": True,
    "sampler_mode": "all_to_all",
},
"training": {
    "enabled": True,
}
```

In all-to-all mode:

- Every model acts as a judge for every scenario
- Every model's response is evaluated by every judge
- Reflections are **per-judge** (each judge reflects independently on each response)
- All ordered pairs `(eval1, eval2)` are compared

This produces the most complete evaluation matrix but scales as `O(scenarios × models² × models²)`

## Bootstrap Resampling

Adds error bars to EigenBench Elo scores by resampling comparisons and retraining BT/BTD models.

```python
"training": {
    "bootstrap": {
        "enabled": True,
        "n_bootstraps": 100,
        "random_seed": 42,
        "save_models": False,
        "save_trust_matrices": True,
    },
}
```

> [!WARNING]
> Bootstrap only retrains the BT/BTD model. Run it locally on CPU to avoid wasting GPU compute time.

## Outputs

Per run folder (`runs/<run_name>/`):

- `evaluations.jsonl` (if collection ran)
- `btd_d<dim>/` folders (if training ran), containing:
  - `training_loss.png`
  - `model.pt`
  - `eigentrust.txt`
  - `uv_embeddings_pca.png`
  - `eigenbench.png`
  - `log_train.txt`
  - `bootstrap/` (if bootstrap enabled):
    - `samples.json`
    - `summary.json`
    - `bootstrap_elo.png`

## Repo Layout

```text
EigenBench/
├── pipeline/
│   ├── eval/          # collection orchestration + sampling
│   │   ├── collect.py             # OpenRouter-only collection
│   │   ├── mixed_collect.py       # mixed OpenRouter + vLLM collection (+ all-to-all)
│   │   ├── criteria_collectors.py # prompt builders + single-group collection
│   │   ├── samplers.py            # judge/evaluee sampling strategies
│   │   └── flows.py               # response-only collection
│   ├── train/         # BT/BTD fitting + plots
│   │   ├── bt_models.py           # VectorBT, VectorBTD, CriteriaVectorBTD
│   │   ├── train.py               # training loop + utilities
│   │   └── plots.py               # embedding + Elo visualizations
│   ├── trust/         # trust matrix + EigenTrust
│   ├── utils/         # record IO + comparison extraction
│   ├── config/        # run-spec + dataset/constitution loaders
│   └── providers/     # model API calls (OpenRouter + vLLM)
├── scripts/
│   ├── run.py                    # only user entrypoint
│   ├── run_collect.py            # internal: routes to mixed or OpenRouter-only collection
│   ├── run_collect_responses.py  # internal: response cache stage
│   └── run_train.py              # internal: training stage
├── notebooks/
│   ├── mixed_openrouter_local_collection.ipynb  # legacy notebook (now integrated into CLI)
│   ├── bootstrap_resampling.ipynb               # bootstrap analysis
├── runs/
│   └── <run_name>/
│       ├── spec.py            # per-run config
│       ├── evaluations.jsonl  # collected judgments
│       └── btd_d<dim>/        # training outputs
├── data/
│   ├── constitutions/         # committed constitutions
│   ├── scenarios/             # local scenario datasets
│   └── responses/             # shared cached responses
```

## Datasets Used in the Paper

- AskReddit: https://www.kaggle.com/datasets/rodmcn/askreddit-questions-and-answers
- OpenAssistant: https://huggingface.co/datasets/OpenAssistant/oasst1
- AIRiskDilemmas (LitmusValues): https://huggingface.co/datasets/kellycyy/AIRiskDilemmas

## Citation

```bibtex
@misc{chang2025eigenbenchcomparativebehavioralmeasure,
      title={EigenBench: A Comparative Behavioral Measure of Value Alignment},
      author={Jonathn Chang and Leonhard Piff and Suvadip Sana and Jasmine X. Li and Lionel Levine},
      year={2025},
      eprint={2509.01938},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.01938},
}
```
