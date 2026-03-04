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
  - [Spec Mode: Mixed HF Local + OpenRouter (Notebook)](#spec-mode-mixed-hf-local--openrouter-notebook)
- [Outputs](#outputs)
- [Repo Layout](#repo-layout)
- [Datasets Used in the Paper](#datasets-used-in-the-paper)
- [Citation](#citation)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy scikit-learn matplotlib tqdm python-dotenv openai
```

Set API keys in `.env`:
- `OPENROUTER_API_KEY`

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
- common toggles:
  - `verbose`
  - `collection.enabled`
  - `collection.cached_responses_path` (optional shared cache)
  - `training.enabled`

4. Run:

```bash
python scripts/run.py runs/my_run/spec.py
```

Quick mixed-model option:
- For mixed OpenRouter + local Hugging Face models, use [notebooks/mixed_openrouter_local_collection.ipynb](notebooks/mixed_openrouter_local_collection.ipynb) for collection.
- Then run the standard training path from your spec (for example, with `collection.enabled=False` if collection is already complete).

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

### Spec Mode: Mixed HF Local + OpenRouter (Notebook)

Use [notebooks/mixed_openrouter_local_collection.ipynb](notebooks/mixed_openrouter_local_collection.ipynb) for mixed populations where model values can include `hf_local:<hf_model_path>` alongside OpenRouter model IDs.

After notebook collection completes, re-run standard training with collection disabled:

```python
"collection": {
    "enabled": False,
    "evaluations_path": "runs/my_run/evaluations.jsonl",
},
"training": {
    "enabled": True,
}
```

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

## Repo Layout

```text
EigenBench/
├── pipeline/
│   ├── eval/          # collection orchestration + sampling
│   ├── train/         # BT/BTD fitting + plots
│   ├── trust/         # trust matrix + EigenTrust
│   ├── utils/         # record IO + comparison extraction
│   ├── config/        # run-spec + dataset/constitution loaders
│   └── providers/     # model API calls
├── scripts/
│   ├── run.py                    # only user entrypoint
│   ├── run_collect.py            # internal stage module
│   ├── run_collect_responses.py  # internal stage module
│   └── run_train.py              # internal stage module
├── notebooks/
│   └── mixed_openrouter_local_collection.ipynb  # mixed HF-local + OpenRouter collection
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
