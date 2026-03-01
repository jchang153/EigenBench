# EigenBench

EigenBench benchmarks model-vs-model behavioral alignment from pairwise LM judgments, then fits BT/BTD and aggregates with EigenTrust.

Paper: [EigenBench: A Comparative Behavioral Measure of Value Alignment](https://arxiv.org/abs/2509.01938)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy scipy pandas scikit-learn matplotlib tqdm python-dotenv openai anthropic google-genai
```

Set API keys in `.env`:
- `OPENROUTER_API_KEY`

Note: the new collection path is OpenRouter-only.

## Quick Start (Create a Run)

1. Create a run folder:

```bash
mkdir -p runs/my_run
```

2. Create `runs/my_run/spec.py` (copy from `runs/example/spec.py` and edit):

- update `models`
- set `dataset.path` to your scenarios JSON
- set `constitution.path` to your constitution JSON
- set `constitution.num_criteria` (required; used to cap/truncate criteria)
- set `verbose` (`False` for quiet runs, `True` for detailed logs)
- stage toggles:
  - `collection.enabled`: run evaluation collection
  - `collection.cached_responses_path`: if set, response cache collection runs first
  - `training.enabled`: run training + eigentrust
- optional dataset selection controls:
  - `dataset.start`: start offset in the scenario list (default `0`)
  - `dataset.count`: how many scenarios to use after `start` (omit for all remaining)
  - `dataset.shuffle`: shuffle before slicing (default `False`)
  - `dataset.shuffle_seed`: seed for reproducible shuffle (optional)

3. Run the full pipeline (either form works):

```bash
python scripts/run.py runs.my_run.spec
python scripts/run.py runs/my_run/spec.py
```

Default outputs for this run go into:
- `runs/my_run/evaluations.jsonl`
- `runs/my_run/btd_d<dim>/`

Optional shared response cache (for reuse across runs):
- set `collection.cached_responses_path`, e.g. `data/cache/responses/reddit_main.jsonl`

## Repo Layout

```text
EigenBench/
├── pipeline/
│   ├── eval/          # collection orchestration + samplers + response-only flow
│   ├── train/         # BT/BTD models + training
│   ├── trust/         # trust matrix and EigenTrust
│   ├── utils/         # JSON/JSONL evaluation IO + comparison extraction
│   ├── config/        # run-spec loading + dataset/constitution path helpers
│   └── providers/     # OpenRouter call helper
├── scripts/
│   ├── run.py                    # only user entrypoint: cache (optional) + collect (optional) + train (optional)
│   ├── run_collect.py            # internal stage module
│   ├── run_collect_responses.py  # internal stage module
│   └── run_train.py              # internal stage module
├── runs/
│   └── <run_name>/
│       ├── spec.py            # per-run settings (models, dataset, constitution, training)
│       ├── evaluations.jsonl  # collected judge comparisons for this run
│       └── btd_d<dim>/        # model checkpoints, loss plots, u/v PCA plot, eigentrust outputs
├── data/
│   ├── constitutions/         # committed constitution JSON files used by specs
│   │   └── *.json
│   ├── scenarios/             # local scenario datasets (git-ignored)
│   │   └── *.json
│   └── cache/
│       └── responses/         # shared cached responses across runs (git-ignored)
```

## How to Run

Use only one entrypoint:

```bash
python scripts/run.py runs.example.spec
python scripts/run.py runs/example/spec.py
```

`scripts/run.py` executes stages in this order:
1. Response cache collection (only if `collection.cached_responses_path` is set)
2. Evaluation collection (only if `collection.enabled=True`)
3. Training + EigenTrust (only if `training.enabled=True`)

Logging:
- `verbose=False` suppresses most stage/training parser logs
- `verbose=True` shows detailed progress/debug output

Training outputs include:
- `training_loss.png`
- `uv_embeddings_pca.png` (side-by-side 2D PCA of `u` and `v`, with model index/name and EigenBench scores in legend)
- `eigentrust.txt`

## Common Spec Modes

1. Full pipeline (cache + collect + train)
- Set `collection.cached_responses_path`
- Set `collection.enabled=True`
- Set `training.enabled=True`

2. Train only from an existing evaluations file
- Set `collection.enabled=False`
- Point `collection.evaluations_path` to your existing file
- Set `constitution.num_criteria` to the expected criterion count in that file
- Set `training.enabled=True`

3. Collect only (no training)
- Set `collection.enabled=True`
- Set `training.enabled=False`

4. Cache-only refresh
- Set `collection.cached_responses_path`
- Set `collection.enabled=False`
- Set `training.enabled=False`

## Datasets Used in the Paper

- AskReddit dataset: https://www.kaggle.com/datasets/rodmcn/askreddit-questions-and-answers
- OpenAssistant: https://huggingface.co/datasets/OpenAssistant/oasst1
- AIRiskDilemmas (LitmusValues): https://huggingface.co/datasets/kellycyy/AIRiskDilemmas

## Legacy Artifacts

Legacy experiment code, notebooks, and old experiment outputs were moved under `deprecated/`.
