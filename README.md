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

3. Run the full pipeline (either form works):

```bash
python scripts/run_pipeline.py runs.my_run.spec
python scripts/run_pipeline.py runs/my_run/spec.py
```

Default outputs for this run go into:
- `runs/my_run/evaluations.jsonl`
- `runs/my_run/train/`

Optional shared response cache (for reuse across runs):
- set `collection.cached_responses_path`, e.g. `data/cache/responses/reddit_main.jsonl`

## Repo Layout

```text
EigenBench/
├── pipeline/
│   ├── eval/          # collection orchestration + samplers + response-only flow
│   ├── train/         # BT/BTD models + training
│   ├── trust/         # trust matrix and EigenTrust
│   ├── io/            # JSON/JSONL evaluation IO + comparison extraction
│   ├── config/        # run-spec loading + dataset/constitution path helpers
│   └── providers/     # OpenRouter call helper
├── scripts/
│   ├── run_collect.py            # collect evaluations only
│   ├── run_collect_responses.py  # collect response cache only
│   ├── run_train.py              # train only
│   └── run_pipeline.py           # collect + train
├── runs/
│   └── <run_name>/
│       ├── spec.py            # per-run settings (models, dataset, constitution, training)
│       ├── evaluations.jsonl  # collected judge comparisons for this run
│       └── train/             # model checkpoints, loss plots, eigentrust outputs
├── data/
│   ├── constitutions/         # committed constitution JSON files used by specs
│   │   └── *.json
│   ├── scenarios/             # local scenario datasets (git-ignored)
│   │   └── *.json
│   └── cache/
│       └── responses/         # shared cached responses across runs (git-ignored)
```

## What Each Script Does

- All run scripts accept either:
  - a dotted module path, e.g. `runs.example.spec`
  - a file path, e.g. `runs/example/spec.py`

- `scripts/run_collect.py`: reads a run spec, samples scenarios, collects model responses + judge reflections + pairwise judge choices, and appends rows to `collection.evaluations_path`.
  If that path is omitted, it defaults to `runs/<run_name>/evaluations.jsonl`.

- `scripts/run_collect_responses.py`: reads a run spec and collects only evaluee responses into `collection.cached_responses_path`.
  `collection.cached_responses_path` is required for this script.

- `scripts/run_train.py`: reads evaluations from `collection.evaluations_path`, trains BT/BTD per `training` config, and computes EigenTrust at the end of each training run (`eigentrust.txt` in each output folder).
  If `training.output_dir` is omitted, it defaults to `runs/<run_name>/train`.

- `scripts/run_pipeline.py`: runs collect first, then training, using one run spec.

## Common Use Cases

1. Run the full pipeline (collect + train)

```bash
python scripts/run_pipeline.py runs.example.spec
python scripts/run_pipeline.py runs/example/spec.py
```

2. You already have an evaluations file and only want training

- Point `collection.evaluations_path` in your run spec to that existing `json/jsonl` file.
- Then run:

```bash
python scripts/run_train.py your.spec.module
python scripts/run_train.py runs/my_run/spec.py
```

3. You only want to collect evaluations

```bash
python scripts/run_collect.py your.spec.module
python scripts/run_collect.py runs/my_run/spec.py
```

4. Collect responses first, cache them, then do judging later

- Step A: set `collection.cached_responses_path` in your run spec (recommended shared path under `data/cache/responses/`).
- Step B: run response-only collection:

```bash
python scripts/run_collect_responses.py runs.example.spec
python scripts/run_collect_responses.py runs/example/spec.py
```

- Step C: run normal collection (`run_collect.py`). It will reuse cached evaluee responses and only do reflection/comparison calls.

## Datasets Used in the Paper

- AskReddit dataset: https://www.kaggle.com/datasets/rodmcn/askreddit-questions-and-answers
- OpenAssistant: https://huggingface.co/datasets/OpenAssistant/oasst1
- AIRiskDilemmas (LitmusValues): https://huggingface.co/datasets/kellycyy/AIRiskDilemmas

## Legacy Artifacts

Legacy experiment code, notebooks, and old experiment outputs were moved under `deprecated/`.
