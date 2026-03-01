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

2. Create `runs/my_run/config.py` (copy from `runs/example/config.py` and edit):

- update `models`
- set `dataset.path` to your scenarios JSON
- set `constitution.path` to your constitution JSON

3. Run the full pipeline (either form works):

```bash
python scripts/run_pipeline.py runs.my_run
python scripts/run_pipeline.py runs/my_run/config.py
```

Default outputs for this run go into:
- `runs/my_run/out/evaluations.jsonl`
- `runs/my_run/out/cached_responses.jsonl`
- `runs/my_run/out/train/`

## Repo Layout

```text
EigenBench/
├── pipeline/
│   ├── eval/          # collection orchestration + samplers + response-only flow
│   ├── train/         # BT/BTD models + training
│   ├── trust/         # trust matrix and EigenTrust
│   ├── io/            # JSON/JSONL evaluation IO + comparison extraction
│   ├── config/        # dataset + constitution access helpers
│   └── providers/     # OpenRouter call helper
├── scripts/
│   ├── run_collect.py
│   ├── run_collect_responses.py
│   ├── run_merge_evaluations.py
│   ├── run_train.py
│   └── run_pipeline.py
├── notebooks/
│   └── mixed_openrouter_local_collection.ipynb
├── runs/
│   └── <run_name>/    # run package + run artifacts
│       ├── __init__.py   # optional (needed only for dotted import style)
│       ├── config.py
│       └── out/
│           ├── evaluations.jsonl
│           ├── cached_responses.jsonl
│           └── train/
├── data/
│   ├── scenarios/
│   │   ├── reddit_questions.json
│   │   ├── oasst_questions.json
│   │   └── airiskdilemmas.json
│   ├── constitutions/
│   │   └── *.json
│   └── cache/
│       └── responses/ # recommended location for cached model responses
```

## What Each Script Does

- All run scripts accept either:
  - a dotted module path, e.g. `runs.example`
  - a file path, e.g. `runs/example/config.py`

- `scripts/run_collect.py`: reads a run spec, samples scenarios, collects model responses + judge reflections + pairwise judge choices, and appends rows to `collection.evaluations_path`.
  If that path is omitted, it defaults to `runs/<run_name>/out/evaluations.jsonl`.

- `scripts/run_collect_responses.py`: reads a run spec and collects only evaluee responses into `collection.cached_responses_path`.
  If omitted, it defaults to `runs/<run_name>/out/cached_responses.jsonl`.

- `scripts/run_train.py`: reads evaluations from `collection.evaluations_path`, trains BT/BTD per `training` config, and computes EigenTrust at the end of each training run (`eigentrust.txt` in each output folder).
  If `training.output_dir` is omitted, it defaults to `runs/<run_name>/out/train`.

- `scripts/run_pipeline.py`: runs collect first, then training, using one run spec.

- `scripts/run_merge_evaluations.py`: merges multiple evaluation files, remaps `eval1/eval2/judge` indices from `*_name` fields to your run-spec model order, and writes one deduplicated merged evaluations file.

## Common Use Cases

1. Run the full pipeline (collect + train)

```bash
python scripts/run_pipeline.py runs.example
python scripts/run_pipeline.py runs/example/config.py
```

2. You already have an evaluations file and only want training

- Point `collection.evaluations_path` in your run spec to that existing `json/jsonl` file.
- Then run:

```bash
python scripts/run_train.py your.spec.module
```

3. You only want to collect evaluations

```bash
python scripts/run_collect.py your.spec.module
```

4. Collect responses first, cache them, then do judging later

- Step A: optionally set `collection.cached_responses_path` in your run spec (otherwise default is used).
- Step B: run response-only collection:

```bash
python scripts/run_collect_responses.py runs.example
python scripts/run_collect_responses.py runs/example/config.py
```

- Step C: run normal collection (`run_collect.py`). It will reuse cached evaluee responses and only do reflection/comparison calls.

5. Mixed OpenRouter + local/HF models

- Use the notebook:
- `notebooks/mixed_openrouter_local_collection.ipynb`
- It collects in the same evaluation schema as `run_collect.py`, but allows `hf_local:<model_id>` entries.
- Keep model nicknames consistent with your run spec if you plan to train on merged data.
- If you collected separate files (for example OpenRouter-only and local/HF runs), merge them:

```bash
python scripts/run_merge_evaluations.py runs.example runs/example/out/evaluations_merged.jsonl runs/example/out/evaluations_or.jsonl runs/example/out/evaluations_local_openrouter.jsonl
```

- Point `collection.evaluations_path` at the merged file and run:

```bash
python scripts/run_train.py runs.example
python scripts/run_train.py runs/example/config.py
```

## Datasets Used in the Paper

- AskReddit dataset: https://www.kaggle.com/datasets/rodmcn/askreddit-questions-and-answers
- OpenAssistant: https://huggingface.co/datasets/OpenAssistant/oasst1
- AIRiskDilemmas (LitmusValues): https://huggingface.co/datasets/kellycyy/AIRiskDilemmas

## Legacy Artifacts

Legacy experiment code, notebooks, and old experiment outputs were moved under `deprecated/`.
