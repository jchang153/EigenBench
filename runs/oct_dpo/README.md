# Post-DPO OCT EigenBench Run

These eleven run specs repeat the published post-introspection OCT evaluation
configuration with the released post-DPO, pre-introspection LoRA adapters.

Each row has a self-contained `spec.py` with:

- eleven post-DPO Qwen LoRAs, the Qwen base model, and the three historical
  OpenRouter models;
- AIRiskDilemmas scenarios 100-199;
- the grammatically corrected constitutions currently checked into EigenBench;
- one random judge and four evaluees per scenario, with `sampler_seed=42`;
- BTD with ties at dimension 2 and 100 bootstrap runs.

The sampler generates assignments normally. There is no separate frozen
assignment file; the collection checkpoint records the generated assignments
only so an interrupted API run can resume consistently.

On the GPU pod, install the repository requirements and set:

- `OPENROUTER_API_KEY` for the three API models;
- `SPACE_SECRET`, obtained from the ValueArena maintainer, to authorize the
  automatic Space submission.

`HF_TOKEN` is optional for downloading the public Hugging Face artifacts. The
ValueArena Space uses its own write credential, so your pod does not need a
Hugging Face write token. EigenBench itself does not use a `RUNPOD_API_KEY`.

Download AIRiskDilemmas once after cloning the repository. The scenario file
is a local run input and is intentionally ignored by Git:

```bash
python -c 'from datasets import load_dataset; import json, pathlib; p = pathlib.Path("data/scenarios/airiskdilemmas.json"); p.parent.mkdir(parents=True, exist_ok=True); json.dump(load_dataset("kellycyy/AIRiskDilemmas", split="test").to_list(), p.open("w"), indent=2)'
```

Automatic ValueArena upload is enabled in every row's spec. After each row's
collection finishes, local BTD training is skipped and the evaluations are
submitted to the ValueArena Space, which trains BTD, runs the bootstrap, and
publishes the result under `oct-dpo/<trait>`. Submission runs in the
background; `scripts/run.py` prints the corresponding `/tmp/va_submit_*.log`
path.

Run the goodness row as a pilot:

```bash
python scripts/run.py runs/oct_dpo/goodness/spec.py
```

After its collection and ValueArena upload succeed, launch each remaining row
through the existing EigenBench entrypoint:

```bash
python scripts/run.py runs/oct_dpo/humor/spec.py
python scripts/run.py runs/oct_dpo/impulsiveness/spec.py
python scripts/run.py runs/oct_dpo/loving/spec.py
python scripts/run.py runs/oct_dpo/mathematical/spec.py
python scripts/run.py runs/oct_dpo/misalignment/spec.py
python scripts/run.py runs/oct_dpo/nonchalance/spec.py
python scripts/run.py runs/oct_dpo/poeticism/spec.py
python scripts/run.py runs/oct_dpo/remorse/spec.py
python scripts/run.py runs/oct_dpo/sarcasm/spec.py
python scripts/run.py runs/oct_dpo/sycophancy/spec.py
```

Re-running the same row uses its collection checkpoint to reuse completed
OpenRouter calls. Evaluation and training outputs are written inside that
row's directory and are ignored by Git.
