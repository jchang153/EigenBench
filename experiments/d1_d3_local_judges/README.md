# Local-judge D1 versus D3 comparison

This experiment repeats the balanced D1/D3 prompt-structure comparison while
keeping the original five evaluees, ten scenarios, and fifty cached responses.
Only the judge axis changes:

- Qwen 2.5 7B Instruct
- Qwen 2.5 7B (base)
- OLMo 3 7B Instruct SFT
- OLMo 2 7B Instruct

Model revisions are pinned in config.json. The original D1 and D3 prompt files
are copied verbatim from jashvira/EigenBench at direct-ratings commit a99cbfa.
Outputs are constrained with vLLM JSON schemas, matching the externally
enforced structured-output contract used in the OpenRouter experiment.

Qwen 2.5 7B is intentionally the pretrained base model. It has no judge-tuned
chat behavior, so the runner renders its system and user messages as a plain
transcript. The other three judges use their native chat templates.

## Inputs

The prior result zip does not contain response text. Supply the original file:

    data/output/valuearena/processed/full8_kindness/askreddit_1000_responses_completed.json

The runner extracts only the original ten scenarios and five evaluees. It
verifies all fifty scenario and response hashes against reference_manifest.json
before loading any model. A source file containing only those exact records is
also accepted.

The reference scenario indices are:

    30, 110, 121, 150, 236, 258, 289, 664, 764, 769

## RunPod setup

Use a single NVIDIA GPU with at least 24 GB VRAM and enough persistent disk for
roughly 65 GB of model weights plus environment caches. From the repository
root on RunPod:

    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt

Confirm the immutable model revisions and call plan without downloading models:

    .venv/bin/python -m experiments.d1_d3_local_judges.run --plan

After transferring the response cache, verify all fifty hashes without
downloading models:

    .venv/bin/python -m experiments.d1_d3_local_judges.run \
      --responses /workspace/askreddit_1000_responses_completed.json \
      --validate-input

Run inside tmux so an SSH or laptop disconnect does not stop collection:

    tmux new -s d1-d3-local
    .venv/bin/python -m experiments.d1_d3_local_judges.run \
      --responses /workspace/askreddit_1000_responses_completed.json \
      --output-dir /workspace/d1_d3_local_judges \
      --batch-size 64 \
      --verbose

Detach with Ctrl-b followed by d. Reattach with:

    tmux attach -t d1-d3-local

The models are loaded and released one at a time. The command is resumable:
rerun it with the identical response file and output directory after any
interruption. To run one model at a time, repeat --judge with an exact name;
the matrices finalize once all four judges are complete.

These public Hugging Face models do not require an API key. An optional
HF_TOKEN can be supplied through the RunPod environment to avoid anonymous
download rate limits.

## Run shape

For every judge, the runner rates fifty fixed response cells:

- D1: one combined reflection and one combined judgment per cell, 400 calls
  over all four judges.
- D3: one structured reflection and eight isolated judgments per cell, 1,800
  calls over all four judges.
- Total: 2,200 local inference calls.

## Outputs

The output directory contains:

- cell_results.jsonl: 400 final judge/evaluee/scenario/design records
- stage_results.jsonl: parsed reflections and judgments
- raw_calls.jsonl: raw completions, attempts, and token counts
- inputs.jsonl: the fifty exact scenario/response inputs for future reuse
- manifest.json: hashes, model revisions, prompts, and execution settings
- pair_summary.csv: paired metrics for every judge/evaluee pair
- judge_summary.csv: paired metrics kept separate by judge
- matrix_d1.csv and matrix_d3.csv: judge-row/evaluee-column score means
- matrix_difference.csv: D3 minus D1 mean ratings
- matrix_mae.csv: matched mean absolute rating changes
- summary.json: overall, judge-wise, completeness, bootstrap, and token metrics
- checkpoint/: per-call resumable state

Each matrix entry averages only across the ten matched scenarios and eight
criteria for that particular judge/evaluee pair. Judges and evaluees are never
pooled before that cell-level comparison.
