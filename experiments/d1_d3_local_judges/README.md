# Local-judge D1 versus D3 comparison

This experiment repeats the balanced D1/D3 prompt-structure comparison with
four local 7B judges and a bundled 10-by-5 cached-response matrix. It keeps the
kindness constitution and exact D1/D3 prompts, but uses AiriskDilemmas inputs
from the published reasoning-versus-instant ValueArena run instead of the
original AskReddit inputs. The judge axis is:

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

Because both the judges and response sample differ from the original balanced
matrix, compare D1 versus D3 within this run; do not compare its absolute
ratings directly with the prior AskReddit matrix.

## Inputs

The repository bundles the fifty responses at:

    experiments/d1_d3_local_judges/data/airisk_10x5_responses.jsonl

They are the first ten AiriskDilemmas scenarios (indices 0 through 9) and the
instant responses from five distinct model families:

- GPT-5.6 Luna
- Claude Haiku 4.5
- Grok 4.3
- DeepSeek V4 Flash 0423
- Gemini 3.1 Flash Lite

The source cache is from the ValueArena run
`reasoning-vs-instant/airisk-kindness-200`. The runner verifies every scenario
and response hash against reference_manifest.json before loading a judge. An
alternate response path can still be supplied with `--responses`, but it must
contain these exact fifty cells.

## RunPod setup

Use a single NVIDIA GPU with at least 24 GB VRAM and enough persistent disk for
roughly 65 GB of model weights plus environment caches. From the repository
root on RunPod:

    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt

Confirm the immutable model revisions and call plan without downloading models:

    .venv/bin/python -m experiments.d1_d3_local_judges.run --plan

Verify all fifty bundled response hashes without downloading models:

    .venv/bin/python -m experiments.d1_d3_local_judges.run --validate-input

Run inside tmux so an SSH or laptop disconnect does not stop collection:

    tmux new -s d1-d3-local
    .venv/bin/python -m experiments.d1_d3_local_judges.run \
      --output-dir /workspace/d1_d3_local_judges \
      --batch-size 64 \
      --verbose

Detach with Ctrl-b followed by d. Reattach with:

    tmux attach -t d1-d3-local

The models are loaded and released one at a time. The command is resumable:
rerun it with the same output directory after any interruption. To run one
model at a time, repeat --judge with an exact name; the matrices finalize once
all four judges are complete.

Each call is attempted up to `max_attempts`. If its output still fails
validation, the terminal failure remains in the checkpoint and collection
continues without retrying it on later resumes. If a reflection fails, its
dependent judgment call(s) are explicitly marked as dependency-skipped. A
judge/evaluee/scenario cell is included in the D1-versus-D3 analysis only when
both designs have complete ratings, so all reported comparisons remain paired.

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

- cell_results.jsonl: up to 400 complete judge/evaluee/scenario/design records
- stage_results.jsonl: parsed reflections and judgments
- raw_calls.jsonl: raw completions, attempts, and token counts
- failed_calls.jsonl: every exhausted failure and dependency-skipped call,
  including its full judge/evaluee/scenario/design identity
- failure_summary.json: separate lists and counts for actual model failures and
  calls skipped because their reflection dependency failed
- inputs.jsonl: the fifty exact scenario/response inputs for future reuse
- manifest.json: hashes, model revisions, prompts, and execution settings
- pair_summary.csv: paired metrics for every judge/evaluee pair
- judge_summary.csv: paired metrics kept separate by judge
- matrix_d1.csv and matrix_d3.csv: judge-row/evaluee-column score means
- matrix_difference.csv: D3 minus D1 mean ratings
- matrix_mae.csv: matched mean absolute rating changes
- summary.json: overall, judge-wise, completeness, bootstrap, and token metrics
- checkpoint/: per-call resumable state

Each matrix entry averages only across matched scenarios and eight criteria for
that particular judge/evaluee pair. When a call fails, the corresponding
scenario is excluded from that judge/evaluee entry and its exact identity is
listed in the failure artifacts. Judges and evaluees are never pooled before
that cell-level comparison.
