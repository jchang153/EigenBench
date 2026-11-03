# Reproducing the native direct-rating task

This procedure prepares issue #35. Two complete hosted-model runs and their
result comparison are still required before Register submission. No release
commit or model results are certified by this document.

## Source and environment

After this change and its metrics dependency (#44) are reviewed and published,
use that public **40-character commit**, rather than a moving branch:

```bash
git clone https://github.com/jchang153/EigenBench.git
cd EigenBench
git checkout <validated-40-character-commit>
uv sync --frozen --python python3.11
uv run --frozen inspect --version
git rev-parse HEAD
git status --porcelain
```

Use native Python 3.11 for your operating system. Apple Silicon must use an arm64
interpreter rather than an older cached Intel interpreter. The lockfile fixes
the hosted environment, including Inspect 0.3.279. Its optional Linux local extra
does not certify GPU serving. The compatibility suite also exercises Inspect
0.3.268.

Scenarios come from `kellycyy/AIRiskDilemmas`, revision
`8674d1f5844c3909b05e06d9f30bbc2b7c753f39`, file `model_eval.jsonl`.
Consecutive action rows are validated as pairs, then each dilemma is retained
once in source order. The task selects the first 100 unique dilemmas. The eight
kindness criteria are checked in at `data/constitutions/kindness.json`.
Task metadata records selected scenario and criteria hashes, resolved panel,
sampling, generation, cache, retry and scoring settings. Samples retain scenario
text, responses, reflections and judgments.

Hosted model IDs identify provider endpoints; their weights and routing may
change. Source and dependency pins make runs auditable, but cannot guarantee
identical future API completions.

## Two complete runs

Review the call estimate before authorizing inference:

```bash
uv run --frozen python scripts/run.py runs/default/spec.py --estimate-calls
```

Each default run has 400 responses and 1,600 directed judgments, including
self-judgments. Responses, reflections and ratings total 3,600 logical
generations before retries. These commands use two existing panel members as
the Inspect target, retaining the same four-member panel in each run:

```bash
uv run --frozen inspect eval inspect_pipeline/eigenbench.py@eigenbench \
  --model openrouter/openai/gpt-4.1-nano -T cache=false \
  --metadata source_commit="$(git rev-parse HEAD)" --log-dir logs/nano
uv run --frozen inspect eval inspect_pipeline/eigenbench.py@eigenbench \
  --model openrouter/google/gemini-2.5-flash-lite -T cache=false \
  --metadata source_commit="$(git rev-parse HEAD)" --log-dir logs/flash-lite
```

Set `OPENROUTER_API_KEY` in the environment. Both runs make paid calls to the
whole panel. Do not use `--limit`, `--sample-id`, `--no-score` or multiple epochs
for submission evidence. A target outside the panel adds a fifth member,
changing the denominator to 2,500 judgments. Provider errors or invalid judgments
must be resolved before calling the run complete.

Check each log in `inspect view`: status is success, all 1,600 samples succeeded,
`ranking_available=1`, `coverage/scored=coverage/planned=1600`, and each member
has EigenTrust, Elo, rank and bootstrap metrics. Both runs include the same four
members, but their generated responses are independent.

## Sharing logs

Retain private originals. Prepare a new copy of each complete log:

```bash
uv run --frozen python scripts/prepare_inspect_log.py \
  logs/nano/<completed-log>.eval public-logs/nano.eval
uv run --frozen python scripts/prepare_inspect_log.py \
  logs/flash-lite/<completed-log>.eval public-logs/flash-lite.eval
```

The command checks native rankings, complete planned coverage and a full source
SHA. It removes operational local paths, the current username, raw API payloads,
tracebacks, credential fields and credential values present in the environment.
Use `--redact <identifier>` for another private identifier. It refuses redactions
that change benchmark text or dictionary keys. Unknown private content still
needs manual inspection before upload. Public copies are for review; use the
original logs for retries, exports and rescoring.

## Result comparison

The default is a **direct-rating variant**, with criterion-wise integer ratings
on a 1–10 scale and `zscore_softmax` normalization. The paper's pairwise task uses
grouped responses, ordered comparisons and a BTD fit. Their numbers are not
interchangeable. Sampling, prompts, ties, normalization and bootstrap units are
preserved by the migration; no new estimator changes are approved here.

Frozen synthetic fixtures in `tests/fixtures/inspect_migration/` establish
compatibility with analysis at `a903a21`: direct bootstrap summaries within
`1e-10`, BTD trust within `1e-7` and BTD Elo within `1e-4`. They do not reproduce
the paper's experimental results. Direct intervals resample scenarios; pairwise
intervals resample cleaned comparison rows. The pairwise bootstrap seed does not
seed initialization or loader shuffling. Issue #41 remains deferred.

Before closing #35, record the validated public SHA, both full log filenames,
model endpoint IDs, coverage, per-member rankings and intervals, and compare the
results with an identified existing EigenBench output. Explain differences in
protocol, panel, scenarios, criteria and decoding before attributing score
differences to the implementation. #36 tracks the subsequent Register submission.
