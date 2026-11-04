# Reproducing the native direct-rating task

Two full hosted-model runs completed on 3 November 2026. Both succeeded on all
1,600 planned judgments with native rankings and bootstrap intervals. The logs,
source pin and comparison below provide the run evidence for #35. Register
submission is tracked separately in #36.

## Source and environment

The validation used this public **40-character commit**, rather than a moving
branch:

```bash
git clone https://github.com/jchang153/EigenBench.git
cd EigenBench
git checkout cd1c3f2e576529a435a17ec49ff4d04792ae3d9d
uv sync --frozen --python "$(command -v python3.11)"
uv run --frozen inspect --version
git rev-parse HEAD
git status --porcelain
```

Install native Python 3.11 and put it on `PATH` before running these commands.
Passing its full path prevents uv from choosing an older cached interpreter.
Apple Silicon requires arm64 Python. The lockfile fixes
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

Retain private originals. Log preparation has a separate source pin containing
the complete-path privacy fix. Switching here changes the sharing helper; the
completed logs retain their original collection SHA.

```bash
git checkout 28e57cfe98b1b665e5d41df871d5c90232a4b345
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

## Completed validation — 3 November 2026

| Inspect target | Successful judgments | Criterion ratings | Native rankings / bootstrap |
| --- | ---: | ---: | --- |
| GPT-4.1 nano | 1,600 / 1,600 | 12,800 | Present / 100 scenario resamples |
| Gemini 2.5 Flash-Lite | 1,600 / 1,600 | 12,800 | Present / 100 scenario resamples |

Each log contains exactly 400 responses shared consistently across judges and
400 self-judgments. Both use the same scenarios and criteria; 399 of the 400
response hashes differ between the independent collection draws. The target
names label CLI runs; all four panel members act as judges and evaluees in both.

| Panel member | Nano-target EigenTrust | Nano-target Elo [95% interval] | Flash-target EigenTrust | Flash-target Elo [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| GPT-4.1 nano | 0.044583 | 1200.49 [1166.03, 1355.78] | 0.034652 | 1156.72 [1149.70, 1243.39] |
| Gemini 2.5 Flash-Lite | 0.301025 | 1532.26 [1390.30, 1614.57] | 0.405401 | 1583.98 [1446.64, 1635.17] |
| Llama 4 Scout | 0.257149 | 1504.90 [1390.86, 1620.09] | 0.252034 | 1501.41 [1390.24, 1611.64] |
| Mistral Small 3.2 | 0.397244 | 1580.45 [1481.26, 1639.50] | 0.307913 | 1536.20 [1400.87, 1626.44] |

These are panel-relative scores from stochastic draws. The overlapping intervals
and changes in order do not establish a general model ranking.

The initial nano-target attempt failed after an upstream Gemini 429. Its 97
logged samples (88 successful) and original log hash are retained in the run
summary; it does not count as a complete run. The full retry used adaptive
connections, up to 128 concurrent samples, 12 API retries and a 900-second retry
window. The Flash-target run started with conservative limits and recorded its
live control changes. Generation budgets, temperatures, sampling and scoring
were unchanged; exact launch commands and controls are in the summary.

Usage was approximately **$2.20**, below the approved $10 ceiling, including the
failed attempt. The token-price estimate is $2.07. The key-counter delta is
approximate because its persisted baseline was captured after the first attempt
began; both measurements and their scope are recorded in the summary.

Private originals were retained. Public copies preserve all exported judgment
records and numeric metrics. Archive scans and an independent review check the
complete copies before publication.

Download [nano.eval](https://github.com/jchang153/EigenBench/releases/download/validation-pr46-2026-11-03/nano.eval), [flash-lite.eval](https://github.com/jchang153/EigenBench/releases/download/validation-pr46-2026-11-03/flash-lite.eval),
[summary.json](https://github.com/jchang153/EigenBench/releases/download/validation-pr46-2026-11-03/summary.json), [historical-parity.json](https://github.com/jchang153/EigenBench/releases/download/validation-pr46-2026-11-03/historical-parity.json)
and [SHA256SUMS](https://github.com/jchang153/EigenBench/releases/download/validation-pr46-2026-11-03/SHA256SUMS).

The historical comparison uses 1,600 real records from
`runs/direct_rating_frontier_200/kindness`. Current analysis reproduces all eight
stored ranking rows with maximum EigenTrust difference `5.55e-17` and Elo
difference `2.27e-13`. It uses the original sparse, nonself, eight-model settings
only to check historical analysis parity. The new collections include self.

Historical and live runs have different panels, generation budgets and scenario
ordering; only six historical scenario texts overlap the new first 100. Their
absolute Elo levels cannot be compared as a controlled model experiment. The
historical artifact records source hashes, settings, per-model differences and
these limits. No claim of paper-result replication or live GPU support follows
from these checks.
