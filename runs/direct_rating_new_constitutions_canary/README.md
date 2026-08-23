# New-constitution direct-rating canary

This canary uses AIRiskDilemmas scenarios 200-209, five OpenRouter models, and
four provisional constitutions. Responses use the original EigenBench prompt
and are cached once for reuse across constitutions.

Run each constitution in this order so the first run creates the shared cache:

```bash
.venv/bin/python scripts/run.py runs.direct_rating_new_constitutions_canary.buddhism
.venv/bin/python scripts/run.py runs.direct_rating_new_constitutions_canary.care_ethics
.venv/bin/python scripts/run.py runs.direct_rating_new_constitutions_canary.lockean_rights
.venv/bin/python scripts/run.py runs.direct_rating_new_constitutions_canary.marxism
```

Add `--estimate-calls` to any command to inspect its planned request count
without making API calls.

## Generation settings

The response, reflection, and XML rating prompts match `jonathan/main`. The
temperatures are also unchanged: 0.7, 0.2, and 0.0 respectively.

The final output caps are 4096 tokens for all three phases. DeepSeek uses 8192
for reflection and rating. The original 2048-token reflection and 512-token
rating caps were too short for the selected reasoning models: some calls used
the full cap before returning visible text or all 12 XML ratings.

## Canary result (2026-08-23)

- Shared responses: 10 scenarios x 5 models = 50
- Direct-rating records: 50 per constitution = 200
- Parsed criterion ratings: 200 x 12 = 2400
- Final checkpoint failures: 0
- Pre-run cost estimate: about $2.50
- OpenRouter account usage increase during the run: $5.46 (2.2x estimate)

The estimate assumed the original token caps and no recovery calls. The run used
higher caps for reasoning models, discarded 32 completed reflections during the
clean restart, and made 32 failed attempts before the cap problems were fixed.
The prompts themselves were not changed.

All four finalized files passed record-count, checkpoint-hash, response-cache,
assignment, rating-range, and XML-parser checks. Generated responses and
ratings remain local and are ignored by Git.
