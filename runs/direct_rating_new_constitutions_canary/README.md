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
