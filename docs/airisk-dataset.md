# Automatic AIRiskDilemmas loading

A built-in dataset ID needs no notebook setup:

```python
RUN_SPEC["dataset"] = {"id": "airisk", "start": 0, "count": 400}
```

On first use, the loader downloads `model_eval.jsonl` from
`kellycyy/AIRiskDilemmas` at revision
`8674d1f5844c3909b05e06d9f30bbc2b7c753f39`. Later loads use the Hugging Face cache
without a network check. The loader validates adjacent action pairs and retains
each question once, in source order, before selection. The pinned source has
2,999 unique question texts. `count` refers to unique questions.

The ID bypasses the old `data/scenarios/airiskdilemmas.json`, which notebooks may
have overwritten with raw rows. Manual `dataset.path` files remain supported.
The optional `scripts/prepare_airiskdilemmas.py` command materializes the same
prepared question list. Raw action rows and duplicate scenario text in manual
files are rejected.

## Existing runs

Prepared question indices differ from indices in the raw action-row dataset.
Do not resume an old raw-row run or reuse its indexed response cache with the
new ID. Use a new run/output directory. For a legitimate historical extension,
retain the exact original unique scenario file and reference it by explicit path.
