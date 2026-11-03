# Frozen migration fixtures

`protocol.json` stores rendered prompt bytes and direct assignment plans from
baseline `a903a214886af74d9dbd4478d41448a7036bdcee`. Pairwise seeded plans and
prompt bytes also have literal fixtures in `test_inspect_pairwise.py`.

`direct.jsonl` and `pairwise.jsonl` are synthetic historical-format records,
covering three models, three scenarios, two criteria, self-judging and ties.
They are not published experimental results. `analysis.json` freezes the
baseline `scripts.run_train.main` results for these records, including trust
matrices, EigenTrust/Elo, fitted BTD loss, and bootstrap samples and intervals.

The CPU fixture uses seed 73 for Python, NumPy, and PyTorch, pooled-criteria BTD,
dimension 2, three training epochs, batch size 32, and three bootstrap replicates
with resampling seed 23. These small settings test compatibility, not estimator
quality. No inference was used to generate these files.
