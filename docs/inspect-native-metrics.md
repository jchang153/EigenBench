# Native EigenBench metrics

Inspect's `panel_scorer` reports EigenTrust, display Elo and rank for each panel
member. An added `--model` joins as judge and evaluee. Names and indices follow
the task's `model_order`, including Python overrides. Direct self-judgment and
normalization follow the run spec.

Direct metric keys are `<member>/eigentrust`, `<member>/elo` and `<member>/rank`.
Pairwise keys include the BTD dimension, as in `btd_d2/<member>/elo`. Rank is
one plus the number of higher scores.

With `training.bootstrap.enabled`, the scorer adds `elo_mean`, `elo_std`,
`elo_ci_lower` and `elo_ci_upper` per member. Intervals use the 2.5th and 97.5th
percentiles. Direct ratings resample scenarios, while pairwise BTD resamples
retained comparison rows. `random_seed` controls resampling only.

If a direct-rating bootstrap draw has multiple stationary distributions, the
scorer sets `bootstrap_unavailable_nonunique_eigentrust=1` and keeps any unique
full-panel ranking. Publication omits bootstrap intervals, samples and plots.

Native `inspect eval` and log rescoring compute metrics in memory. Use
`--no-score` to skip all scorers. In `scripts/run.py`, set
`training.enabled=False` to keep per-judgment diagnostics and skip panel fitting
and bootstrap. A fresh run with training enabled writes analysis files from the
same fit and bootstrap draws as its native metrics. Reused exports and
analysis-only runs score the saved judgments. Rescoring can produce a new fit.
Rescoring incomplete logs requires source-error validation to report rankings.

`coverage/scored` and `coverage/planned` count judgments.
`coverage/scored_fraction` is their ratio. The plan counts every requested
epoch, and unknown planned coverage is NaN. Multi-epoch runs collect judgments
and set `ranking_unsupported_epochs=1` in place of a combined ranking.
Incomplete strict runs set `ranking_incomplete=1` and preserve sample errors.
Provider errors and other non-omittable failures report `ranking_refused_errors`
and suppress rankings and analysis artifacts.

The scoring-only `eigenbench_panel` log links a phased run's source logs and
coverage. Use the judge logs for per-judgment diagnostics. Extension analysis
scores the merged export.

Pairwise metrics count `comparisons_parsed`, `comparisons_retained` and
`comparisons_dropped`. Repeated observations are retained.
`comparisons_ambiguous` counts rows whose reverse presentation could not be
uniquely matched within a draw.
An empty retained set reports `ranking_unavailable_cleaning=1`. A panel member
absent from all retained comparisons reports `ranking_unavailable_model_coverage=1`.
Native fitting supports `btd_ties` with a single trust matrix. Multi-criterion
`separate_criteria=True` has no defined aggregation and fails before fitting.

Offline fixtures check both protocols against baseline `a903a21`, including
ties, self-judgments and bootstrap draws.
