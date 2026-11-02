# Native EigenBench metrics

Fresh direct-rating and pairwise Inspect tasks include a `panel_scorer` alongside
per-judgment diagnostics. Its native metrics report each effective panel member's
EigenTrust, display Elo and rank. `--model` retains its existing role as both judge
and evaluee, joining the configured panel when it is not already present. Names
and indices follow the task's effective `model_order`, including Python overrides.
Direct self-judgment and normalization settings come from the run spec.

Direct metric keys are `<member>/eigentrust`, `<member>/elo` and `<member>/rank`.
Pairwise keys add the configured dimension, for example `btd_d2/<member>/elo`.
Rank is one plus the number of strictly higher scores, so equal scores share a
rank. These are panel-relative results, not a separate evaluation of only the
Inspect `--model` target.

When `training.bootstrap.enabled` is true, the same metric also exposes
`elo_mean`, `elo_std`, `elo_ci_lower` and `elo_ci_upper` per member. The intervals
are the existing 2.5th/97.5th percentiles. Direct ratings resample whole scenarios;
pairwise BTD resamples comparison rows after legacy cleaning. Configured bootstrap
counts and `random_seed` are unchanged. That seed controls bootstrap resampling;
it does **not** seed PyTorch initialization or training-loader shuffling. The
existing training split, ties, EigenTrust damping and Elo conversion are unchanged.

Native `inspect eval` and public log re-scoring compute metrics without writing
analysis artifacts. Inspect's normal `--no-score` option skips scoring entirely.
The canonical runner preserves `training.enabled=False` as collection-only,
retaining per-judgment diagnostics but skipping panel fitting and bootstrap.
With training enabled, the runner explicitly requests the existing analysis files
from the same fitted result and bootstrap replicates as its native metrics.
Repeated Inspect progress callbacks reuse that task's completed-panel result,
while incomplete progress updates do not fit the panel. Fresh collection reports
explicit artifact completion to the runner, which skips its second analysis
stage. Reused exports and analysis-only runs retain the ordinary analysis stage.
A new re-score may produce a new stochastic fit, but does not overwrite artifacts.
Incomplete logs re-scored without source-error validation report unavailable
rankings rather than assuming their omitted samples were permissible.

`coverage/scored` counts exported judgment records, `coverage/planned` counts
planned judgments, and `coverage/scored_fraction` divides those counts. This is
the collection report's judgment unit, not the number of criterion comparisons.
Inspect's resolved epoch count multiplies the plan, including when an epoch has
failed samples. Unknown planned coverage is NaN. Multi-epoch collection remains
supported, but combined panel rankings are unavailable with
`ranking_unsupported_epochs=1`; no new aggregation of independent epoch draws is
introduced. Incomplete strict runs report `ranking_incomplete=1` and preserve the
original sample errors. Provider errors and other non-omittable failures report
`ranking_refused_errors` and never produce rankings or analysis artifacts.

Phased collection creates one scoring-only `eigenbench_panel` log after combining
all judge records. This log makes no model calls, is the manifest's primary log,
and links its source logs and collection coverage. Its explicit `panel_metrics`
role and record-only samples prevent double counting in `collection_report.json`.
It cannot be re-exported as new judgments. Individual judge logs retain their
per-judgment diagnostics. Extensions retain their existing merged-export analysis
stage because historical source exports may not have complete Inspect provenance.

Pairwise metrics expose `comparisons_parsed`, `comparisons_retained` and
`comparisons_dropped`. The legacy cleaner can discard repeated observations
(issue #41). Its behavior is preserved. If no rows survive, rankings and artifacts
are unavailable (`ranking_available=0`, `ranking_unavailable_cleaning=1`) and a
warning identifies the cause. Multi-criterion `separate_criteria=True` fails
before fitting because no approved square trust-matrix aggregation
exists. Native fitting supports `btd_ties`; the unresolved legacy BT adapter is
rejected explicitly.

Offline regression fixtures freeze baseline `a903a21` analysis for both protocols,
including ties, self-judgments and bootstrap draws. CPU tolerances are `1e-10` for
direct bootstrap summaries, `1e-7` for BTD trust and `1e-4` for BTD Elo. These small
synthetic fixtures establish compatibility, not estimator quality.
