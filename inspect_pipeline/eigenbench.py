"""EigenBench direct rating as a native Inspect AI task.

    inspect eval inspect_pipeline/eigenbench.py -T spec=runs/my_run/spec.py
    inspect eval inspect_pipeline/eigenbench.py   # runs/default/spec.py

One sample per directed judge->evaluee edge. The sampling plan, prompts, and
rating validation come from ``pipeline.protocol.direct_rating``; export the
resulting log to the legacy ``evaluations.jsonl`` with
``scripts/export_evaluations.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Inspect loads this file with its directory on sys.path. Prefer the package over
# the neighboring eigenbench.py, including when the checkout is already on PYTHONPATH.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[:1] != [_REPO_ROOT]:
    sys.path.insert(0, _REPO_ROOT)

from inspect_ai import Task, task

from eigenbench.inspect.context import (
    DEFAULT_MAX_ATTEMPTS, DEFAULT_SPEC, _eval_model, _model_resolver, _run_context,
    _run_context_from_spec, has_local_models, load_selection, resolve_spec_ref, sanitize, with_eval_model,
)
from eigenbench.inspect.direct_rating import (
    _edge_sample_id, _samples_view, build_direct_rating_task, build_edge_samples,
)
from inspect_pipeline.metrics import panel_scorer as panel_scorer
from inspect_pipeline.phases import direct_rating_scorer as direct_rating_scorer

@task
def eigenbench(
    spec: str = DEFAULT_SPEC,
    *,
    models: dict[str, object] | None = None,
    cache: bool | None = None,
    panel_metrics: bool = True,
    write_artifacts: bool = False,
    resolved_sampler_seed: int | None = None,
) -> Task:
    """Direct-rating EigenBench, one sample per directed judge->evaluee edge.

    Args:
        spec: run spec module or path, e.g. ``runs/my_run/spec.py``. Defaults to
            ``runs/default/spec.py``: four inexpensive models, 100 AIRiskDilemmas
            scenarios and the kindness constitution.
        models: optional override of the spec's models (Python callers only).
        cache: override ``collection.inspect.cache``.

    ``--model`` joins the spec's panel as one more evaluee and judge; without it the panel
    is the spec's models alone.
    """

    return build_direct_rating_task(
        spec, models=models, cache=cache, panel_metrics=panel_metrics,
        write_artifacts=write_artifacts, resolved_sampler_seed=resolved_sampler_seed,
    )
