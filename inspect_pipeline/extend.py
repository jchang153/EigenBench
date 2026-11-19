"""Native Inspect entry point for extending a saved EigenBench run."""

from __future__ import annotations

import sys
from pathlib import Path

# Inspect loads this file with its directory on sys.path. Prefer the package over
# the neighboring eigenbench.py, including when the checkout is already on PYTHONPATH.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[:1] != [_REPO_ROOT]:
    sys.path.insert(0, _REPO_ROOT)

from inspect_ai import Task, task

from eigenbench.inspect.collection.extension import (
    _use_phased_extension, build_extension_task, check_spec_covers, edge_samples, eigenbench_extend,
    extend_run,
)
from eigenbench.protocol.extension import (
    DIRECT, PAIRWISE, Edge, Plan, count_matrix, detect_mode, model_names, plan_addition,
    plan_additions, projected_counts, response_index, scenarios_of,
)
from inspect_pipeline.phases import direct_rating_scorer as direct_rating_scorer


@task
def extend_model(
    spec: str, *, new_model: str | None = None, model_id: str | None = None,
    additional_scenarios: int | None = None,
) -> Task:
    """Collect an extension spec, with optional legacy single-model arguments."""
    return build_extension_task(
        spec, new_model=new_model, model_id=model_id, additional_scenarios=additional_scenarios,
    )
