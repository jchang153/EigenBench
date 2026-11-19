"""Collect the additional judgments in an EigenBench extension spec."""

from __future__ import annotations

import sys
from pathlib import Path

# Inspect loads task files outside their package.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inspect_ai import Task, task

from inspect_pipeline.collection.extension import _extension_task, _prepare_extension, _use_phased_extension
from inspect_pipeline.models import check_distinct_local_servers


@task
def extend_model(
    spec: str, *, new_model: str | None = None, model_id: str | None = None,
    additional_scenarios: int | None = None,
) -> Task:
    """Collect an extension spec, with optional legacy single-model arguments."""
    if (new_model is None) != (model_id is None):
        raise ValueError("new_model and model_id must be supplied together")
    additions = {new_model: model_id} if new_model is not None else None
    prepared = _prepare_extension(
        spec, additions, additional_scenarios=additional_scenarios,
    )
    if _use_phased_extension(prepared):
        raise ValueError(
            "this extension requires phased execution; use scripts/extend_run.py "
            "instead of the native Inspect task"
        )
    check_distinct_local_servers(prepared["models"])
    return _extension_task(prepared)
