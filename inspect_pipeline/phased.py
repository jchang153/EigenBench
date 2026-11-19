"""Native Inspect entry points for phased direct runs."""

from __future__ import annotations

import sys
from pathlib import Path

# Inspect loads this file with its directory on sys.path. Prefer the package over
# the neighboring eigenbench.py, including when the checkout is already on PYTHONPATH.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[:1] != [_REPO_ROOT]:
    sys.path.insert(0, _REPO_ROOT)

from inspect_ai import Task, task

from eigenbench.inspect.direct_rating import build_judge_task, build_response_task
from inspect_pipeline.phases import direct_rating_scorer as direct_rating_scorer


@task
def eigenbench_responses(
    spec: str,
    *,
    model_nick: str,
    models: dict[str, object] | None = None,
    scenario_indices: list[int] | None = None,
) -> Task:
    """Generate one model's responses for the selected scenarios."""
    return build_response_task(spec, model_nick=model_nick, models=models, scenario_indices=scenario_indices)


@task
def eigenbench_judge(
    spec: str,
    *,
    judge_nick: str,
    edges: list[dict],
    models: dict[str, object] | None = None,
) -> Task:
    """Collect one judge's ratings over responses already generated."""
    return build_judge_task(spec, judge_nick=judge_nick, edges=edges, models=models)
