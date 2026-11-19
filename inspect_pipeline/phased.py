"""Collect responses and judgments one model at a time for local panels."""

from __future__ import annotations

import sys
from pathlib import Path

# Inspect puts this directory on sys.path. Prefer the eigenbench package.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[:1] != [_REPO_ROOT]:
    sys.path.insert(0, _REPO_ROOT)

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.viewer import ViewerConfig

from inspect_pipeline.collection.phased import prepared_run_context
from inspect_pipeline.context import _run_context, sanitize
from inspect_pipeline.eigenbench import _edge_sample_id, _samples_view
from inspect_pipeline.generation import ResponsePool
from inspect_pipeline.phases import direct_rating_scorer, direct_rating_solver, response_only_solver


@task
def eigenbench_responses(
    spec: str,
    *,
    model_nick: str,
    models: dict[str, object] | None = None,
    scenario_indices: list[int] | None = None,
) -> Task:
    """One evaluee's responses, in isolation, for ``scenario_indices`` (default: all).

    Phase one of a local run: only this model is needed, so only its server has
    to be resident.
    """

    ctx = prepared_run_context(spec, models)
    if ctx is None:
        ctx = _run_context(spec, models, build_assignments=False)
    wanted = None if scenario_indices is None else {int(s) for s in scenario_indices}
    samples = [
        Sample(
            input=scenario,
            id=f"s{int(s_idx):04d} · {model_nick}",
            metadata={
                "scenario_index": int(s_idx),
                "scenario": scenario,
                "eval_nick": model_nick,
            },
        )
        for s_idx, scenario in ctx["selected"]
        if wanted is None or int(s_idx) in wanted
    ]
    safe = sanitize(model_nick)
    return Task(
        dataset=MemoryDataset(samples=samples, name=f"responses_{safe}"),
        solver=response_only_solver(
            resolve_model=ctx["resolve_model"],
            generation_cfg=ctx["generation"]["response"],
            max_attempts=ctx["max_attempts"],
            cache_enabled=ctx["cache_enabled"],
        ),
        name=f"eigenbench_responses_{safe}",
        display_name=f"EigenBench responses — {model_nick}",
        metadata={"eigenbench_phase": "response", "evaluee": model_nick},
    )


@task
def eigenbench_judge(
    spec: str,
    *,
    judge_nick: str,
    edges: list[dict],
    models: dict[str, object] | None = None,
) -> Task:
    """One judge's reflections and ratings, over responses already generated."""

    ctx = prepared_run_context(spec, models)
    if ctx is None:
        ctx = _run_context(spec, models, build_assignments=False)
    criteria = ctx["criteria"]
    samples = [
        Sample(input=e["scenario"], id=_edge_sample_id(e), metadata=dict(e))
        for e in edges
    ]
    safe = sanitize(judge_nick)
    return Task(
        dataset=MemoryDataset(samples=samples, name=f"judge_{safe}"),
        solver=direct_rating_solver(
            criteria=criteria,
            resolve_model=ctx["resolve_model"],
            response_pool=ResponsePool(),
            generation=ctx["generation"],
            max_attempts=ctx["max_attempts"],
            cache_enabled=ctx["cache_enabled"],
            scale_min=ctx["scale_min"],
            scale_max=ctx["scale_max"],
        ),
        scorer=direct_rating_scorer(),
        viewer=ViewerConfig(task_samples_view=_samples_view(criteria)),
        name=f"eigenbench_judge_{safe}",
        display_name=f"EigenBench judge — {judge_nick}",
        metadata={"eigenbench": {**ctx["meta"], "judge": judge_nick}},
    )
