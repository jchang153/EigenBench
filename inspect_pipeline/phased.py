"""Phased execution for runs that cannot hold every model in memory at once.

A judgment needs both its evaluee and its judge, so the edge-per-sample task in
``eigenbench.py`` keeps every model live for the whole run. That is free for
hosted models and fatal on a single GPU, so local runs split the work the way
the legacy collector did: every response first, then every judgment, one model
resident at a time.

These tasks are driven by ``inspect_pipeline.collect`` rather than the CLI, so
they live outside the entrypoint module -- ``inspect eval eigenbench.py`` runs
every ``@task`` it finds in a file.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.viewer import ViewerConfig

from inspect_pipeline.eigenbench import (
    _edge_sample_id,
    _run_context,
    _samples_view,
    sanitize,
)
from inspect_pipeline.phases import (
    ResponsePool,
    direct_rating_scorer,
    direct_rating_solver,
    response_only_solver,
)


@task
def eigenbench_responses(
    spec: str,
    *,
    model_nick: str,
    models: dict[str, object] | None = None,
) -> Task:
    """One evaluee's responses, in isolation.

    Phase one of a local run: only this model is needed, so only its server has
    to be resident.
    """

    ctx = _run_context(spec, models)
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

    ctx = _run_context(spec, models)
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
