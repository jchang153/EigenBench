"""EigenBench direct rating as a native Inspect AI task.

    inspect eval inspect_pipeline/eigenbench.py -T spec=runs/my_run/spec.py
    inspect eval inspect_pipeline/eigenbench.py   # runs/default/spec.py

Each sample is one judge rating one response. Export the log to
``evaluations.jsonl`` with ``scripts/export_evaluations.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Inspect puts this directory on sys.path. Prefer the eigenbench package.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[:1] != [_REPO_ROOT]:
    sys.path.insert(0, _REPO_ROOT)

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.viewer import TaskSamplesColumn, TaskSamplesSort, TaskSamplesView, ViewerConfig

from inspect_pipeline.collection.phased import prepared_run_context
from inspect_pipeline.context import DEFAULT_SPEC, _run_context, with_eval_model
from inspect_pipeline.generation import ResponsePool
from inspect_pipeline.metrics import analysis_config, panel_scorer
from inspect_pipeline.models import check_distinct_local_servers
from inspect_pipeline.phases import criterion_key, criterion_label, direct_rating_scorer, direct_rating_solver
from inspect_pipeline.provenance import direct_resume_identity


def build_edge_samples(assignments: list[dict]) -> list[Sample]:
    """Expand assignments into one sample per directed edge.

    ``edge_index`` preserves legacy record ordering through the log.
    """

    samples: list[Sample] = []
    for assignment in assignments:
        s_idx = int(assignment["scenario_index"])
        for eval_idx, eval_nick in zip(
            assignment["eval_idxs"], assignment["eval_nicks"]
        ):
            samples.append(
                Sample(
                    input=assignment["scenario"],
                    id=_edge_sample_id(
                        {
                            "scenario_index": s_idx,
                            "sampling_round": assignment.get("sampling_round", 0),
                            "judge_nick": assignment["judge_nick"],
                            "eval_nick": eval_nick,
                        }
                    ),
                    metadata={
                        "edge_index": len(samples),
                        "scenario_index": s_idx,
                        "scenario": assignment["scenario"],
                        "judge_idx": int(assignment["judge_idx"]),
                        "judge_nick": assignment["judge_nick"],
                        "eval_idx": int(eval_idx),
                        "eval_nick": eval_nick,
                        "sampler_mode": assignment.get("sampler_mode"),
                        "sampling_round": int(assignment.get("sampling_round", 0)),
                        "group_index": int(assignment.get("group_index", 0)),
                    },
                )
            )
    return samples


def _edge_sample_id(edge: dict) -> str:
    return (
        f"s{int(edge['scenario_index']):04d} r{int(edge.get('sampling_round', 0))} · "
        f"{edge['judge_nick']} → {edge['eval_nick']}"
    )


def _samples_view(criteria: list[str]) -> TaskSamplesView:
    scorer_name = "direct_rating_scorer"
    return TaskSamplesView(
        name="Judgments",
        columns=[
            TaskSamplesColumn(id="sampleId"),
            TaskSamplesColumn(id="answer"),
            TaskSamplesColumn.score(scorer_name, "mean"),
            *[
                TaskSamplesColumn.score(scorer_name, criterion_key(i))
                for i in range(len(criteria))
            ],
            TaskSamplesColumn(id="input", visible=False),
            TaskSamplesColumn(id="tokens", visible=False),
        ],
        # Weakest judgments first: the point of reading these is finding where a
        # judge broke from the pack, not admiring the middle of the scale.
        sort=[TaskSamplesSort.score(scorer_name, "mean", dir="asc")],
        multiline=False,
        compact_scores=True,
        color_scales_enabled=True,
        score_labels={
            "mean": "Mean",
            **{criterion_key(i): criterion_label(i, c) for i, c in enumerate(criteria)},
        },
        score_color_scales={
            "mean": "good-high",
            **{criterion_key(i): "good-high" for i in range(len(criteria))},
        },
    )


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

    ctx = prepared_run_context(spec, models)
    if ctx is None:
        ctx = _run_context(spec, with_eval_model(spec, models), cache,
                           resolved_sampler_seed=resolved_sampler_seed)
    elif cache is not None:
        ctx = {**ctx, "cache_enabled": bool(cache),
               "meta": {**ctx["meta"], "cache_enabled": bool(cache)}}
    if not ctx["is_direct"]:
        raise ValueError(
            "inspect_pipeline.eigenbench supports evaluation.mode='direct_rating' "
            "only; pairwise runs use inspect_pipeline.pairwise."
        )
    # One task serves the whole panel concurrently.
    check_distinct_local_servers(ctx["models"])
    criteria = ctx["criteria"]
    samples = build_edge_samples(ctx["assignments"])
    config = analysis_config(ctx, planned=len(samples), write_artifacts=write_artifacts) if panel_metrics else None
    return Task(
        dataset=MemoryDataset(
            samples=samples,
            name=f"eigenbench_{ctx['run_spec']['name']}",
        ),
        solver=direct_rating_solver(
            criteria=criteria,
            resolve_model=ctx["resolve_model"],
            response_pool=ResponsePool(seed=ctx["seed"]),
            generation=ctx["generation"],
            max_attempts=ctx["max_attempts"],
            cache_enabled=ctx["cache_enabled"],
            scale_min=ctx["scale_min"],
            scale_max=ctx["scale_max"],
        ),
        scorer=([direct_rating_scorer(), panel_scorer(config)] if config is not None
                else direct_rating_scorer()),
        viewer=ViewerConfig(task_samples_view=_samples_view(criteria)),
        name=f"eigenbench_{ctx['run_spec']['name']}",
        display_name=f"EigenBench direct rating — {ctx['run_spec']['name']}",
        metadata={"eigenbench": {**ctx["meta"], "resume_identity": direct_resume_identity(ctx),
                                 **({"analysis": config} if config is not None else {})}},
        # Omitting invalid judgments needs the eval to survive a failed sample; export still vets each.
        fail_on_error=(False if ctx["meta"]["failure_policy"] == "omit_invalid_judgments" else None),
    )
