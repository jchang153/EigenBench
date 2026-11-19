"""Inspect task factories, solver and scorer for direct ratings."""

from __future__ import annotations

from typing import Callable

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, Model
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import Generate, Solver, TaskState
from inspect_ai.viewer import TaskSamplesColumn, TaskSamplesSort, TaskSamplesView, ViewerConfig

from eigenbench.protocol import RESPONSE_SYSTEM_MESSAGE
from eigenbench.protocol.direct_rating import (
    build_direct_rating_prompt,
    build_direct_rating_user_prompt,
    build_direct_reflection_prompt,
    build_direct_reflection_user_prompt,
    direct_rating_validator,
    parse_direct_ratings,
)

from .collection.phased import prepared_run_context
from .context import DEFAULT_SPEC, _run_context, sanitize, with_eval_model
from .generation import (
    STORE_RESPONSE, InvalidJudgmentError, ResponsePool, generate_validated, phase_config,
)
from .models import check_distinct_local_servers

STORE_REFLECTION = "eigenbench:reflection"
STORE_JUDGMENT_RAW = "eigenbench:judgment_raw"
STORE_RATINGS = "eigenbench:ratings"


def direct_rating_solver(
    *,
    criteria: list[str],
    resolve_model: Callable[[str], Model],
    response_pool: ResponsePool,
    generation: dict,
    max_attempts: int,
    cache_enabled: bool,
    scale_min: int = 1,
    scale_max: int = 10,
) -> Solver:
    criteria_text = "\n".join(criteria)
    reflection_system = build_direct_reflection_prompt()
    rating_system = build_direct_rating_prompt()
    validator = direct_rating_validator(len(criteria), scale_min, scale_max)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        md = state.metadata
        scenario = md["scenario"]
        s_idx = int(md["scenario_index"])
        judge_nick = md["judge_nick"]
        eval_nick = md["eval_nick"]
        identity = f"scenario_index={s_idx} judge={judge_nick} evaluee={eval_nick}"

        response_messages = [
            ChatMessageSystem(content=RESPONSE_SYSTEM_MESSAGE),
            ChatMessageUser(content=scenario),
        ]

        async def make_response() -> str:
            output = await generate_validated(
                resolve_model(eval_nick),
                response_messages,
                config=phase_config(generation["response"], eval_nick),
                max_attempts=max_attempts,
                cache_enabled=cache_enabled,
                validator=None,
                identity=f"response scenario_index={s_idx} evaluee={eval_nick}",
            )
            return output.completion

        supplied = md.get("response")
        if isinstance(supplied, str) and supplied.strip():
            response = supplied
        else:
            response = await response_pool.get((s_idx, eval_nick), make_response)

        judge = resolve_model(judge_nick)
        reflection_messages = [
            ChatMessageSystem(content=reflection_system),
            ChatMessageUser(
                content=build_direct_reflection_user_prompt(
                    criteria_text, scenario, response
                )
            ),
        ]
        reflection_output = await generate_validated(
            judge,
            reflection_messages,
            config=phase_config(generation["reflection"], judge_nick),
            max_attempts=max_attempts,
            cache_enabled=cache_enabled,
            validator=None,
            identity=f"reflection {identity}",
            exhausted=InvalidJudgmentError,
        )
        reflection = reflection_output.completion

        rating_messages = [
            ChatMessageSystem(content=rating_system),
            ChatMessageUser(
                content=build_direct_rating_user_prompt(
                    criteria_text, scenario, response, reflection
                )
            ),
        ]
        rating_output = await generate_validated(
            judge,
            rating_messages,
            config=phase_config(generation["direct_rating"], judge_nick),
            max_attempts=max_attempts,
            cache_enabled=cache_enabled,
            validator=validator,
            identity=f"direct_rating {identity}",
            exhausted=InvalidJudgmentError,
        )
        raw = rating_output.completion
        parsed = parse_direct_ratings(
            raw, num_criteria=len(criteria), scale_min=scale_min, scale_max=scale_max
        )

        state.store.set(STORE_RESPONSE, response)
        state.store.set(STORE_REFLECTION, reflection)
        state.store.set(STORE_JUDGMENT_RAW, raw)
        state.store.set(
            STORE_RATINGS,
            [
                {"criterion_index": index, "criterion": criteria[index], "rating": value}
                for index, value in parsed.items()
            ],
        )
        state.messages = (
            response_messages
            + reflection_messages
            + [reflection_output.message]
            + rating_messages
            + [rating_output.message]
        )
        state.output = rating_output
        return state

    return solve


def criterion_key(index: int) -> str:
    """Stable score key for a criterion, ordered for column display."""

    return f"c{index + 1:02d}"


def criterion_label(index: int, criterion: str) -> str:
    """A criterion's text trimmed to something that fits a column header."""

    text = criterion.split(":", 1)[-1].strip()
    for prefix in ("prefer the response that ", "prefer the response "):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.strip()
    if len(text) > 42:
        text = text[:41].rstrip() + "…"
    return f"{index + 1}. {text}" if text else f"Criterion {index + 1}"


def direct_rating_scorer():
    """One score per criterion, plus their mean.

    Scoring per criterion rather than as a single number is what lets the
    viewer lay a judgment out as a row of criterion cells instead of one
    opaque average.
    """

    async def score(state: TaskState, target: Target) -> Score:
        ratings = state.store.get(STORE_RATINGS) or []
        values = [entry["rating"] for entry in ratings]
        value: dict[str, float] = {
            criterion_key(entry["criterion_index"]): entry["rating"] for entry in ratings
        }
        value["mean"] = round(sum(values) / len(values), 2) if values else float("nan")
        md = state.metadata
        return Score(
            value=value,
            answer=f"{md.get('judge_nick')} → {md.get('eval_nick')}",
            explanation=state.store.get(STORE_JUDGMENT_RAW),
            metadata={"ratings": ratings},
        )

    return score


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


def build_direct_rating_task(
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
    from eigenbench.inspect.provenance import direct_resume_identity
    from inspect_pipeline.metrics import panel_scorer
    from inspect_pipeline.phases import direct_rating_scorer, direct_rating_solver

    from .metrics import analysis_config
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


def build_response_task(
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
    from inspect_pipeline.phases import response_only_solver

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


def build_judge_task(
    spec: str,
    *,
    judge_nick: str,
    edges: list[dict],
    models: dict[str, object] | None = None,
) -> Task:
    """One judge's reflections and ratings, over responses already generated."""
    from inspect_pipeline.phases import direct_rating_scorer, direct_rating_solver

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
