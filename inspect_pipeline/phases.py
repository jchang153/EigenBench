"""Inspect solvers and scorer for generating responses and direct ratings."""

from __future__ import annotations

from typing import Callable

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, Model
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from pipeline.protocol import RESPONSE_SYSTEM_MESSAGE
from pipeline.protocol.direct_rating import (
    build_direct_rating_prompt,
    build_direct_rating_user_prompt,
    build_direct_reflection_prompt,
    build_direct_reflection_user_prompt,
    direct_rating_validator,
    parse_direct_ratings,
)

from .generation import STORE_RESPONSE, InvalidJudgmentError, ResponsePool, generate_validated, phase_config

STORE_REFLECTION = "eigenbench:reflection"
STORE_JUDGMENT_RAW = "eigenbench:judgment_raw"
STORE_RATINGS = "eigenbench:ratings"


@solver
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


@solver
def response_only_solver(
    *,
    resolve_model: Callable[[str], Model],
    generation_cfg: dict,
    max_attempts: int,
    cache_enabled: bool,
    cache_scope: Callable[[dict], str] | None = None,
) -> Solver:
    """Generate one evaluee response per sample, and nothing else."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        md = state.metadata
        s_idx = int(md["scenario_index"])
        eval_nick = md["eval_nick"]
        config = phase_config(generation_cfg, eval_nick)
        messages = [
            ChatMessageSystem(content=RESPONSE_SYSTEM_MESSAGE),
            ChatMessageUser(content=md["scenario"]),
        ]
        output = await generate_validated(
            resolve_model(eval_nick),
            messages,
            config=config,
            max_attempts=max_attempts,
            cache_enabled=cache_enabled,
            validator=None,
            identity=f"response scenario_index={s_idx} evaluee={eval_nick}",
            cache_scope=cache_scope(md) if cache_scope else None,
        )
        state.messages = messages + [output.message]
        state.output = output
        state.store.set(STORE_RESPONSE, output.completion)
        return state

    return solve


@scorer(metrics=[mean()])
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
