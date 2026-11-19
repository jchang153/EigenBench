"""Inspect registrations for existing generation and direct-rating log names."""

from __future__ import annotations

from typing import Callable

from inspect_ai.model import Model
from inspect_ai.scorer import mean, scorer
from inspect_ai.solver import Solver, solver

from eigenbench.inspect.direct_rating import (
    STORE_JUDGMENT_RAW, STORE_RATINGS, STORE_REFLECTION, criterion_key, criterion_label,
    direct_rating_scorer as _direct_rating_scorer,
    direct_rating_solver as _direct_rating_solver,
)
from eigenbench.inspect.generation import (
    FAILURE_POLICIES, RESPONSE_SYSTEM_MESSAGE, STORE_RESPONSE, InvalidJudgmentError,
    ResponsePool, generate_validated, phase_config,
    response_only_solver as _response_only_solver,
)


@solver
def response_only_solver(
    *,
    resolve_model: Callable[[str], Model],
    generation_cfg: dict,
    max_attempts: int,
    cache_enabled: bool,
    cache_scope: Callable[[dict], str] | None = None,
) -> Solver:
    return _response_only_solver(
        resolve_model=resolve_model,
        generation_cfg=generation_cfg,
        max_attempts=max_attempts,
        cache_enabled=cache_enabled,
        cache_scope=cache_scope,
    )


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
    return _direct_rating_solver(
        criteria=criteria,
        resolve_model=resolve_model,
        response_pool=response_pool,
        generation=generation,
        max_attempts=max_attempts,
        cache_enabled=cache_enabled,
        scale_min=scale_min,
        scale_max=scale_max,
    )


@scorer(metrics=[mean()])
def direct_rating_scorer():
    return _direct_rating_scorer()
