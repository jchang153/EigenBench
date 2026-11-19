"""Jonathan's grouped pairwise protocol as a native Inspect task.

    inspect eval inspect_pipeline/pairwise.py -T spec=runs/my_run/spec.py

One sample per ordered comparison. Responses and reflections are shared within
sampled groups. See docs/inspect-pairwise-protocol.md for reference behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Inspect loads this file with its directory on sys.path. Prefer the package over
# the neighboring eigenbench.py, including when the checkout is already on PYTHONPATH.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[:1] != [_REPO_ROOT]:
    sys.path.insert(0, _REPO_ROOT)

from inspect_ai import Task, task
from inspect_ai.model import Model
from inspect_ai.scorer import mean, scorer
from inspect_ai.solver import Solver, solver

from eigenbench.inspect.pairwise import (
    CHOICE_SCORE_KEYS, STORE_EVAL1_REFLECTION, STORE_EVAL1_RESPONSE, STORE_EVAL2_REFLECTION,
    STORE_EVAL2_RESPONSE, STORE_JUDGE_RESPONSE, build_comparison_user, build_pairwise_eval_task,
    build_pairwise_task, build_reflection_user, eigenbench_pairwise_extend, pairwise_cache_scope,
    pairwise_choice_scorer as _pairwise_choice_scorer, pairwise_comparison_samples, pairwise_edge_samples,
    pairwise_solver as _pairwise_solver,
    pairwise_record, parse_pairwise_choices, planned_pairwise_comparisons, records_from_pairwise_log,
)
from inspect_pipeline.metrics import panel_scorer as panel_scorer


@task
def eigenbench_pairwise(
    spec: str,
    *,
    models: dict[str, object] | None = None,
    cache: bool | None = None,
    write_artifacts: bool = False,
) -> Task:
    """The original pairwise EigenBench protocol, one sample per ordered comparison.

    Args:
        spec: run spec module or path with ``evaluation.mode='pairwise_btd'``.
        models: optional override of the spec's models (Python callers only).
        cache: override ``collection.inspect.cache``.

    ``--model`` joins the panel as one more judge and evaluee.
    """

    return build_pairwise_eval_task(spec, models=models, cache=cache, write_artifacts=write_artifacts)


@solver
def pairwise_solver(
    *,
    criteria: list[str],
    resolve_model: Callable[[str], Model],
    generation: dict,
    max_attempts: int,
    cache_enabled: bool,
    allow_ties: bool = True,
) -> Solver:
    return _pairwise_solver(
        criteria=criteria,
        resolve_model=resolve_model,
        generation=generation,
        max_attempts=max_attempts,
        cache_enabled=cache_enabled,
        allow_ties=allow_ties,
    )


@scorer(metrics={key: [mean()] for key in CHOICE_SCORE_KEYS})
def pairwise_choice_scorer(num_criteria: int):
    return _pairwise_choice_scorer(
        num_criteria=num_criteria,
    )
