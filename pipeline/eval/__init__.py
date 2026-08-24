"""Evaluation collection and sampling."""

from typing import TYPE_CHECKING

from .samplers import select_sampler
from .collect import (
    collect_core_evaluations,
    collect_planned_evaluations,
    plan_group_assignments,
    sampler_needs_history,
)
from .flows import collect_responses_only
from .criteria_collectors import collect_group_criteria_evaluations

if TYPE_CHECKING:
    from .mixed_collect import collect_mixed_evaluations


_LAZY_EXPORTS = {"collect_mixed_evaluations": ".mixed_collect"}

__all__ = [
    "select_sampler",
    "collect_core_evaluations",
    "collect_planned_evaluations",
    "plan_group_assignments",
    "sampler_needs_history",
    "collect_responses_only",
    "collect_group_criteria_evaluations",
    "collect_mixed_evaluations",
]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        from importlib import import_module

        module = import_module(module_name, __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
