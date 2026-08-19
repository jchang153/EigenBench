"""Evaluation collection and sampling.

Heavy provider dependencies are imported lazily so prompt parsing, planning,
and trust aggregation can be used without installing OpenRouter or vLLM.
"""

__all__ = [
    "select_sampler",
    "collect_core_evaluations",
    "collect_responses_only",
    "collect_group_criteria_evaluations",
    "collect_mixed_evaluations",
    "collect_direct_ratings",
]


def __getattr__(name):
    if name == "select_sampler":
        from .samplers import select_sampler

        return select_sampler
    if name == "collect_core_evaluations":
        from .collect import collect_core_evaluations

        return collect_core_evaluations
    if name == "collect_responses_only":
        from .flows import collect_responses_only

        return collect_responses_only
    if name == "collect_group_criteria_evaluations":
        from .criteria_collectors import collect_group_criteria_evaluations

        return collect_group_criteria_evaluations
    if name == "collect_mixed_evaluations":
        from .mixed_collect import collect_mixed_evaluations

        return collect_mixed_evaluations
    if name == "collect_direct_ratings":
        from .direct_rating import collect_direct_ratings

        return collect_direct_ratings
    raise AttributeError(name)
