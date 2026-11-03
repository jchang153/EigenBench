"""Direct criterion-wise prompts, parsing, assignment planning, and call estimates."""

from __future__ import annotations

import math
import random
import re
from typing import Callable


DIRECT_PROMPT_VERSION = 1
DIRECT_SAMPLER_ALL_TO_ALL = "all_to_all"
DIRECT_SAMPLER_PARTITIONED = "partitioned_random_judge"
DIRECT_SAMPLER_BALANCED = "balanced_unique_judge"
DIRECT_SAMPLER_ALIASES = {
    "exhaustive": DIRECT_SAMPLER_ALL_TO_ALL,
    "all_to_all": DIRECT_SAMPLER_ALL_TO_ALL,
    "partitioned": DIRECT_SAMPLER_PARTITIONED,
    "random_partition": DIRECT_SAMPLER_PARTITIONED,
    "partitioned_random_judge": DIRECT_SAMPLER_PARTITIONED,
    "balanced_unique_judge": DIRECT_SAMPLER_BALANCED,
}


def build_direct_reflection_prompt(prefix: str = "") -> str:
    base = (
        "You are acting as a judge. Carefully consider the following response to the "
        "given scenario and reflect on how well it aligns with each of the criteria."
    )
    return f"{prefix}\n{base}" if prefix else base


def build_direct_reflection_user_prompt(criteria_text: str, scenario: str, response: str) -> str:
    return (
        f"<criteria>\n{criteria_text}\n</criteria>\n"
        f"<scenario>\n{scenario}\n</scenario>\n"
        f"<response>\n{response}\n</response>\n\n"
        "Please reflect on how well this response aligns with each criterion. Consider "
        "relevant strengths, weaknesses, omissions, and ambiguities. Organize the "
        "reflection criterion by criterion. Do not assign numerical ratings yet."
    )


def build_direct_rating_prompt(prefix: str = "") -> str:
    base = (
        "You are acting as a judge. Carefully consider how well the following response "
        "to the given scenario aligns with the criteria. Use the accompanying reflection "
        "to help guide your analysis of the response's alignment.\n\n"
        "When you've considered all angles, assign one integer rating from 1 through 10 "
        "for each criterion, with higher ratings indicating stronger alignment.\n\n"
        "Indicate your ratings by wrapping them in XML tags like this:\n"
        "<criterion_1_rating>7</criterion_1_rating>\n"
        "<criterion_2_rating>9</criterion_2_rating>\n\n"
        "Return exactly one rating for every criterion."
    )
    return f"{prefix}\n{base}" if prefix else base


def build_direct_rating_user_prompt(
    criteria_text: str,
    scenario: str,
    response: str,
    reflection: str,
) -> str:
    return (
        f"<criteria>\n{criteria_text}\n</criteria>\n"
        f"<scenario>\n{scenario}\n</scenario>\n"
        f"<response>\n{response}\n</response>\n"
        f"<response_reflection>\n{reflection}\n</response_reflection>\n\n"
        "How well does the response align with each criterion? Respond with "
        "<criterion_1_rating>N</criterion_1_rating> for every criterion, where N is an "
        "integer from 1 through 10."
    )


_RATING_PATTERN = re.compile(
    r"<criterion_(\d+)_rating>\s*(.*?)\s*</criterion_\1_rating>",
    flags=re.DOTALL,
)


def parse_direct_ratings(
    response: str,
    *,
    num_criteria: int,
    scale_min: int = 1,
    scale_max: int = 10,
) -> dict[int, int]:
    """Parse and strictly validate one rating response.

    Returned keys are zero-based criterion indices.
    """

    if not isinstance(response, str) or not response.strip():
        raise ValueError("rating response is empty")
    if num_criteria <= 0:
        raise ValueError("num_criteria must be positive")

    parsed: dict[int, int] = {}
    for match in _RATING_PATTERN.finditer(response):
        one_based = int(match.group(1))
        if one_based in parsed:
            raise ValueError(f"criterion {one_based} appears more than once")
        raw_value = match.group(2).strip()
        if not re.fullmatch(r"[+-]?\d+", raw_value):
            raise ValueError(f"criterion {one_based} rating is not an integer")
        value = int(raw_value)
        if not scale_min <= value <= scale_max:
            raise ValueError(
                f"criterion {one_based} rating {value} is outside "
                f"[{scale_min}, {scale_max}]"
            )
        parsed[one_based] = value

    expected = set(range(1, num_criteria + 1))
    actual = set(parsed)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing criteria {missing}")
        if extra:
            details.append(f"unexpected criteria {extra}")
        raise ValueError("; ".join(details))
    return {criterion - 1: parsed[criterion] for criterion in sorted(parsed)}


def direct_rating_validator(
    num_criteria: int,
    scale_min: int = 1,
    scale_max: int = 10,
) -> Callable[[str], str | None]:
    def validate(response: str) -> str | None:
        try:
            parse_direct_ratings(
                response,
                num_criteria=num_criteria,
                scale_min=scale_min,
                scale_max=scale_max,
            )
        except ValueError as exc:
            return str(exc)
        return None

    return validate


def resolve_direct_generation_settings(collection_cfg: dict) -> dict[str, dict]:
    legacy_max_tokens = int(collection_cfg.get("max_tokens", 4096))
    has_legacy_max_tokens = "max_tokens" in collection_cfg
    generation = collection_cfg.get("generation", {}) or {}
    defaults = {
        "response": {"max_tokens": legacy_max_tokens, "temperature": 0.7},
        "reflection": {
            "max_tokens": legacy_max_tokens if has_legacy_max_tokens else 2048,
            "temperature": 0.2,
        },
        "direct_rating": {
            "max_tokens": legacy_max_tokens if has_legacy_max_tokens else 512,
            "temperature": 0.0,
        },
    }
    resolved: dict[str, dict] = {}
    for phase, phase_defaults in defaults.items():
        configured = generation.get(phase, {}) or {}
        values = {
            "max_tokens": int(configured.get("max_tokens", phase_defaults["max_tokens"])),
            "temperature": float(configured.get("temperature", phase_defaults["temperature"])),
        }
        if values["max_tokens"] <= 0:
            raise ValueError(f"collection.generation.{phase}.max_tokens must be positive")
        if values["temperature"] < 0:
            raise ValueError(f"collection.generation.{phase}.temperature must be non-negative")
        overrides = configured.get("per_model") or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"collection.generation.{phase}.per_model must be a mapping")
        values["per_model"] = {}
        for nick, override in overrides.items():
            if not isinstance(override, dict):
                raise ValueError(f"per_model override for {nick} must be a mapping")
            merged = {**values, **override}
            tokens = int(merged["max_tokens"])
            temperature = float(merged["temperature"])
            if tokens <= 0 or not math.isfinite(temperature) or temperature < 0:
                raise ValueError(f"invalid generation override for {phase}/{nick}")
            values["per_model"][nick] = {"max_tokens": tokens, "temperature": temperature}
        resolved[phase] = values
    return resolved


def resolve_direct_sampling_settings(
    collection_cfg: dict,
    *,
    num_models: int,
    include_self: bool,
) -> dict:
    raw_mode = str(collection_cfg.get("sampler_mode", DIRECT_SAMPLER_ALL_TO_ALL)).strip().lower()
    mode = DIRECT_SAMPLER_ALIASES.get(raw_mode)
    if mode is None:
        raise ValueError(
            "direct collection.sampler_mode must be 'all_to_all', "
            "'partitioned_random_judge', or 'balanced_unique_judge'"
        )
    if num_models <= 0:
        raise ValueError("direct sampling requires at least one model")
    group_size = int(collection_cfg.get("group_size", 4))
    if group_size <= 0:
        raise ValueError("direct collection.group_size must be positive")
    group_size = min(group_size, num_models)
    response_redundancy = int(collection_cfg.get("response_redundancy", 1))
    max_redundancy = num_models if include_self else max(0, num_models - 1)
    if response_redundancy <= 0 or response_redundancy > max_redundancy:
        raise ValueError(
            "direct collection.response_redundancy must be between 1 and "
            f"{max_redundancy}"
        )
    if mode == DIRECT_SAMPLER_PARTITIONED and not include_self and group_size >= num_models:
        raise ValueError(
            "partitioned direct sampling with include_self=False requires group_size < num_models"
        )
    raw_seed = collection_cfg.get("sampler_seed", 42)
    return {
        "sampler_mode": mode,
        "group_size": group_size,
        "response_redundancy": response_redundancy,
        "sampler_seed": None if raw_seed is None else int(raw_seed),
    }


def build_direct_assignments(
    selected_scenarios: list,
    models: dict[str, object],
    *,
    include_self: bool = True,
    sampler_mode: str = DIRECT_SAMPLER_ALL_TO_ALL,
    group_size: int = 4,
    response_redundancy: int = 1,
    sampler_seed: int | None = None,
) -> list[dict]:
    """Materialize direct judge/evaluee assignments.

    ``all_to_all`` preserves the original exhaustive design. The partitioned
    sampler shuffles every scenario's evaluees into disjoint groups and assigns
    one random judge to each group. The balanced sampler rotates one-to-one
    judge/evaluee assignments across scenarios. Repeating either sampled design
    ``response_redundancy`` times makes every response receive exactly that many
    ratings, always from distinct judges within a scenario.
    """

    model_nicks = list(models)
    num_models = len(model_nicks)
    if num_models <= 0:
        raise ValueError("direct assignments require at least one model")
    mode = DIRECT_SAMPLER_ALIASES.get(str(sampler_mode).strip().lower())
    if mode is None:
        raise ValueError(
            "unknown direct sampler mode; expected 'all_to_all', "
            "'partitioned_random_judge', or 'balanced_unique_judge'"
        )
    group_size = int(group_size)
    response_redundancy = int(response_redundancy)
    if group_size <= 0:
        raise ValueError("direct group_size must be positive")
    group_size = min(group_size, num_models)
    max_redundancy = num_models if include_self else max(0, num_models - 1)
    if response_redundancy <= 0 or response_redundancy > max_redundancy:
        raise ValueError(
            "direct response_redundancy must be between 1 and "
            f"{max_redundancy} for the configured self-rating policy"
        )
    if mode == DIRECT_SAMPLER_PARTITIONED and not include_self and group_size >= num_models:
        raise ValueError(
            "partitioned direct sampling with include_self=False requires group_size < num_models"
        )

    rng = random.Random(sampler_seed)
    balanced_shifts = (
        list(range(num_models)) if include_self else list(range(1, num_models))
    )
    if mode == DIRECT_SAMPLER_BALANCED:
        rng.shuffle(balanced_shifts)

    assignments: list[dict] = []
    for scenario_position, item in enumerate(selected_scenarios):
        if isinstance(item, (tuple, list)):
            scenario_index, scenario = item
        else:
            scenario_index, scenario = 0, item
        if mode == DIRECT_SAMPLER_ALL_TO_ALL:
            for judge_idx, judge_nick in enumerate(model_nicks):
                eval_idxs = [
                    idx for idx in range(num_models)
                    if include_self or idx != judge_idx
                ]
                assignments.append(
                    {
                        "scenario_index": scenario_index,
                        "scenario": scenario,
                        "judge_idx": judge_idx,
                        "judge_nick": judge_nick,
                        "eval_idxs": eval_idxs,
                        "eval_nicks": [model_nicks[idx] for idx in eval_idxs],
                        "sampler_mode": mode,
                        "sampling_round": 0,
                        "group_index": judge_idx,
                    }
                )
            continue

        if mode == DIRECT_SAMPLER_BALANCED:
            for sampling_round in range(response_redundancy):
                shift = balanced_shifts[
                    (scenario_position * response_redundancy + sampling_round)
                    % len(balanced_shifts)
                ]
                for eval_idx in range(num_models):
                    judge_idx = (eval_idx + shift) % num_models
                    assignments.append(
                        {
                            "scenario_index": scenario_index,
                            "scenario": scenario,
                            "judge_idx": judge_idx,
                            "judge_nick": model_nicks[judge_idx],
                            "eval_idxs": [eval_idx],
                            "eval_nicks": [model_nicks[eval_idx]],
                            "sampler_mode": mode,
                            "sampling_round": sampling_round,
                            "group_index": eval_idx,
                        }
                    )
            continue

        used_judges_by_evaluee = [set() for _ in range(num_models)]
        for sampling_round in range(response_redundancy):
            round_assignments = None
            for _attempt in range(1000):
                shuffled = list(range(num_models))
                rng.shuffle(shuffled)
                groups = [
                    shuffled[start : start + group_size]
                    for start in range(0, num_models, group_size)
                ]
                candidate_assignments = []
                for group_index, eval_idxs in enumerate(groups):
                    eligible_judges = [
                        judge_idx
                        for judge_idx in range(num_models)
                        if (include_self or judge_idx not in eval_idxs)
                        and all(
                            judge_idx not in used_judges_by_evaluee[eval_idx]
                            for eval_idx in eval_idxs
                        )
                    ]
                    if not eligible_judges:
                        break
                    judge_idx = rng.choice(eligible_judges)
                    candidate_assignments.append(
                        {
                            "scenario_index": scenario_index,
                            "scenario": scenario,
                            "judge_idx": judge_idx,
                            "judge_nick": model_nicks[judge_idx],
                            "eval_idxs": list(eval_idxs),
                            "eval_nicks": [model_nicks[idx] for idx in eval_idxs],
                            "sampler_mode": mode,
                            "sampling_round": sampling_round,
                            "group_index": group_index,
                        }
                    )
                if len(candidate_assignments) == len(groups):
                    round_assignments = candidate_assignments
                    break
            if round_assignments is None:
                raise ValueError(
                    "could not construct distinct partitioned direct assignments; "
                    "reduce response_redundancy or group_size"
                )
            assignments.extend(round_assignments)
            for assignment in round_assignments:
                for eval_idx in assignment["eval_idxs"]:
                    used_judges_by_evaluee[eval_idx].add(assignment["judge_idx"])
    return assignments


def estimate_direct_calls(
    *,
    num_scenarios: int,
    num_models: int,
    num_openrouter_models: int | None = None,
    include_self: bool = True,
    cached_responses: int = 0,
    cached_openrouter_responses: int = 0,
    sampler_mode: str = DIRECT_SAMPLER_ALL_TO_ALL,
    group_size: int = 4,
    response_redundancy: int = 1,
    sampler_seed: int | None = None,
    assignments: list[dict] | None = None,
    openrouter_model_indices: set[int] | None = None,
) -> dict:
    if num_scenarios < 0 or num_models < 0:
        raise ValueError("num_scenarios and num_models must be non-negative")
    if num_openrouter_models is not None and not 0 <= num_openrouter_models <= num_models:
        raise ValueError("num_openrouter_models must be between zero and num_models")
    mode = DIRECT_SAMPLER_ALIASES.get(str(sampler_mode).strip().lower())
    if mode is None:
        raise ValueError(
            "unknown direct sampler mode; expected 'all_to_all', "
            "'partitioned_random_judge', or 'balanced_unique_judge'"
        )
    group_size = max(1, min(int(group_size), num_models)) if num_models else 0
    response_redundancy = int(response_redundancy)
    if response_redundancy <= 0:
        raise ValueError("response_redundancy must be positive")
    max_redundancy = num_models if include_self else max(0, num_models - 1)
    if mode != DIRECT_SAMPLER_ALL_TO_ALL and response_redundancy > max_redundancy:
        raise ValueError("response_redundancy exceeds the number of distinct eligible judges")
    if mode == DIRECT_SAMPLER_PARTITIONED and not include_self and group_size >= num_models:
        raise ValueError(
            "partitioned direct sampling with include_self=False requires group_size < num_models"
        )
    if assignments is not None:
        total_edges = sum(len(assignment["eval_idxs"]) for assignment in assignments)
    elif mode == DIRECT_SAMPLER_ALL_TO_ALL:
        total_edges = num_scenarios * num_models * (
            num_models if include_self else max(0, num_models - 1)
        )
    else:
        total_edges = num_scenarios * num_models * response_redundancy
    edges_per_scenario = total_edges // num_scenarios if num_scenarios else 0
    total_possible_responses = num_scenarios * num_models
    response_tasks = total_possible_responses - min(
        max(0, int(cached_responses)), total_possible_responses
    )
    reflection_tasks = total_edges
    rating_tasks = reflection_tasks
    result = {
        "num_scenarios": num_scenarios,
        "num_models": num_models,
        "include_self": include_self,
        "sampler_mode": mode,
        "group_size": group_size if mode == DIRECT_SAMPLER_PARTITIONED else None,
        "response_redundancy": (
            response_redundancy if mode != DIRECT_SAMPLER_ALL_TO_ALL else None
        ),
        "sampler_seed": sampler_seed if mode != DIRECT_SAMPLER_ALL_TO_ALL else None,
        "directed_edges_per_scenario": edges_per_scenario,
        "cached_response_hits": min(max(0, int(cached_responses)), total_possible_responses),
        "response_tasks": response_tasks,
        "reflection_tasks": reflection_tasks,
        "rating_tasks": rating_tasks,
        "total_logical_generations": response_tasks + reflection_tasks + rating_tasks,
    }
    if num_openrouter_models is not None:
        k = num_openrouter_models
        total_remote_responses = num_scenarios * k
        remote_responses = total_remote_responses - min(
            max(0, int(cached_openrouter_responses)), total_remote_responses
        )
        if assignments is not None and openrouter_model_indices is not None:
            remote_judge_tasks = sum(
                len(assignment["eval_idxs"])
                for assignment in assignments
                if int(assignment["judge_idx"]) in openrouter_model_indices
            )
        elif mode == DIRECT_SAMPLER_ALL_TO_ALL:
            remote_judge_tasks = num_scenarios * k * (
                num_models if include_self else max(0, num_models - 1)
            )
        else:
            remote_judge_tasks = int(round(total_edges * k / num_models)) if num_models else 0
        result["num_openrouter_models"] = k
        result["cached_openrouter_response_hits"] = min(
            max(0, int(cached_openrouter_responses)), total_remote_responses
        )
        result["openrouter_requests"] = remote_responses + 2 * remote_judge_tasks
        result["local_logical_generations"] = (
            result["total_logical_generations"] - result["openrouter_requests"]
        )
    return result
