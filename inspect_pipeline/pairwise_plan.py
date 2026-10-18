"""Plan Jonathan's grouped pairwise protocol with an isolated RNG.

See docs/pairwise-protocol.md for the pinned compatibility reference.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from pipeline.eval.samplers import select_sampler

PAIRWISE_PLAN_VERSION = 1

PAIRWISE_PHASES = ("response", "reflection", "comparison")

# Original OpenRouter collection used 4096 tokens and inherited temperature 1.0
# from get_openrouter_response. collection.max_tokens was never forwarded.
PAIRWISE_GENERATION_DEFAULTS = {
    phase: {"max_tokens": 4096, "temperature": 1.0} for phase in PAIRWISE_PHASES
}

# Historical pairwise specs can contain a direct-rating collection.generation
# block that the original collector ignored. Use a separate override key so
# those settings do not silently change the pairwise protocol.
PAIRWISE_GENERATION_KEY = "pairwise_generation"

PREPLANNED_SAMPLERS = ("random_judge_group", "uniform")


def resolve_pairwise_generation_settings(collection_cfg: dict | None) -> dict[str, dict]:
    """Per-phase ``{max_tokens, temperature, per_model}`` for the pairwise protocol."""

    collection_cfg = collection_cfg or {}
    configured_all = collection_cfg.get(PAIRWISE_GENERATION_KEY) or {}
    if not isinstance(configured_all, dict):
        raise ValueError(f"collection.{PAIRWISE_GENERATION_KEY} must be a mapping")
    unknown = set(configured_all) - set(PAIRWISE_PHASES)
    if unknown:
        raise ValueError(
            f"collection.{PAIRWISE_GENERATION_KEY} has unknown phases {sorted(unknown)}; "
            f"expected {list(PAIRWISE_PHASES)}"
        )

    resolved: dict[str, dict] = {}
    for phase, defaults in PAIRWISE_GENERATION_DEFAULTS.items():
        configured = configured_all.get(phase) or {}
        if not isinstance(configured, dict):
            raise ValueError(f"collection.{PAIRWISE_GENERATION_KEY}.{phase} must be a mapping")
        values = _checked(
            {
                "max_tokens": configured.get("max_tokens", defaults["max_tokens"]),
                "temperature": configured.get("temperature", defaults["temperature"]),
            },
            phase,
        )
        overrides = configured.get("per_model")
        overrides = {} if overrides is None else overrides
        if not isinstance(overrides, dict):
            raise ValueError(
                f"collection.{PAIRWISE_GENERATION_KEY}.{phase}.per_model must be a mapping"
            )
        values["per_model"] = {}
        for nick, override in overrides.items():
            if not isinstance(override, dict):
                raise ValueError(f"per_model override for {nick} must be a mapping")
            values["per_model"][nick] = _checked(
                {**{k: values[k] for k in ("max_tokens", "temperature")}, **override},
                f"{phase}/{nick}",
            )
        resolved[phase] = values
    return resolved


def _checked(values: dict, label: str) -> dict:
    tokens, temperature = values["max_tokens"], values["temperature"]
    if isinstance(tokens, bool) or int(tokens) != tokens or int(tokens) <= 0:
        raise ValueError(f"invalid pairwise max_tokens for {label}: {tokens!r}")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(f"invalid pairwise temperature for {label}: {temperature!r}")
    return {"max_tokens": int(tokens), "temperature": temperature}


def resolve_pairwise_sampling_settings(collection_cfg: dict | None, *, num_models: int) -> dict:
    """Sampler settings as ``collect_core_evaluations`` interpreted them."""

    cfg = collection_cfg or {}
    mode = (cfg.get("sampler_mode", "random_judge_group") or "random_judge_group").strip().lower()
    if mode == "adaptive_inverse_count":
        raise ValueError(
            "sampler_mode='adaptive_inverse_count' cannot be planned before collection: "
            "the original draws each round from judge/evaluee counts of comparisons that "
            "already parsed as valid (pipeline/eval/collect.py). Use random_judge_group or "
            "uniform, or keep the legacy collector for adaptive runs."
        )
    if mode == "all_to_all":
        raise ValueError(
            "pairwise all_to_all runs went through the mixed local collector, which is not "
            "the reference protocol for the Inspect pairwise task"
        )
    if mode not in PREPLANNED_SAMPLERS:
        raise ValueError(f"Unknown sampler mode: {mode}")

    num_models = int(num_models)
    group_size = int(cfg.get("group_size", 4))
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    group_size = min(group_size, num_models)
    if group_size < 2:
        raise ValueError("pairwise collection needs at least two models in each group")
    groups = max(1, int(cfg.get("groups", 1)))

    seed = cfg.get("sampler_seed", 42)
    if seed is None or isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("pairwise plans need an integer collection.sampler_seed")
    return {
        "sampler_mode": mode,
        "group_size": group_size,
        "groups": groups,
        "sampler_seed": int(seed),
    }


@dataclass(frozen=True)
class PairwiseGroup:
    """One sampled judge and evaluee group for one scenario round."""

    scenario_index: int
    scenario: str
    sampling_round: int
    group_id: str
    judge_idx: int
    judge_nick: str
    eval_idxs: tuple[int, ...]
    eval_nicks: tuple[str, ...]


def plan_pairwise_groups(
    selected,
    model_nicks,
    *,
    sampler_mode: str,
    group_size: int,
    groups: int,
    sampler_seed: int,
) -> list[PairwiseGroup]:
    """Every (scenario, round) group of the original collector, in collection order."""

    nicks = list(model_nicks)
    if sampler_mode not in PREPLANNED_SAMPLERS:
        raise ValueError(f"sampler_mode {sampler_mode!r} cannot be preplanned")
    if not 2 <= int(group_size) <= len(nicks):
        raise ValueError("group_size must be between 2 and the number of models")
    sampler = select_sampler(sampler_mode)
    rng = random.Random(int(sampler_seed))

    planned: list[PairwiseGroup] = []
    seen: set[int] = set()
    for scenario_index, scenario in selected:
        scenario_index = int(scenario_index)
        if scenario_index in seen:
            raise ValueError(f"scenario index {scenario_index} is selected twice")
        seen.add(scenario_index)
        for sampling_round in range(max(1, int(groups))):
            judge_idx, eval_idxs = sampler(len(nicks), int(group_size), rng=rng)
            planned.append(
                PairwiseGroup(
                    scenario_index=scenario_index,
                    scenario=scenario,
                    sampling_round=sampling_round,
                    group_id=f"s{scenario_index:04d}:r{sampling_round}",
                    judge_idx=judge_idx,
                    judge_nick=nicks[judge_idx],
                    eval_idxs=tuple(int(i) for i in eval_idxs),
                    eval_nicks=tuple(nicks[i] for i in eval_idxs),
                )
            )
    return planned


def pairwise_comparisons(groups: list[PairwiseGroup]) -> list[dict]:
    """Match ``collect_group_criteria_evaluations`` ordering and group sharing.

    Iterate j then k in the group's sampled order, skipping j == k, with j
    shown first. This emits both presentation orders of every distinct pair.
    Set ``response_group`` and ``group_id`` to the group's identity so its
    comparisons reuse the same generated responses and judge reflections.
    Separate groups draw independently, though an explicit response cache
    can supply the same response text to several groups.
    """

    rows: list[dict] = []
    for group in groups:
        names = dict(zip(group.eval_idxs, group.eval_nicks))
        for j in group.eval_idxs:
            for k in group.eval_idxs:
                if j == k:
                    continue
                rows.append(
                    {
                        "edge_index": len(rows),
                        "scenario_index": group.scenario_index,
                        "scenario": group.scenario,
                        "sampling_round": group.sampling_round,
                        "group_id": group.group_id,
                        "response_group": group.group_id,
                        "judge_idx": group.judge_idx,
                        "judge_nick": group.judge_nick,
                        "eval1_idx": j,
                        "eval1_nick": names[j],
                        "eval2_idx": k,
                        "eval2_nick": names[k],
                    }
                )
    return rows


def pairwise_call_counts(comparisons: list[dict]) -> dict:
    """Count generations after cached responses have been added to the samples.

    Count one response per response group and evaluee unless supplied, one
    reflection per group, judge and evaluee, and one verdict per comparison.
    """

    responses, reflections = set(), set()
    for row in comparisons:
        for side in ("eval1", "eval2"):
            nick = row[f"{side}_nick"]
            text = row.get(f"{side}_response")
            if not (isinstance(text, str) and text.strip()):
                responses.add((row["response_group"], nick))
            reflections.add((row["group_id"], row["judge_nick"], nick))
    return {
        "response_tasks": len(responses),
        "reflection_tasks": len(reflections),
        "comparison_tasks": len(comparisons),
        "total_logical_generations": len(responses) + len(reflections) + len(comparisons),
    }
