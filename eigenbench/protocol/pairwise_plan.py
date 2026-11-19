"""Plan Jonathan's grouped pairwise protocol with an isolated RNG.

See docs/inspect-pairwise-protocol.md for the pinned compatibility reference.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from eigenbench.protocol.samplers import select_sampler
from eigenbench.config.generation import resolve_direct_generation_settings

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

    return resolve_direct_generation_settings(
        collection_cfg,
        generation_key=PAIRWISE_GENERATION_KEY,
        defaults=PAIRWISE_GENERATION_DEFAULTS,
    )


def resolve_pairwise_sampling_settings(collection_cfg: dict | None, *, num_models: int) -> dict:
    """Sampler settings as ``collect_core_evaluations`` interpreted them."""

    cfg = collection_cfg or {}
    mode = (cfg.get("sampler_mode", "random_judge_group") or "random_judge_group").strip().lower()
    if mode == "adaptive_inverse_count":
        raise ValueError(
            "sampler_mode='adaptive_inverse_count' cannot be planned before collection: "
            "the original draws each round from judge/evaluee counts of comparisons that "
            "already parsed as valid. The legacy collector is retired. Inspect supports "
            "random_judge_group and uniform; changing samplers changes the protocol."
        )
    if mode == "all_to_all":
        raise ValueError(
            "pairwise all_to_all runs went through the mixed local collector, which is not "
            "the reference protocol for the Inspect pairwise task. The legacy collector "
            "is retired; select random_judge_group or uniform only if that protocol is intended."
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


def pairwise_call_counts(comparisons: list[dict], *, model_routes: dict[str, str] | None = None) -> dict:
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
    result = {
        "response_tasks": len(responses),
        "reflection_tasks": len(reflections),
        "comparison_tasks": len(comparisons),
        "total_logical_generations": len(responses) + len(reflections) + len(comparisons),
    }
    if model_routes is not None:
        from collections import Counter

        calls = Counter(model_routes[nick] for _, nick in responses)
        calls.update(model_routes[judge] for _, judge, _ in reflections)
        calls.update(model_routes[row["judge_nick"]] for row in comparisons)
        result.update(
            num_openrouter_models=sum(route == "openrouter" for route in model_routes.values()),
            openrouter_requests=calls["openrouter"],
            local_logical_generations=calls["managed_vllm"],
            other_logical_generations=calls["other"],
        )
    return result
