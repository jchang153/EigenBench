"""Plan additional judgments from an existing set of records."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations


DIRECT = "direct_rating"
PAIRWISE = "pairwise_btd"


def detect_mode(records: list[dict]) -> str:
    """Which protocol produced these records."""

    for r in records:
        if r.get("record_type") == "direct_rating":
            return DIRECT
        if "eval1" in r and "eval2" in r:
            return PAIRWISE
    raise ValueError("could not tell direct from pairwise records")


def model_names(records: list[dict], mode: str) -> list[str]:
    """Model nicks in index order, as the run recorded them."""

    by_index: dict[int, str] = {}
    for r in records:
        if mode == DIRECT:
            by_index[int(r["judge"]["index"])] = r["judge"]["name"]
            by_index[int(r["evaluee"]["index"])] = r["evaluee"]["name"]
        else:
            by_index[int(r["judge"])] = r["judge_name"]
            by_index[int(r["eval1"])] = r["eval1_name"]
            by_index[int(r["eval2"])] = r["eval2_name"]
    if set(by_index) != set(range(len(by_index))):
        raise ValueError(f"model indices are not contiguous: {sorted(by_index)}")
    return [by_index[i] for i in range(len(by_index))]


def count_matrix(records: list[dict], mode: str, n: int) -> list[list[int]]:
    """counts[judge][evaluee]. A comparison counts once for each side."""

    counts = [[0] * n for _ in range(n)]
    for r in records:
        if mode == DIRECT:
            counts[int(r["judge"]["index"])][int(r["evaluee"]["index"])] += 1
        else:
            j = int(r["judge"])
            counts[j][int(r["eval1"])] += 1
            counts[j][int(r["eval2"])] += 1
    return counts


def response_index(records: list[dict], mode: str) -> dict[tuple[int, str], str]:
    """(scenario_index, model) -> the response already collected.

    A new judge must rate the same text the others saw, so these are reused
    rather than regenerated.
    """

    out: dict[tuple[int, str], str] = {}
    for r in records:
        s = int(r["scenario_index"])
        if mode == DIRECT:
            out[(s, r["evaluee"]["name"])] = r["response"]
        else:
            out[(s, r["eval1_name"])] = r["eval1 response"]
            out[(s, r["eval2_name"])] = r["eval2 response"]
    return out


def scenarios_of(records: list[dict]) -> dict[int, str]:
    return {int(r["scenario_index"]): r["scenario"] for r in records}


@dataclass
class Edge:
    """One judgment to collect: judge rates evaluee on a scenario."""

    scenario_index: int
    judge: str
    evaluee: str
    # Set when the run already holds this response.
    response: str | None = None
    # Pairwise only.
    opponent: str | None = None
    opponent_response: str | None = None
    group: str | None = None


@dataclass
class Plan:
    mode: str
    new_models: list[str]
    new_scenarios: dict[int, str] = field(default_factory=dict)
    as_evaluee: list[Edge] = field(default_factory=list)
    as_judge: list[Edge] = field(default_factory=list)
    targets: dict = field(default_factory=dict)

    @property
    def new_model(self) -> str:
        """Compatibility label for single-model task names."""
        return self.new_models[0] if len(self.new_models) == 1 else f"{len(self.new_models)}_models"

    @property
    def edges(self) -> list[Edge]:
        return self.as_evaluee + self.as_judge

    def responses_needed(self) -> set[int]:
        """Scenarios covered by the response-collection portion of the plan."""

        return {e.scenario_index for e in self.as_evaluee}

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "new_model": self.new_model,
            "new_models": self.new_models,
            "additional_scenarios": len(self.new_scenarios),
            "edges_as_evaluee": len(self.as_evaluee),
            "edges_as_judge": len(self.as_judge),
            "new_responses": len({
                (e.scenario_index, nick)
                for e in self.edges
                for nick, response in ((e.evaluee, e.response),
                                       (e.opponent, e.opponent_response))
                if nick is not None and response is None
            }),
            "reused_responses": sum(1 for e in self.as_judge if e.response is not None),
            **self.targets,
        }


def plan_addition(records, new_model: str, **kwargs) -> Plan:
    """Compatibility wrapper for adding a single model."""
    return plan_additions(records, [new_model], **kwargs)


def plan_additions(
    records: list[dict],
    new_models: list[str],
    *,
    new_scenarios: dict[int, str] | None = None,
    include_self: bool = True,
    seed: int | None = 42,
    mode: str | None = None,
    pairwise_group_size: int = 4,
) -> Plan:
    """Plan the whole expanded population together, preserving original indices."""
    if not records:
        raise ValueError("no existing records to extend")
    mode = mode or detect_mode(records)
    names = model_names(records, mode)
    new_models = list(new_models)
    if len(set(new_models)) != len(new_models) or set(names) & set(new_models):
        raise ValueError("new model names must be unique and absent from the existing run")
    extra = dict(new_scenarios or {})
    scenarios = sorted(scenarios_of(records))
    if set(extra) & set(scenarios):
        raise ValueError("additional scenarios overlap existing scenario indices")
    if not new_models and not extra:
        raise ValueError("specify at least one new model or additional scenario")
    population = names + new_models
    if not include_self and len(population) < 2:
        raise ValueError("include_self=False requires at least two models")
    counts = count_matrix(records, mode, len(names))
    responses = response_index(records, mode)
    rng = random.Random(seed)
    col_sums = [sum(row[j] for row in counts) for j in range(len(names))]
    row_sums = [sum(row) for row in counts]
    col_target = sorted(col_sums)[len(col_sums) // 2]
    row_target = round(sum(row_sums) / len(row_sums))
    plan = Plan(mode=mode, new_models=new_models, new_scenarios=extra)
    plan.targets = {
        "existing_models": len(names),
        "column_target": len(scenarios) if mode == DIRECT else col_target,
        "row_target": row_target,
        "existing_col_sums": col_sums,
        "existing_row_sums": row_sums,
    }
    if mode == PAIRWISE:
        return _plan_pairwise(
            plan, names, responses, scenarios, rng, col_target, row_target,
            include_self, pairwise_group_size,
        )

    # Each new response has one judge drawn from the entire expanded population.
    for new in new_models:
        judges = [j for j in population if include_self or j != new]
        rng.shuffle(judges)
        slots = [j for j, want in zip(judges, _split(len(scenarios), len(judges)))
                 for _ in range(want)]
        rng.shuffle(slots)
        plan.as_evaluee.extend(Edge(s, j, new) for s, j in zip(scenarios, slots))

    # New judges may already judge themselves or other newcomers. Count that
    # work before filling the remainder from existing, saved responses.
    assigned = Counter(e.judge for e in plan.as_evaluee)
    for new in new_models:
        remaining = max(0, row_target - assigned[new])
        evaluees = list(names)
        rng.shuffle(evaluees)
        for target, want in zip(evaluees, _split(remaining, len(names))):
            available = [s for s in scenarios if (s, target) in responses]
            rng.shuffle(available)
            plan.as_judge.extend(
                Edge(s, new, target, response=responses[(s, target)])
                for s in available[:want]
            )

    # Fresh scenarios evaluate the entire population with the existing balanced
    # one-to-one sampler. Every response is rated once and every model judges once.
    if extra:
        from eigenbench.protocol.direct_rating import build_direct_assignments

        assignments = build_direct_assignments(
            sorted(extra.items()), dict.fromkeys(population),
            include_self=include_self, sampler_mode="balanced_unique_judge",
            response_redundancy=1, sampler_seed=seed,
        )
        plan.as_evaluee.extend(
            Edge(a["scenario_index"], a["judge_nick"], target)
            for a in assignments for target in a["eval_nicks"]
        )
    return plan


def _plan_pairwise(plan, names, responses, scenarios, rng, col_target, row_target,
                   include_self, group_size):
    """Collect unique comparison pairs in both presentation orders."""
    population = names + plan.new_models
    seen = set()
    cols, rows = Counter(), Counter()

    def add_pair(s, judge, a, b, destination, group=None):
        key = (s, judge, tuple(sorted((a, b))))
        if key in seen or (not include_self and judge in (a, b)):
            return False
        seen.add(key)
        for first, second in ((a, b), (b, a)):
            destination.append(Edge(
                s, judge, first, response=responses.get((s, first)),
                opponent=second, opponent_response=responses.get((s, second)),
                group=group,
            ))
        cols[a] += 2
        cols[b] += 2
        rows[judge] += 4
        return True

    # Round-robin over judges/opponents; each pass visits unused scenarios.
    for new in plan.new_models:
        combinations_to_visit = [(j, other) for j in population for other in population
                                 if other != new and
                                 (include_self or j not in (new, other))]
        rng.shuffle(combinations_to_visit)
        shuffled = list(scenarios)
        rng.shuffle(shuffled)
        for round_index in range(len(combinations_to_visit)):
            for offset, s in enumerate(shuffled):
                if cols[new] >= col_target:
                    break
                judge, opponent = combinations_to_visit[
                    (round_index + offset) % len(combinations_to_visit)
                ]
                add_pair(s, judge, new, opponent, plan.as_evaluee)
            if cols[new] >= col_target:
                break

    for new in plan.new_models:
        pairs = list(combinations(names, 2))
        rng.shuffle(pairs)
        shuffled = list(scenarios)
        rng.shuffle(shuffled)
        for round_index in range(len(pairs)):
            for offset, s in enumerate(shuffled):
                if rows[new] >= row_target:
                    break
                a, b = pairs[(round_index + offset) % len(pairs)]
                if (s, a) in responses and (s, b) in responses:
                    add_pair(s, new, a, b, plan.as_judge)
            if rows[new] >= row_target:
                break

    # New scenarios: partition all responses into groups and compare every pair
    # in each group, in both orders, as in the legacy grouped protocol.
    if plan.new_scenarios and len(population) < 2:
        raise ValueError("pairwise collection requires at least two models")
    if group_size < 2:
        raise ValueError("pairwise group_size must be at least two")
    judge_load = Counter()
    for s in sorted(plan.new_scenarios):
        shuffled = list(population)
        rng.shuffle(shuffled)
        groups = [shuffled[i:i + group_size] for i in range(0, len(shuffled), group_size)]
        if len(groups) > 1 and len(groups[-1]) == 1:
            groups[-2].extend(groups.pop())
        for group_index, group in enumerate(groups):
            eligible = [j for j in population if include_self or j not in group]
            if not eligible:
                raise ValueError("pairwise no-self collection needs a judge outside each group")
            rng.shuffle(eligible)
            judge = min(eligible, key=lambda j: judge_load[j])
            for a, b in combinations(group, 2):
                add_pair(s, judge, a, b, plan.as_evaluee,
                         group=f"s{s:04d}:extension:g{group_index}")
                judge_load[judge] += 2
    return plan


def _split(total: int, parts: int) -> list[int]:
    """Spread `total` over `parts` as evenly as integers allow."""

    if parts <= 0:
        return []
    base, extra = divmod(max(0, total), parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def projected_counts(records: list[dict], plan: Plan) -> list[list[int]]:
    """The count matrix the run would have once the plan is collected."""

    names = model_names(records, plan.mode)
    n = len(names)
    counts = count_matrix(records, plan.mode, n)
    size = n + len(plan.new_models)
    grid = [row + [0] * len(plan.new_models) for row in counts]
    grid += [[0] * size for _ in plan.new_models]
    index = {name: i for i, name in enumerate(names)}
    index.update({name: n + i for i, name in enumerate(plan.new_models)})
    for e in plan.edges:
        grid[index[e.judge]][index[e.evaluee]] += 1
        if e.opponent is not None:
            grid[index[e.judge]][index[e.opponent]] += 1
    return grid
