"""Add one model to a finished run, collecting only the inference it needs.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from inspect_ai import task

if TYPE_CHECKING:
    from inspect_ai import Task


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


@dataclass
class Plan:
    mode: str
    new_model: str
    as_evaluee: list[Edge] = field(default_factory=list)
    as_judge: list[Edge] = field(default_factory=list)
    targets: dict = field(default_factory=dict)

    @property
    def edges(self) -> list[Edge]:
        return self.as_evaluee + self.as_judge

    def responses_needed(self) -> set[int]:
        """Scenarios where the new model must produce a response of its own."""

        return {e.scenario_index for e in self.as_evaluee}

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "new_model": self.new_model,
            "edges_as_evaluee": len(self.as_evaluee),
            "edges_as_judge": len(self.as_judge),
            "new_responses": len(self.responses_needed()),
            "reused_responses": sum(1 for e in self.as_judge if e.response is not None),
            **self.targets,
        }


def plan_addition(
    records: list[dict],
    new_model: str,
    *,
    include_self: bool = True,
    seed: int | None = 42,
    mode: str | None = None,
) -> Plan:
    """Decide which judgments to collect so the new row and column fit in.
    """

    if not records:
        raise ValueError("no existing records to extend")
    mode = mode or detect_mode(records)
    names = model_names(records, mode)
    if new_model in names:
        raise ValueError(f"{new_model!r} is already in this run")

    n = len(names)
    counts = count_matrix(records, mode, n)
    responses = response_index(records, mode)
    scenarios = sorted(scenarios_of(records))
    rng = random.Random(seed)

    col_sums = [sum(counts[i][j] for i in range(n)) for j in range(n)]
    row_sums = [sum(row) for row in counts]
    col_target = sorted(col_sums)[len(col_sums) // 2]      # median: it is uniform by design
    row_target = round(sum(row_sums) / len(row_sums))      # mean: judging load is lopsided

    plan = Plan(mode=mode, new_model=new_model)
    plan.targets = {
        "existing_models": n,
        "column_target": col_target,
        "row_target": row_target,
        "existing_col_sums": col_sums,
        "existing_row_sums": row_sums,
    }

    if mode == PAIRWISE:
        return _plan_pairwise(
            plan, names, responses, scenarios, rng, col_target, row_target
        )

    # As evaluee. Self-ratings land in this column too, so they take a share
    # of the target rather than adding to it.
    per_judge = _split(col_target, n + 1)[:n] if include_self else _split(col_target, n)
    pool = list(scenarios)
    rng.shuffle(pool)
    cursor = 0
    for j, want in enumerate(per_judge):
        for _ in range(want):
            if cursor >= len(pool):       # more edges than scenarios: wrap around
                rng.shuffle(pool)
                cursor = 0
            s = pool[cursor]; cursor += 1
            plan.as_evaluee.append(Edge(scenario_index=s, judge=names[j], evaluee=new_model))

    # As judge. These responses already exist, so only the judging half is new.
    evaluees = list(range(n)) + ([n] if include_self else [])
    per_evaluee = _split(row_target, len(evaluees))
    for slot, want in zip(evaluees, per_evaluee):
        target_name = new_model if slot == n else names[slot]
        available = [s for s in scenarios if (s, target_name) in responses]
        if slot == n:
            available = sorted(plan.responses_needed())
        if not available:
            continue
        rng.shuffle(available)
        for k in range(want):
            s = available[k % len(available)]
            plan.as_judge.append(
                Edge(
                    scenario_index=s,
                    judge=new_model,
                    evaluee=target_name,
                    response=responses.get((s, target_name)),
                )
            )
    return plan


def _plan_pairwise(plan, names, responses, scenarios, rng, col_target, row_target):
    """Fill the new row and column with comparisons rather than single ratings.

    Each comparison adds one to the judge's row per side, so the model needs
    `col_target` comparisons as a side and `row_target // 2` of its own.
    """

    n = len(names)
    new = plan.new_model

    # As evaluee. Opponents rotate so their counts rise evenly.
    per_judge = _split(col_target, n)
    for j, want in enumerate(per_judge):
        pool = [s for s in scenarios if (s, names[j]) in responses] or list(scenarios)
        rng.shuffle(pool)
        for t in range(want):
            s = pool[t % len(pool)]
            opp = names[(j + 1 + t) % n]          # never compare a judge's own slot first
            plan.as_evaluee.append(
                Edge(
                    scenario_index=s,
                    judge=names[j],
                    evaluee=new,
                    opponent=opp,
                    opponent_response=responses.get((s, opp)),
                )
            )

    # As judge. Both responses already exist.
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(pairs)
    for t in range(max(0, row_target // 2)):
        a, b = pairs[t % len(pairs)]
        ok = [s for s in scenarios
              if (s, names[a]) in responses and (s, names[b]) in responses]
        if not ok:
            continue
        # Walk rather than draw, so a pair spreads before repeating.
        s = ok[(t // len(pairs)) % len(ok)]
        plan.as_judge.append(
            Edge(
                scenario_index=s,
                judge=new,
                evaluee=names[a],
                response=responses.get((s, names[a])),
                opponent=names[b],
                opponent_response=responses.get((s, names[b])),
            )
        )
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
    grid = [row + [0] for row in counts] + [[0] * (n + 1)]
    index = {name: i for i, name in enumerate(names)}
    index[plan.new_model] = n
    for e in plan.edges:
        grid[index[e.judge]][index[e.evaluee]] += 1
        if e.opponent is not None:
            grid[index[e.judge]][index[e.opponent]] += 1
    return grid


# collection: build the samples to collect

def edge_samples(plan: Plan, scenarios: dict[int, str], order: list[str]) -> list:
    """The plan's edges as Inspect samples.

    A known response rides in metadata and the solver uses it as-is.
    """

    from inspect_ai.dataset import Sample

    index = {name: i for i, name in enumerate(order)}
    index[plan.new_model] = len(order)

    samples = []
    for k, e in enumerate(plan.edges):
        md = {
            "edge_index": k,
            "scenario_index": e.scenario_index,
            "scenario": scenarios[e.scenario_index],
            "judge_idx": index[e.judge],
            "judge_nick": e.judge,
            "eval_idx": index[e.evaluee],
            "eval_nick": e.evaluee,
            "sampler_mode": "extend",
            "sampling_round": 0,
            "group_index": 0,
        }
        if e.response is not None:
            md["response"] = e.response
        samples.append(
            Sample(
                input=md["scenario"],
                id=f"s{e.scenario_index:04d} · {e.judge} → {e.evaluee}",
                metadata=md,
            )
        )
    return samples


def eigenbench_extend(
    spec: str,
    *,
    new_model: str,
    plan: Plan,
    scenarios: dict[int, str],
    order: list[str],
    models: dict[str, object] | None = None,
):
    """Task collecting only a plan's edges, on the run's original settings."""

    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.viewer import ViewerConfig

    from inspect_pipeline.eigenbench import _run_context, _samples_view
    from inspect_pipeline.phases import (
        ResponsePool,
        direct_rating_scorer,
        direct_rating_solver,
    )

    ctx = _run_context(spec, models)
    criteria = ctx["criteria"]
    return Task(
        dataset=MemoryDataset(
            samples=edge_samples(plan, scenarios, order), name=f"extend_{new_model}"
        ),
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
        name=f"eigenbench_extend_{new_model}",
        display_name=f"EigenBench extend — {new_model}",
        metadata={"eigenbench": {**ctx["meta"], "extends_with": new_model}},
    )


def check_spec_covers(spec_models, records: list[dict]) -> None:
    """Fail early when the spec names fewer models than the records use.
    """

    mode = detect_mode(records)
    used = model_names(records, mode)
    missing = [m for m in used if m not in spec_models]
    if missing:
        raise SystemExit(
            f"{len(missing)} model(s) appear in evaluations.jsonl but not in the "
            f"spec's models: {missing}. Add them before analysing."
        )


def extend_run(
    spec_ref: str,
    new_model: str,
    model_ref: object,
    *,
    include_self: bool = True,
    seed: int | None = 42,
    dry_run: bool = False,
) -> dict:
    """Collect one model into an existing run and append it to evaluations.jsonl."""

    import json
    from pathlib import Path

    from inspect_ai import eval as inspect_eval

    from pipeline.config import load_run_spec
    from pipeline.utils import load_records

    from .collect import eval_kwargs
    from .export import records_from_log, write_evaluations_atomic

    spec, _run_dir = load_run_spec(spec_ref)
    collection_cfg = spec.get("collection", {})
    evaluations_path = Path(collection_cfg["evaluations_path"])
    records = load_records(str(evaluations_path))
    if not records:
        raise SystemExit(f"no existing records at {evaluations_path}")

    order = model_names(records, detect_mode(records))
    plan = plan_addition(records, new_model, include_self=include_self, seed=seed)
    summary = plan.summary()
    print(json.dumps(summary, indent=2))
    if dry_run:
        return summary

    models = dict(spec["models"])
    models[new_model] = model_ref

    inspect_cfg = collection_cfg.get("inspect", {}) or {}
    log_dir = Path(inspect_cfg.get("log_dir") or "inspect_logs")
    if not log_dir.is_absolute():
        log_dir = evaluations_path.parent / log_dir

    if plan.mode == DIRECT:
        task = eigenbench_extend(
            spec_ref, new_model=new_model, plan=plan,
            scenarios=scenarios_of(records), order=order, models=models,
        )
    else:
        from .pairwise import eigenbench_pairwise_extend

        task = eigenbench_pairwise_extend(
            spec_ref, new_model=new_model, plan=plan,
            scenarios=scenarios_of(records), order=order, models=models,
        )

    logs = inspect_eval(tasks=task, model=None, **eval_kwargs(inspect_cfg, log_dir))

    if plan.mode == DIRECT:
        new_records = records_from_log(logs[0])
    else:
        from .pairwise import records_from_pairwise_log
        from inspect_pipeline.eigenbench import _run_context

        new_records = records_from_pairwise_log(
            logs[0], _run_context(spec_ref, models)["criteria"]
        )
    write_evaluations_atomic(evaluations_path, records + new_records)
    print(f"added {len(new_records)} judgments for {new_model}; "
          f"{evaluations_path} now holds {len(records) + len(new_records)}")
    print(
        "\nAdd the model to the spec before analysing, or aggregation will fail "
        "with 'model index out of range':\n"
        f'    "models": {{..., {new_model!r}: {model_ref!r}}},'
    )
    return {**summary, "collected": len(new_records)}


@task
def extend_model(spec: str, *, new_model: str, model_id: str) -> "Task":
    """Collect one more model into a finished run.

        inspect eval inspect_pipeline/extend.py \
            -T spec=runs/my_run/spec.py -T new_model="Claude Sonnet 4.5" \
            -T model_id=anthropic/claude-sonnet-4-5
    """

    from pipeline.config import load_run_spec
    from pipeline.utils import load_records

    from .eigenbench import resolve_spec_ref

    run_spec, _run_dir = load_run_spec(resolve_spec_ref(spec))
    records = load_records(run_spec["collection"]["evaluations_path"])
    if not records:
        raise ValueError("no existing records to extend")

    order = model_names(records, detect_mode(records))
    plan = plan_addition(records, new_model, seed=42)
    models = dict(run_spec["models"])
    models[new_model] = model_id

    if plan.mode == DIRECT:
        return eigenbench_extend(
            spec, new_model=new_model, plan=plan,
            scenarios=scenarios_of(records), order=order, models=models,
        )
    from .pairwise import eigenbench_pairwise_extend

    return eigenbench_pairwise_extend(
        spec, new_model=new_model, plan=plan,
        scenarios=scenarios_of(records), order=order, models=models,
    )
