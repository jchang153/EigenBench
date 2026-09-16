"""Extend a finished run with models and scenarios, reusing existing records.
"""

import random
from collections import Counter
from itertools import combinations
from dataclasses import dataclass, field
import sys
from pathlib import Path

# Native Inspect loads task files outside their package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inspect_ai import Task, task


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
        from pipeline.eval.direct_rating import build_direct_assignments

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

    def add_pair(s, judge, a, b, destination):
        key = (s, judge, tuple(sorted((a, b))))
        if key in seen or (not include_self and judge in (a, b)):
            return False
        seen.add(key)
        for first, second in ((a, b), (b, a)):
            destination.append(Edge(
                s, judge, first, response=responses.get((s, first)),
                opponent=second, opponent_response=responses.get((s, second)),
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
        for group in groups:
            eligible = [j for j in population if include_self or j not in group]
            if not eligible:
                raise ValueError("pairwise no-self collection needs a judge outside each group")
            rng.shuffle(eligible)
            judge = min(eligible, key=lambda j: judge_load[j])
            for a, b in combinations(group, 2):
                add_pair(s, judge, a, b, plan.as_evaluee)
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


# collection: build the samples to collect

def edge_samples(plan: Plan, scenarios: dict[int, str], order: list[str]) -> list:
    """The plan's edges as Inspect samples.

    A known response rides in metadata and the solver uses it as-is.
    """

    from inspect_ai.dataset import Sample

    index = {name: i for i, name in enumerate(order)}
    index.update({name: len(order) + i for i, name in enumerate(plan.new_models)})

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
    context: dict | None = None,
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

    ctx = context if context is not None else _run_context(spec, models, build_assignments=False)
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


def _prepare_extension(
    spec_ref, new_models=None, *, additional_scenarios=None, include_self=None, seed=42,
):
    from pathlib import Path

    from pipeline.config import load_run_spec
    from pipeline.utils import load_records

    from inspect_pipeline.eigenbench import load_selection, resolve_spec_ref

    spec_ref = resolve_spec_ref(spec_ref)
    spec, run_dir = load_run_spec(spec_ref)
    extension = spec.get("extension", {})
    output = Path(spec["collection"]["evaluations_path"]).resolve()
    source = Path(extension.get("from_evaluations") or output).expanduser()
    if not source.is_absolute():
        source = run_dir / source
    source = source.resolve()
    if output != source and output.exists():
        raise ValueError(f"extension output already exists: {output}; choose a new run directory")
    records = load_records(str(source))
    if not records:
        raise ValueError(f"no existing records at {source}")
    mode = detect_mode(records)
    if mode != spec["evaluation"]["mode"]:
        raise ValueError("extension evaluation.mode differs from the source records")
    order = model_names(records, mode)
    check_spec_covers(spec["models"], records)
    models = dict(spec["models"])
    models.update(new_models or {})
    if new_models and set(new_models) & set(order):
        raise ValueError("requested new models already appear in the source records")
    additions = [name for name in models if name not in order]
    # Record indices, rather than spec insertion order, are authoritative.
    models = {name: models[name] for name in order + additions}
    selected, criteria = load_selection(spec, run_dir)
    expected_constitution = "\n".join(criteria)
    if any(r.get("constitution") != expected_constitution for r in records):
        raise ValueError("extension criteria differ from the source records")
    old_scenarios = scenarios_of(records)
    selected_map = dict(selected)
    if any(selected_map.get(s) != text for s, text in old_scenarios.items()):
        raise ValueError(
            "dataset selection must include the original scenario indices and text; "
            "keep dataset/start/shuffle settings and increase dataset.count as needed"
        )
    count = (extension.get("additional_scenarios", 0)
             if additional_scenarios is None else additional_scenarios)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("additional_scenarios must be a non-negative integer")
    candidates = [(i, text) for i, text in selected if i not in old_scenarios]
    if len(candidates) < count:
        raise ValueError(
            f"requested {count} additional scenarios, but dataset selection has only "
            f"{len(candidates)} unused scenarios; increase dataset.count"
        )
    extra = dict(candidates[:count])
    if include_self is None:
        include_self = spec["evaluation"].get("direct_rating", {}).get("include_self", True)
    plan = plan_additions(
        records, additions, new_scenarios=extra, include_self=include_self, seed=seed,
        pairwise_group_size=spec["collection"].get("group_size", 4),
    )
    return {
        "spec_ref": spec_ref, "spec": spec, "records": records, "models": models,
        "order": order, "plan": plan, "source": source, "output": output,
        "scenarios": {**old_scenarios, **extra},
    }


def _extension_task(prepared, *, context=None, task_label=None):
    plan = prepared["plan"]
    factory = eigenbench_extend
    if plan.mode == PAIRWISE:
        from inspect_pipeline.pairwise import eigenbench_pairwise_extend
        factory = eigenbench_pairwise_extend
    result = factory(
        prepared["spec_ref"], new_model=task_label or plan.new_model, plan=plan,
        scenarios=prepared["scenarios"], order=prepared["order"],
        models=prepared["models"], context=context,
    )
    from inspect_pipeline.export import records_fingerprint
    result.metadata["eigenbench"]["extension_source"] = str(prepared["source"])
    result.metadata["eigenbench"]["extension_source_sha256"] = records_fingerprint(prepared["records"])
    result.metadata["eigenbench"]["extension_expected_edges"] = len(plan.edges)
    result.metadata["eigenbench"]["extends_with"] = plan.new_models
    result.metadata["eigenbench"]["additional_scenarios"] = list(plan.new_scenarios)
    return result


def _use_phased_extension(prepared) -> bool:
    from inspect_pipeline.eigenbench import has_local_models
    settings = prepared["spec"]["collection"].get("inspect", {}) or {}
    if "phased" in settings:
        return bool(settings["phased"])
    return has_local_models(prepared["models"])


def extend_run(
    spec_ref: str,
    new_model: str | None = None,
    model_ref: object = None,
    *,
    new_models: dict[str, object] | None = None,
    additional_scenarios: int | None = None,
    include_self: bool | None = None,
    seed: int | None = 42,
    dry_run: bool = False,
) -> dict:
    """Extend from a full-population spec, or explicit new model references.

    The legacy positional (spec, nickname, reference) call remains supported.
    """
    import json
    from pathlib import Path

    from inspect_ai import eval as inspect_eval

    from inspect_pipeline.collect import eval_kwargs
    from inspect_pipeline.export import records_from_log, write_evaluations_atomic, write_inspect_run_info

    if new_model is not None:
        if new_models is not None or model_ref is None:
            raise ValueError("supply either new_model/model_ref or new_models")
        new_models = {new_model: model_ref}
    prepared = _prepare_extension(
        spec_ref, new_models, additional_scenarios=additional_scenarios,
        include_self=include_self, seed=seed,
    )
    plan, spec = prepared["plan"], prepared["spec"]
    summary = {**plan.summary(), "source": str(prepared["source"]),
               "output": str(prepared["output"])}
    print(json.dumps(summary, indent=2))
    if dry_run:
        return summary
    inspect_cfg = spec["collection"].get("inspect", {}) or {}
    log_dir = Path(inspect_cfg.get("log_dir") or "inspect_logs")
    if not log_dir.is_absolute():
        log_dir = prepared["output"].parent / log_dir
    if _use_phased_extension(prepared):
        from inspect_pipeline.extension_phased import collect_phased_extension
        new_records, logs = collect_phased_extension(
            prepared, eval_kwargs(inspect_cfg, log_dir),
        )
    else:
        task = _extension_task(prepared)
        logs = inspect_eval(tasks=task, model=None, **eval_kwargs(inspect_cfg, log_dir))
        if plan.mode == DIRECT:
            new_records = records_from_log(logs[0])
        else:
            from inspect_pipeline.pairwise import records_from_pairwise_log
            new_records = records_from_pairwise_log(
                logs[0], task.metadata["eigenbench"]["criteria"]
            )
    if len(new_records) != len(plan.edges):
        raise RuntimeError("extension did not collect the complete plan; nothing appended")
    # Never overwrite concurrent changes to the source or an independently-created output.
    from pipeline.utils import load_records
    if load_records(str(prepared["source"])) != prepared["records"]:
        raise RuntimeError("source records changed during collection; export the log manually")
    if prepared["output"] != prepared["source"] and prepared["output"].exists():
        raise RuntimeError("output appeared during collection; export the log manually")
    write_evaluations_atomic(prepared["output"], prepared["records"] + new_records)
    write_inspect_run_info(prepared["output"].parent, logs)
    missing = [name for name in plan.new_models if name not in spec["models"]]
    if missing:
        print(f"Add these models to the spec before analysis: {missing}")
    print(f"Added {len(new_records)} judgments to {prepared['output']}")
    return {**summary, "collected": len(new_records)}


@task
def extend_model(
    spec: str, *, new_model: str | None = None, model_id: str | None = None,
    additional_scenarios: int | None = None,
) -> "Task":
    """Collect an extension spec, with optional legacy single-model arguments."""
    if (new_model is None) != (model_id is None):
        raise ValueError("new_model and model_id must be supplied together")
    additions = {new_model: model_id} if new_model is not None else None
    prepared = _prepare_extension(
        spec, additions, additional_scenarios=additional_scenarios,
    )
    if _use_phased_extension(prepared):
        raise ValueError(
            "this extension requires phased execution; use scripts/extend_run.py "
            "instead of the native Inspect task"
        )
    return _extension_task(prepared)
