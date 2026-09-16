"""Run extension responses and judgments with one model resident at a time."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import anyio
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample

from .eigenbench import _model_resolver, _run_context, sanitize
from .export import records_from_log
from .extend import DIRECT, _extension_task, response_index
from .pairwise import records_from_pairwise_log
from .phases import STORE_RESPONSE, response_only_solver


def _isolated_context(prepared, nick):
    ctx = _run_context(prepared["spec_ref"], prepared["models"], build_assignments=False)
    # A provider closed after response generation must not be reused when the
    # same model later returns as a judge. Keep clients private to this phase.
    resolve = _model_resolver(prepared["models"], memoize=False)
    loaded = []

    def resolve_only_this_model(name):
        if name != nick:
            raise RuntimeError(f"phased task for {nick!r} attempted to load {name!r}")
        model = resolve(name)
        if not loaded:
            loaded.append(model)
        return model

    ctx["resolve_model"] = resolve_only_this_model
    return ctx, loaded


async def _close_loaded(loaded):
    for model in loaded:
        close = getattr(model.api, "aclose", None)
        if close is not None:
            # A failed close must stop execution, not leave a server resident
            # while the next model tries to claim the GPU.
            await close()


def _run_one(task, loaded, eval_kwargs):
    try:
        logs = inspect_eval(tasks=task, model=None, max_tasks=1, **eval_kwargs)
        if len(logs) != 1 or logs[0].status != "success":
            raise RuntimeError("phased extension task did not complete; nothing appended")
        log = logs[0]
        if len(log.samples or []) != len(task.dataset) or any(s.error for s in log.samples):
            raise RuntimeError("phased extension task has incomplete samples; nothing appended")
        return log
    finally:
        anyio.run(_close_loaded, loaded)


def collect_phased_extension(prepared, eval_kwargs):
    plan = prepared["plan"]
    responses = response_index(prepared["records"], plan.mode)
    required = defaultdict(set)
    for edge in plan.edges:
        for nick, supplied in ((edge.evaluee, edge.response),
                               (edge.opponent, edge.opponent_response)):
            if nick is None:
                continue
            key = (edge.scenario_index, nick)
            if isinstance(supplied, str) and supplied.strip():
                responses[key] = supplied
            if not isinstance(responses.get(key), str) or not responses[key].strip():
                required[nick].add(edge.scenario_index)

    # Only missing responses, not the original run's full population/scenario grid.
    for nick in prepared["models"]:
        if not required[nick]:
            continue
        ctx, loaded = _isolated_context(prepared, nick)
        samples = [Sample(
            input=prepared["scenarios"][s], id=f"s{s} {nick}",
            metadata={"scenario_index": s, "scenario": prepared["scenarios"][s],
                      "eval_nick": nick},
        ) for s in sorted(required[nick])]
        task = Task(
            name=f"extension_responses_{sanitize(nick)}",
            dataset=MemoryDataset(samples),
            solver=response_only_solver(
                resolve_model=ctx["resolve_model"], generation_cfg=ctx["generation"]["response"],
                max_attempts=ctx["max_attempts"], cache_enabled=ctx["cache_enabled"],
            ),
            metadata={"eigenbench_phase": "response", "evaluee": nick},
        )
        log = _run_one(task, loaded, eval_kwargs)
        for sample in log.samples:
            value = (sample.store or {}).get(STORE_RESPONSE)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError("phased extension is missing a response; nothing appended")
            responses[(int(sample.metadata["scenario_index"]), nick)] = value

    by_judge = defaultdict(list)
    for edge in plan.edges:
        by_judge[edge.judge].append(replace(
            edge, response=responses[(edge.scenario_index, edge.evaluee)],
            opponent_response=(responses[(edge.scenario_index, edge.opponent)]
                               if edge.opponent is not None else None),
        ))
    records, logs = [], []
    for nick in prepared["models"]:
        if not by_judge[nick]:
            continue
        ctx, loaded = _isolated_context(prepared, nick)
        portion = replace(plan, as_evaluee=by_judge[nick], as_judge=[])
        task = _extension_task(
            {**prepared, "plan": portion}, context=ctx, task_label=f"judge_{sanitize(nick)}",
        )
        task.metadata["eigenbench"]["extension_phased"] = True
        log = _run_one(task, loaded, eval_kwargs)
        if plan.mode == DIRECT:
            records.extend(records_from_log(log))
        else:
            records.extend(records_from_pairwise_log(log, ctx["criteria"]))
        logs.append(log)

    # Return the same plan ordering as the single-task path.
    def edge_key(edge):
        return (edge.scenario_index, edge.judge, edge.evaluee, edge.opponent)

    def record_key(record):
        if plan.mode == DIRECT:
            return (record["scenario_index"], record["judge"]["name"],
                    record["evaluee"]["name"], None)
        return (record["scenario_index"], record["judge_name"],
                record["eval1_name"], record["eval2_name"])

    rank = {edge_key(edge): i for i, edge in enumerate(plan.edges)}
    records.sort(key=lambda r: rank[record_key(r)])
    return records, logs
