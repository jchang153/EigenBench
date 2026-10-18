"""Collect the pairwise task, optionally with one model resident at a time."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample

from .collect import eval_kwargs
from .eigenbench import has_local_models, sanitize
from .export import write_evaluations_atomic, write_inspect_run_info
from .extension_phased import _isolated_context, _run_one
from .phases import STORE_RESPONSE, response_only_solver


def _group_responses(ctx):
    from .pairwise import pairwise_cache_scope

    return response_only_solver(
        resolve_model=ctx["resolve_model"], generation_cfg=ctx["generation"]["response"],
        max_attempts=ctx["max_attempts"], cache_enabled=ctx["cache_enabled"],
        cache_scope=lambda md: pairwise_cache_scope(
            "response", md["response_group"], md["eval_nick"]
        ),
    )


def _response_cells(samples):
    cells = {}
    for sample in samples:
        md = sample.metadata
        for side in (1, 2):
            key = (md["response_group"], md[f"eval{side}_nick"])
            cells[key] = {
                "scenario_index": md["scenario_index"], "scenario": md["scenario"],
                "response_group": key[0], "eval_nick": key[1],
                "response": md.get(f"eval{side}_response"),
            }
    return cells


def estimate_pairwise_calls(spec_ref: str, *, models=None) -> dict:
    from .pairwise import build_pairwise_task
    from .pairwise_plan import pairwise_call_counts

    task = build_pairwise_task(spec_ref, models=models)
    return {"mode": "pairwise_btd",
            **pairwise_call_counts([sample.metadata for sample in task.dataset])}


def _collect_phased(spec_ref, models, task, eval_kw):
    from .pairwise import build_pairwise_task, records_from_pairwise_log

    prepared = {"spec_ref": spec_ref, "models": models}
    cells = _response_cells(task.dataset)
    responses = {key: md["response"] for key, md in cells.items()
                 if isinstance(md["response"], str) and md["response"].strip()}
    for nick in models:
        missing = [(key, md) for key, md in cells.items()
                   if key[1] == nick and key not in responses]
        if not missing:
            continue
        ctx, loaded = _isolated_context(prepared, nick)
        response_task = Task(
            name=f"pairwise_responses_{sanitize(nick)}",
            dataset=MemoryDataset([
                Sample(input=md["scenario"], id=f"{key[0]} {nick}", metadata=md)
                for key, md in missing
            ]),
            solver=_group_responses(ctx),
            metadata={"eigenbench_phase": "response", "evaluee": nick},
        )
        log = _run_one(response_task, loaded, eval_kw)
        for sample in log.samples:
            content = (sample.store or {}).get(STORE_RESPONSE)
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("pairwise response phase is incomplete")
            responses[(sample.metadata["response_group"], nick)] = content

    by_judge = defaultdict(list)
    for sample in task.dataset:
        md = dict(sample.metadata)
        for side in (1, 2):
            md[f"eval{side}_response"] = responses[(md["response_group"], md[f"eval{side}_nick"])]
        by_judge[md["judge_nick"]].append(
            Sample(input=sample.input, id=sample.id, metadata=md)
        )
    logs, records = [], []
    for nick in models:
        if not by_judge[nick]:
            continue
        ctx, loaded = _isolated_context(prepared, nick)
        judge_task = build_pairwise_task(
            spec_ref, models=models, context=ctx, samples=by_judge[nick],
            name=f"pairwise_judge_{sanitize(nick)}",
        )
        judge_task.metadata["eigenbench"]["pairwise_phased"] = True
        log = _run_one(judge_task, loaded, eval_kw)
        ordered = sorted(log.samples, key=lambda sample: sample.metadata["edge_index"])
        ordered_log = log.model_copy(update={"samples": ordered})
        records.extend(zip(
            [sample.metadata["edge_index"] for sample in ordered],
            records_from_pairwise_log(ordered_log, task.metadata["eigenbench"]["criteria"]),
        ))
        logs.append(log)
    return [record for _, record in sorted(records, key=lambda item: item[0])], logs


def collect_pairwise_inspect(spec_ref: str, *, models=None) -> list[dict]:
    from inspect_ai import eval as inspect_eval
    from pipeline.config import load_run_spec
    from .pairwise import build_pairwise_task, records_from_pairwise_log

    spec, _ = load_run_spec(spec_ref)
    task = build_pairwise_task(spec_ref, models=models)
    cfg = spec["collection"]
    output = Path(cfg["evaluations_path"])
    if output.exists() and output.stat().st_size:
        raise RuntimeError(f"{output} already contains records; use an extension or a new output")
    settings = cfg.get("inspect", {}) or {}
    log_dir = Path(settings.get("log_dir") or "inspect_logs")
    if not log_dir.is_absolute():
        log_dir = output.parent / log_dir
    panel = models if models is not None else spec["models"]
    kwargs = eval_kwargs(settings, log_dir)
    if bool(settings.get("phased", has_local_models(panel))):
        records, logs = _collect_phased(spec_ref, panel, task, kwargs)
    else:
        logs = inspect_eval(tasks=task, model=None, **kwargs)
        if len(logs) != 1:
            raise RuntimeError("pairwise collection did not return one complete log")
        records = records_from_pairwise_log(logs[0], task.metadata["eigenbench"]["criteria"])
    if len(records) != len(task.dataset):
        raise RuntimeError("pairwise collection did not complete the plan; nothing exported")
    if output.exists() and output.stat().st_size:
        raise RuntimeError("pairwise output changed during collection; export the logs manually")
    write_evaluations_atomic(output, records)
    write_inspect_run_info(output.parent, logs)
    return records
