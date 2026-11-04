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
from .pairwise import _group_responses
from .phases import STORE_RESPONSE


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
    response_logs = []
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
        response_logs.append(log)
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
        records.extend(zip(
            [sample.metadata["edge_index"] for sample in ordered],
            records_from_pairwise_log(log, task.metadata["eigenbench"]["criteria"]),
        ))
        logs.append(log)
    return [record for _, record in sorted(records, key=lambda item: item[0])], logs + response_logs


def _persist_responses(path, records) -> None:
    """Append newly generated responses to the scenario response cache.

    Pairwise reads only a scenario's last cache row, so each row also carries the
    scenario's cached responses. A model answering in several groups keeps its first.
    """

    if not path:
        return
    from pipeline.protocol.response_cache import load_cached_responses
    from pipeline.utils import append_records

    cached = load_cached_responses(path, merge_rows=False)
    scenarios, new = {}, defaultdict(dict)
    for record in records:
        s_idx = int(record["scenario_index"])
        scenarios[s_idx] = record["scenario"]
        for side in ("eval1", "eval2"):
            nick = record[f"{side}_name"]
            if nick not in cached.get(s_idx, {}):
                new[s_idx].setdefault(nick, record[f"{side} response"])
    rows = [
        {"scenario": scenarios[s_idx], "scenario_index": s_idx,
         "responses": {**cached.get(s_idx, {}), **responses}}
        for s_idx, responses in sorted(new.items())
    ]
    if rows:
        append_records(path, rows)


def collect_pairwise_inspect(spec_ref: str, *, models=None, analysis_status: dict | None = None,
                             scoring: bool = True) -> list[dict]:
    from inspect_ai import eval as inspect_eval
    from pipeline.config import load_run_spec
    from .pairwise import build_pairwise_task, records_from_pairwise_log

    if analysis_status is not None:
        analysis_status.clear()
    spec, _ = load_run_spec(spec_ref)
    task = build_pairwise_task(spec_ref, models=models, panel_metrics=scoring,
                               write_artifacts=bool(spec.get("training", {}).get("enabled", True)))
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
    phased = bool(settings.get("phased", has_local_models(panel)))
    if phased:
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
    from .coverage import record_collection
    planned = [sample.metadata for sample in task.dataset]
    if phased and scoring:
        from .eigenbench import _run_context
        from .metrics import finalize_panel_metrics
        ctx = _run_context(spec_ref, panel, build_assignments=False)
        panel_log = finalize_panel_metrics(ctx, logs=logs, records=records,
                                          planned=planned, eval_kwargs=kwargs)
        if panel_log is not None:
            logs = [panel_log] + logs
    record_collection(output.parent, "collection", planned=planned, logs=logs,
                      records=records, failure_policy=cfg.get("failure_policy", "strict"),
                      evaluations_path=output)
    write_inspect_run_info(output.parent, logs)
    _persist_responses(cfg.get("cached_responses_path"), records)
    from .metrics import record_analysis_status
    if scoring:
        record_analysis_status(logs[0], analysis_status)
    return records
