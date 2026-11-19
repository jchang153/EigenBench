"""Collect the pairwise task, optionally with one model resident at a time."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample

from .direct_rating import eval_kwargs, log_names, record_refusal
from ..context import _run_context_from_spec, has_local_models, sanitize
from .phased import eval_phase, model_phase
from ..export import write_evaluations_atomic, write_inspect_run_info
from ..pairwise import _group_responses
from ..generation import STORE_RESPONSE


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
    from eigenbench.config import load_run_spec
    from ..context import cached_seed, load_selection, resolve_spec_ref
    from ..models import model_execution
    from ..pairwise import pairwise_comparison_samples, planned_pairwise_comparisons
    from eigenbench.protocol.pairwise_plan import pairwise_call_counts, resolve_pairwise_generation_settings

    spec, run_dir = load_run_spec(resolve_spec_ref(spec_ref))
    panel = models if models is not None else spec["models"]
    selected, _ = load_selection(spec, run_dir)
    collection = spec["collection"]
    resolve_pairwise_generation_settings(collection)
    _, _, comparisons = planned_pairwise_comparisons(spec, selected, list(panel))
    seed = cached_seed(collection.get("cached_responses_path"), selected, panel, merge_rows=False)
    samples = pairwise_comparison_samples(comparisons, cached=seed)
    routes = {nick: model_execution(value) for nick, value in panel.items()}
    return {"mode": "pairwise_btd", **pairwise_call_counts([s.metadata for s in samples], model_routes=routes)}


def _collect_phased(spec_ref, context, task, eval_kw, seen_logs):
    from ..pairwise import build_pairwise_task, records_from_pairwise_log

    models = context["models"]
    response_logs = []
    cells = _response_cells(task.dataset)
    responses = {key: md["response"] for key, md in cells.items()
                 if isinstance(md["response"], str) and md["response"].strip()}
    for nick in models:
        missing = [(key, md) for key, md in cells.items()
                   if key[1] == nick and key not in responses]
        if not missing:
            continue
        with model_phase(context, nick) as ctx:
            response_task = Task(
                name=f"pairwise_responses_{sanitize(nick)}",
                dataset=MemoryDataset([
                    Sample(input=md["scenario"], id=f"{key[0]} {nick}", metadata=md)
                    for key, md in missing
                ]),
                solver=_group_responses(ctx),
                metadata={"eigenbench_phase": "response", "evaluee": nick},
            )
            log = eval_phase(response_task, eval_kw, seen_logs)
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
        with model_phase(context, nick) as ctx:
            judge_task = build_pairwise_task(
                spec_ref, models=models, context=ctx, samples=by_judge[nick],
                name=f"pairwise_judge_{sanitize(nick)}",
            )
            judge_task.metadata["eigenbench"]["pairwise_phased"] = True
            log = eval_phase(judge_task, eval_kw, seen_logs)
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
    from eigenbench.protocol.response_cache import load_cached_responses
    from eigenbench.records import append_records

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
    from eigenbench.config import load_run_spec
    from ..pairwise import build_pairwise_task, records_from_pairwise_log

    if analysis_status is not None:
        analysis_status.clear()
    spec, run_dir = load_run_spec(spec_ref)
    context = _run_context_from_spec(spec, run_dir, models, build_assignments=False)
    task = build_pairwise_task(spec_ref, models=models, context=context, panel_metrics=scoring,
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
    planned = [sample.metadata for sample in task.dataset]
    failure_policy = cfg.get("failure_policy", "strict")
    logs = []
    before = log_names(log_dir)
    try:
        if phased:
            records, logs = _collect_phased(spec_ref, context, task, kwargs, logs)
        else:
            from ..models import check_distinct_local_servers
            check_distinct_local_servers(panel)
            logs = inspect_eval(tasks=task, model=None, **kwargs)
            if len(logs) != 1:
                raise RuntimeError("pairwise collection did not return one complete log")
            records = records_from_pairwise_log(logs[0], task.metadata["eigenbench"]["criteria"])
        if len(records) != len(planned):
            raise RuntimeError("pairwise collection did not complete the plan; nothing exported")
        if output.exists() and output.stat().st_size:
            raise RuntimeError("pairwise output changed during collection; export the logs manually")
        write_evaluations_atomic(output, records)
    except BaseException as exc:  # interrupts too, as in direct collection
        # A different collector may have published here while Inspect was running.
        if not output.is_file() or not output.stat().st_size:
            record_refusal(output.parent, "collection", exc, planned=planned, logs=logs,
                           log_dir=log_dir, before=before, failure_policy=failure_policy,
                           evaluations_path=output)
        raise
    from eigenbench.inspect.coverage import record_collection
    if phased and scoring:
        from ..metrics import finalize_panel_metrics
        panel_log = finalize_panel_metrics(context, logs=logs, records=records,
                                          planned=planned, eval_kwargs=kwargs)
        if panel_log is not None:
            logs = [panel_log] + logs
    record_collection(output.parent, "collection", planned=planned, logs=logs,
                      records=records, failure_policy=failure_policy,
                      evaluations_path=output)
    write_inspect_run_info(output.parent, logs, evaluations=(output, records))
    _persist_responses(cfg.get("cached_responses_path"), records)
    from ..metrics import record_analysis_status
    if scoring:
        record_analysis_status(logs[0], analysis_status)
    return records
