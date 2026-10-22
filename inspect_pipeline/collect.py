"""Programmatic driver: build the task, run it, export evaluations.jsonl.

Equivalent to ``inspect eval inspect_pipeline/eigenbench.py -T spec=...``
followed by ``scripts/export_evaluations.py``; used by scripts/run_inspect.py.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from inspect_ai import eval as inspect_eval

from pipeline.config import load_run_spec
from pipeline.eval.direct_rating import (
    build_direct_assignments,
    count_cached_responses,
    estimate_direct_calls,
    resolve_direct_sampling_settings,
)
from pipeline.model_refs import is_hf_local_model
from pipeline.utils import load_records

from .eigenbench import _run_context, eigenbench, has_local_models, load_selection
from .phased import eigenbench_judge, eigenbench_responses
from .export import (
    check_complete,
    export_log,
    records_from_logs,
    write_evaluations_atomic,
    write_inspect_run_info,
    write_omissions,
)
from .coverage import planned_from_assignments, record_collection
from .provenance import check_reusable, write_fingerprint
from .phases import STORE_RESPONSE


def write_call_estimate(spec: dict, run_dir: Path) -> dict:
    """Exact planned call counts, mirroring scripts/run_collect.py."""

    models = spec["models"]
    collection_cfg = spec.get("collection", {})
    evaluation_cfg = spec.get("evaluation", {})
    include_self = bool(
        (evaluation_cfg.get("direct_rating", {}) or {}).get("include_self", True)
    )
    selected, _criteria = load_selection(spec, run_dir)
    sampling = resolve_direct_sampling_settings(
        collection_cfg, num_models=len(models), include_self=include_self
    )
    assignments = build_direct_assignments(
        selected, models, include_self=include_self, **sampling
    )
    openrouter_nicks = {
        nick for nick, value in models.items() if not is_hf_local_model(value)
    }
    cached_total, cached_remote = count_cached_responses(
        collection_cfg.get("cached_responses_path"),
        scenario_indices={int(item[0]) for item in selected},
        model_nicks=set(models),
        openrouter_nicks=openrouter_nicks,
    )
    estimate = {
        "mode": "direct_rating",
        **estimate_direct_calls(
            num_scenarios=len(selected),
            num_models=len(models),
            num_openrouter_models=len(openrouter_nicks),
            include_self=include_self,
            cached_responses=cached_total,
            cached_openrouter_responses=cached_remote,
            assignments=assignments,
            openrouter_model_indices={
                idx for idx, nick in enumerate(models) if nick in openrouter_nicks
            },
            **sampling,
        ),
    }
    estimate_path = Path(collection_cfg["evaluations_path"]).with_name(
        "direct_call_estimate.json"
    )
    estimate_path.parent.mkdir(parents=True, exist_ok=True)
    estimate_path.write_text(json.dumps(estimate, indent=2) + "\n", encoding="utf-8")
    return estimate


def eval_kwargs(inspect_cfg: dict, log_dir: Path) -> dict:
    kwargs: dict = {"log_dir": str(log_dir), "fail_on_error": False}
    for key in ("max_connections", "max_samples", "retry_on_error"):
        if inspect_cfg.get(key) is not None:
            kwargs[key] = int(inspect_cfg[key])
    if inspect_cfg.get("display") is not None:
        kwargs["display"] = inspect_cfg["display"]
    return kwargs


def log_names(log_dir: Path) -> set[str]:
    """The logs already in ``log_dir``, so a stopped run can find the ones it wrote."""

    from inspect_ai.log import list_eval_logs

    return {Path(info.name).name for info in list_eval_logs(str(log_dir))} if log_dir.is_dir() else set()


def record_refusal(run_dir: Path, kind: str, exc: BaseException, *, planned, logs: list,
                   log_dir: Path, before: set[str], failure_policy: str,
                   evaluations_path: Path | None = None) -> None:
    """Report a collection or extension that stopped, with every log it wrote.

    An interrupt leaves its active log unreturned, so logs new to ``log_dir`` are added.
    """

    try:
        from inspect_ai.log import list_eval_logs, read_eval_log

        known = before | {Path(str(log.location)).name for log in logs}
        if log_dir.is_dir():
            logs = logs + [read_eval_log(info.name) for info in list_eval_logs(str(log_dir))
                           if Path(info.name).name not in known]
        if logs:
            write_inspect_run_info(run_dir, logs, append=kind == "extension", primary=False)
    except Exception as error:  # never hide the run's own outcome
        print(f"Warning: could not list the stopped run's logs: {error}", flush=True)
    record_collection(run_dir, kind, planned=planned, logs=logs, records=[],
                      failure_policy=failure_policy, status="refused",
                      error=str(exc) or type(exc).__name__, evaluations_path=evaluations_path)


def _is_local(value) -> bool:
    return has_local_models({"model": value})


async def _close_model(nick: str, resolve_model) -> None:
    """Terminate a local model's vLLM server so the next one can have the GPU."""

    try:
        api = resolve_model(nick).api
    except Exception:
        return
    aclose = getattr(api, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception as exc:  # a stuck server must not sink the run
            print(f"  warning: could not close {nick}: {exc}")


def _persist_responses(ctx, responses, generated) -> None:
    """Append scenarios with new responses, once every model has one, to the response cache."""

    path = ctx["collection_cfg"].get("cached_responses_path")
    nicks = list(ctx["models"])
    rows = [
        {"scenario": scenario, "scenario_index": int(s_idx),
         "responses": {nick: responses[int(s_idx)][nick] for nick in nicks}}
        for s_idx, scenario in ctx["selected"]
        if any((int(s_idx), nick) in generated for nick in nicks)
        and all(nick in responses[int(s_idx)] for nick in nicks)
    ]
    if path and rows:
        from pipeline.utils import append_records

        append_records(path, rows)


def _run_phased(spec_ref: str, models, ctx, eval_kw: dict, seen_logs: list | None = None) -> list[dict]:
    """Responses then judgments, one model resident at a time.

    A judgment needs both its evaluee and its judge, so an edge-per-sample run
    keeps every model live at once. That is fine for hosted models and fatal on
    one GPU, so local runs phase the work the way the legacy collector did.
    Every log written, responses included, is added to ``seen_logs`` as it lands.
    """

    seen_logs = seen_logs if seen_logs is not None else []

    import anyio

    from inspect_ai import eval as inspect_eval

    resolve_model = ctx["resolve_model"]
    nicks = list(models)
    responses: dict[int, dict[str, str]] = defaultdict(dict)
    for (s_idx, nick), text in ctx["seed"].items():
        responses[s_idx][nick] = text

    judge_logs = []
    response_errors: dict[tuple[int, str], str] = {}

    def close(nick: str) -> None:
        # Free a local server's GPU (it restarts on next use); API clients must stay open to judge.
        if _is_local(models[nick]):
            anyio.run(_close_model, nick, resolve_model)

    print(f"Phase 1/2: responses, {len(nicks)} model(s), one at a time")
    generated: set[tuple[int, str]] = set()
    for nick in nicks:
        missing = [int(s) for s, _ in ctx["selected"] if nick not in responses[int(s)]]
        if not missing:
            continue
        logs = inspect_eval(
            tasks=eigenbench_responses(spec_ref, model_nick=nick, models=models,
                                       scenario_indices=missing),
            model=None,
            max_tasks=1,
            **eval_kw,
        )
        seen_logs.append(logs[0])
        for sample in logs[0].samples or []:
            content = (sample.store or {}).get(STORE_RESPONSE)
            if isinstance(content, str) and content.strip():
                responses[int(sample.metadata["scenario_index"])][nick] = content
                generated.add((int(sample.metadata["scenario_index"]), nick))
            elif sample.error is not None:
                response_errors[(int(sample.metadata["scenario_index"]), nick)] = sample.error.message
        if logs[0].error is not None:
            for s in missing:
                response_errors.setdefault((int(s), nick), logs[0].error.message)
        close(nick)
    _persist_responses(ctx, responses, generated)

    # Every policy needs every response: a judgment cannot be made without one.
    needed = sorted({
        (int(a["scenario_index"]), eval_nick)
        for a in ctx["assignments"]
        for eval_nick in a["eval_nicks"]
        if eval_nick not in responses[int(a["scenario_index"])]
    })
    if needed:
        shown = "\n  ".join(
            f"scenario_index={s} evaluee={nick}: {response_errors.get((s, nick), 'no response in the log')}"
            for s, nick in needed[:10]
        )
        more = f"\n  ... and {len(needed) - 10} more" if len(needed) > 10 else ""
        involved = ", ".join(sorted({nick for _, nick in needed}))
        raise RuntimeError(
            f"{len(needed)} response(s) failed; no judgments run and nothing exported. "
            f"Models involved: {involved}\n  {shown}{more}"
        )

    edges_by_judge: dict[str, list[dict]] = defaultdict(list)
    for a in ctx["assignments"]:
        s_idx = int(a["scenario_index"])
        for eval_idx, eval_nick in zip(a["eval_idxs"], a["eval_nicks"]):
            edges_by_judge[a["judge_nick"]].append(
                {
                    "edge_index": len(edges_by_judge[a["judge_nick"]]),
                    "scenario_index": s_idx,
                    "scenario": a["scenario"],
                    "judge_idx": int(a["judge_idx"]),
                    "judge_nick": a["judge_nick"],
                    "eval_idx": int(eval_idx),
                    "eval_nick": eval_nick,
                    "sampler_mode": a.get("sampler_mode"),
                    "sampling_round": int(a.get("sampling_round", 0)),
                    "group_index": int(a.get("group_index", 0)),
                    "response": responses[s_idx][eval_nick],
                }
            )

    print(f"Phase 2/2: judgments, {len(edges_by_judge)} judge(s), one at a time")
    for nick, edges in edges_by_judge.items():
        logs = inspect_eval(
            tasks=eigenbench_judge(
                spec_ref, judge_nick=nick, edges=edges, models=models
            ),
            model=None,
            max_tasks=1,
            **eval_kw,
        )
        judge_logs.append(logs[0])
        seen_logs.append(logs[0])
        close(nick)

    return judge_logs


def collect_direct_ratings_inspect(
    spec_ref: str,
    *,
    models: dict[str, object] | None = None,
) -> list[dict]:
    """Run collection for a run spec and write evaluations.jsonl."""

    spec, run_dir = load_run_spec(spec_ref)
    collection_cfg = spec.get("collection", {})
    evaluations_path = Path(collection_cfg["evaluations_path"])
    inspect_cfg = collection_cfg.get("inspect", {}) or {}
    # _run_context validates it on every collection path.
    failure_policy = collection_cfg.get("failure_policy", "strict")

    log_dir = Path(inspect_cfg.get("log_dir") or "inspect_logs")
    if not log_dir.is_absolute():
        log_dir = evaluations_path.parent / log_dir

    estimate = write_call_estimate(spec, run_dir)
    expected = estimate["rating_tasks"]
    ctx = _run_context(spec_ref, models)
    # A completed run short-circuits only if it is this plan's collection; anything else
    # is refused with what differs.
    if evaluations_path.exists() and evaluations_path.stat().st_size > 0:
        existing = load_records(str(evaluations_path))
        omitted_path = evaluations_path.with_name("omitted_samples.json")
        omitted = []
        if failure_policy == "omit_invalid_judgments" and omitted_path.exists():
            omitted = json.loads(omitted_path.read_text(encoding="utf-8")).get("samples", [])
        check_reusable(evaluations_path, existing, ctx, omitted)
        print(f"Direct collection already complete: {len(existing)} ratings")
        return existing

    print(
        f"Direct-rating plan: {estimate['total_logical_generations']} logical "
        f"generations ({estimate['response_tasks']} responses, "
        f"{estimate['reflection_tasks']} reflections, {estimate['rating_tasks']} ratings)"
    )

    # The estimate counts the spec's own models, so check completeness against it only then.
    planned = expected if models is None else None
    eval_kw = eval_kwargs(inspect_cfg, log_dir)
    spec_models = models if models is not None else spec["models"]
    phased = bool(inspect_cfg.get("phased", has_local_models(spec_models)))

    seen_logs: list = []
    planned_judgments = None
    before = log_names(log_dir)
    try:
        if phased:
            planned_judgments = planned_from_assignments(ctx["assignments"])
            judge_logs = _run_phased(spec_ref, spec_models, ctx, eval_kw, seen_logs)
            omissions: list[dict] = []
            records = records_from_logs(judge_logs, failure_policy=failure_policy, omissions=omissions)
            if planned is not None:
                check_complete(planned=planned, records=records, omissions=omissions)
            write_evaluations_atomic(evaluations_path, records)
            write_omissions(evaluations_path, planned=planned or len(records) + len(omissions),
                            records=records, omissions=omissions)
            # Judge logs first: the first is the primary log the viewer opens.
            others = [log for log in seen_logs if all(log is not judge for judge in judge_logs)]
            write_inspect_run_info(evaluations_path.parent, judge_logs + others)
            target = evaluations_path
        else:
            task = eigenbench(spec_ref, models=models) if models else eigenbench(spec_ref)
            planned_judgments = [sample.metadata for sample in task.dataset]
            logs = inspect_eval(tasks=task, model=None, **eval_kw)
            seen_logs.extend(logs)
            records, target = export_log(
                logs[0],
                evaluations_path=str(evaluations_path),
                cached_responses_path=collection_cfg.get("cached_responses_path"),
                failure_policy=failure_policy,
                planned=planned,
                report=False,
            )
    except BaseException as exc:  # interrupts too: report what ran, then stop
        record_refusal(evaluations_path.parent, "collection", exc, planned=planned_judgments,
                       logs=seen_logs, log_dir=log_dir, before=before,
                       failure_policy=failure_policy, evaluations_path=evaluations_path)
        raise
    record_collection(evaluations_path.parent, "collection", planned=planned_judgments,
                      logs=seen_logs, records=records, failure_policy=failure_policy,
                      evaluations_path=evaluations_path)
    write_fingerprint(evaluations_path, ctx)

    print(f"Direct collection complete. {len(records)} ratings saved to {target}")
    return records
