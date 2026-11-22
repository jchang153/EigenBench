"""Collect direct ratings with Inspect and export the judgments."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from secrets import randbits

from inspect_ai import eval as inspect_eval
from inspect_ai import eval_retry
from inspect_ai.log import EvalLog, read_eval_log

from inspect_pipeline.coverage import (
    _exported_key, evaluations_sha256, judgment_key, planned_from_assignments, record_collection,
)
from inspect_pipeline.provenance import (
    check_reusable, direct_resume_identity, read_resolved_sampler_seed, write_fingerprint,
)
from eigenbench.config import load_run_spec
from eigenbench.protocol.direct_rating import (
    DIRECT_SAMPLER_ALL_TO_ALL,
    DIRECT_SAMPLER_EXPLICIT,
    direct_response_keys,
    estimate_direct_calls,
    resolve_direct_generation_settings,
    resolve_direct_sampling_settings,
)
from eigenbench.protocol.response_cache import cached_seed
from eigenbench.records import load_records

from ..assignment_preview import preview_path
from ..context import _run_context_from_spec, direct_plan, has_local_models, load_selection, resolve_spec_ref
from .phased import eval_phase, model_phase, use_run_context
from inspect_pipeline.eigenbench import eigenbench
from inspect_pipeline.phased import eigenbench_judge, eigenbench_responses
from ..export import (
    append_response_updates,
    check_complete,
    export_log,
    records_from_log,
    records_from_logs,
    write_evaluations_atomic,
    write_inspect_run_info,
    write_omissions,
)
from ..models import model_execution
from ..generation import STORE_RESPONSE


def estimate_direct_calls_for_spec(spec: dict, run_dir: Path, *, preview: bool = False) -> dict:
    """Count planned generations offline, optionally saving the draw for review."""
    models, cfg = spec["models"], spec["collection"]
    include_self = bool((spec["evaluation"].get("direct_rating") or {}).get("include_self", True))
    resolve_direct_generation_settings(cfg)
    selected, criteria = load_selection(spec, run_dir)
    sampling = resolve_direct_sampling_settings(cfg, num_models=len(models), include_self=include_self)
    assignments, seed = direct_plan(spec, selected, criteria, models, include_self, sampling, preview=preview)
    cached = cached_seed(cfg.get("cached_responses_path"), selected, models, merge_rows=True)
    if seed is not None:
        sampling = {**sampling, "sampler_seed": seed}
    return _estimate_direct_plan(selected, models, include_self, sampling, assignments, cached)


def _estimate_direct_plan(selected, models, include_self, sampling, assignments, seed) -> dict:
    if sampling["sampler_mode"] == DIRECT_SAMPLER_EXPLICIT:
        needed = direct_response_keys(assignments)
        seed = {key: answer for key, answer in seed.items() if key in needed}
    routes = {nick: model_execution(value) for nick, value in models.items()}
    cached_routes = Counter(routes[nick] for _, nick in seed)
    openrouter_indices = {i for i, route in enumerate(routes.values()) if route == "openrouter"}
    other_indices = {i for i, route in enumerate(routes.values()) if route == "other"}
    estimate = {
        "mode": "direct_rating",
        **estimate_direct_calls(
            num_scenarios=len(selected),
            num_models=len(models),
            num_openrouter_models=len(openrouter_indices),
            include_self=include_self,
            cached_responses=len(seed),
            cached_openrouter_responses=cached_routes["openrouter"],
            assignments=assignments,
            openrouter_model_indices=openrouter_indices,
            other_model_indices=other_indices,
            cached_other_responses=cached_routes["other"],
            **sampling,
        ),
    }
    return estimate


def write_call_estimate(
    spec: dict, run_dir: Path, *, context: dict | None = None, resolved_sampler_seed: int | None = None,
) -> dict:
    if context is None:
        estimate = estimate_direct_calls_for_spec(spec, run_dir)
    else:
        sampling = dict(context["sampling"])
        if resolved_sampler_seed is not None:
            sampling["sampler_seed"] = resolved_sampler_seed
        estimate = _estimate_direct_plan(
            context["selected"], context["models"], context["include_self"],
            sampling, context["assignments"], context["seed"],
        )
    estimate_path = Path(spec["collection"]["evaluations_path"]).with_name(
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


def _persist_responses(ctx, responses, generated) -> None:
    """Append newly generated responses to the shared cache."""

    path = ctx["collection_cfg"].get("cached_responses_path")
    nicks = list(ctx["models"])
    explicit = ctx["sampling"]["sampler_mode"] == DIRECT_SAMPLER_EXPLICIT
    rows = [
        {"scenario": scenario, "scenario_index": int(s_idx),
         "responses": dict(responses[int(s_idx)])}
        for s_idx, scenario in ctx["selected"]
        if any((int(s_idx), nick) in generated for nick in nicks)
        and (explicit or all(nick in responses[int(s_idx)] for nick in nicks))
    ]
    if path and rows:
        if explicit:
            append_response_updates(path, rows)
        else:
            from eigenbench.records import append_records

            append_records(path, rows)


def _run_phased(spec_ref: str, models, ctx, eval_kw: dict, seen_logs: list | None = None) -> list[dict]:
    """Responses then judgments, one model resident at a time.

    A judgment needs both its evaluee and its judge, so an edge-per-sample run
    keeps every model live at once. That is fine for hosted models and fatal on
    one GPU, so local runs phase the work in response and judgment phases.
    Every log written, responses included, is added to ``seen_logs`` as it lands.
    """

    seen_logs = seen_logs if seen_logs is not None else []

    nicks = list(models)
    responses: dict[int, dict[str, str]] = defaultdict(dict)
    for (s_idx, nick), text in ctx["seed"].items():
        responses[s_idx][nick] = text

    judge_logs = []
    response_errors: dict[tuple[int, str], str] = {}
    needed_cells = direct_response_keys(ctx["assignments"])
    explicit = ctx["sampling"]["sampler_mode"] == DIRECT_SAMPLER_EXPLICIT

    print(f"Phase 1/2: responses, {len(nicks)} model(s), one at a time")
    generated: set[tuple[int, str]] = set()
    for nick in nicks:
        missing = [
            int(s) for s, _ in ctx["selected"]
            if nick not in responses[int(s)] and (not explicit or (s, nick) in needed_cells)
        ]
        if not missing:
            continue
        with model_phase(ctx, nick, reuse_resolver=True) as phase:
            with use_run_context(spec_ref, models, phase):
                task = eigenbench_responses(spec_ref, model_nick=nick, models=models, scenario_indices=missing)
            phase["resolve_model"](nick)
            log = eval_phase(task, eval_kw, seen_logs, require_complete=False)
        for sample in log.samples or []:
            content = (sample.store or {}).get(STORE_RESPONSE)
            if isinstance(content, str) and content.strip():
                responses[int(sample.metadata["scenario_index"])][nick] = content
                generated.add((int(sample.metadata["scenario_index"]), nick))
            elif sample.error is not None:
                response_errors[(int(sample.metadata["scenario_index"]), nick)] = sample.error.message
        if log.error is not None:
            for s in missing:
                response_errors.setdefault((int(s), nick), log.error.message)
    _persist_responses(ctx, responses, generated)

    # Every policy needs every response: a judgment cannot be made without one.
    needed = sorted((s, nick) for s, nick in needed_cells if nick not in responses[s])
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
        with model_phase(ctx, nick, reuse_resolver=True) as phase:
            with use_run_context(spec_ref, models, phase):
                task = eigenbench_judge(spec_ref, judge_nick=nick, edges=edges, models=models)
            phase["resolve_model"](nick)
            judge_logs.append(eval_phase(task, eval_kw, seen_logs, require_complete=False))

    return judge_logs


def _resume_judgments(log: EvalLog, task, ctx, spec_ref: str, existing: list[dict],
                      eval_kw: dict, seen_logs: list) -> list[EvalLog]:
    """Resume a hosted direct task without recomputing answers or valid judgments."""
    args = log.eval.task_args_passed or {}
    original_spec = args.get("spec")
    if (log.eval.task != task.name or log.eval.model != "none/none"
            or not isinstance(original_spec, str)
            or Path(resolve_spec_ref(original_spec)).resolve() != Path(resolve_spec_ref(spec_ref)).resolve()
            or args.get("models") is not None):
        raise ValueError("Resume requires an Inspect log from this spec's hosted direct-rating task")

    old_meta = (log.eval.metadata or {}).get("eigenbench", {})
    identity = ("criteria_sha256", "dataset", "panel", "include_self", "scale_min",
                "scale_max", "sampling", "generation")
    if any(key not in old_meta or old_meta[key] != ctx["meta"][key] for key in identity):
        raise ValueError("Cannot resume: the log's dataset, rubric, model panel or settings differ")
    if old_meta.get("assignments_sha256") != ctx["meta"].get("assignments_sha256"):
        raise ValueError("Cannot resume: the judgment plan changed")
    config = log.eval.config
    if (config.limit is not None or config.sample_id is not None or config.sample_shuffle
            or config.epochs not in (None, 1)):
        raise ValueError("Resume requires a log of the full judgment plan, not a sample selection")
    if config.log_samples is False or log.invalidated:
        raise ValueError("Resume requires a log with intact sample records")

    wanted = {sample.id: sample.metadata for sample in task.dataset}
    logged = set()
    done = set()
    for sample in log.samples or []:
        md = wanted.get(sample.id)
        if (md is None or sample.id in logged or sample.epoch != 1
                or judgment_key(sample.metadata) != judgment_key(md)
                or sample.metadata.get("scenario") != md["scenario"]):
            raise ValueError("Cannot resume: logged samples do not match the planned judgments")
        logged.add(sample.id)
        if sample.error is None:
            done.add(sample.id)

    prior = records_from_log(log, strict=False) if logged else []
    if len(prior) != len(done):
        raise ValueError("Cannot resume: a successful sample is missing judgment content")
    _require_retained_records(existing, prior)
    missing = wanted.keys() - done
    for sample_id in missing:
        md = wanted[sample_id]
        if (int(md["scenario_index"]), md["eval_nick"]) not in ctx["seed"]:
            raise ValueError(f"Cannot resume without the original cached answer for {sample_id}")
    for row in prior:
        key = (row["scenario_index"], row["evaluee"]["name"])
        if key in ctx["seed"] and ctx["seed"][key] != row["response"]:
            raise ValueError(f"Cannot resume: cached answer differs from the log at {key}")

    resume_identity = old_meta.get("resume_identity")
    if resume_identity is not None and resume_identity != direct_resume_identity(ctx):
        raise ValueError("Cannot resume: cached answers or judging prompts changed since this log")
    if not missing:
        seen_logs.append(log)
        return [log]
    if resume_identity is None:
        # Older logs cannot prove the contents of their original cache for unstarted samples.
        print("Warning: this log predates answer fingerprints; use its original frozen response cache.")
    print(f"Resuming {len(missing)}/{len(wanted)} judgments with Inspect; retaining {len(done)}")
    # The solver's attempt cache can contain the same invalid completions. Disable
    # it only for this retry; Inspect supplies completed samples from the original log.
    retry_log = log.model_copy(update={
        "eval": log.eval.model_copy(update={"task_args_passed": {**args, "cache": False}})
    })
    with use_run_context(original_spec, None, ctx):
        logs = eval_retry(retry_log, **eval_kw)
    seen_logs.extend(logs)
    _require_retained_records(prior, records_from_log(logs[0], strict=False))
    return logs


def _require_retained_records(prior: list[dict], current: list[dict]) -> None:
    by_key = {_exported_key(row): row for row in current}
    if any(by_key.get(_exported_key(row)) != row for row in prior):
        raise ValueError("Cannot resume: the Inspect log would discard or change existing judgments")


def collect_direct_ratings_inspect(
    spec_ref: str,
    *,
    models: dict[str, object] | None = None,
    analysis_status: dict | None = None,
    scoring: bool = True,
    resume_log: str | None = None,
) -> list[dict]:
    """Run collection for a run spec and write evaluations.jsonl."""

    if analysis_status is not None:
        analysis_status.clear()
    spec, run_dir = load_run_spec(resolve_spec_ref(spec_ref))
    collection_cfg = spec.get("collection", {})
    evaluations_path = Path(collection_cfg["evaluations_path"])
    inspect_cfg = collection_cfg.get("inspect", {}) or {}
    # The run context validates it on every collection path.
    failure_policy = collection_cfg.get("failure_policy", "strict")

    log_dir = Path(inspect_cfg.get("log_dir") or "inspect_logs")
    if not log_dir.is_absolute():
        log_dir = evaluations_path.parent / log_dir

    spec_models = models if models is not None else spec["models"]
    phased = bool(inspect_cfg.get("phased", has_local_models(spec_models)))
    if resume_log and (phased or models is not None):
        raise ValueError("Resuming from a log supports hosted direct ratings without model overrides")
    previous = read_eval_log(str(Path(resume_log).resolve())) if resume_log else None
    before_resume = evaluations_sha256(evaluations_path) if previous is not None else None
    has_output = evaluations_path.exists() and evaluations_path.stat().st_size > 0
    existing = load_records(str(evaluations_path)) if has_output else []
    resolved_seed = None
    if (collection_cfg.get("sampler_mode") not in (DIRECT_SAMPLER_ALL_TO_ALL, DIRECT_SAMPLER_EXPLICIT)
            and collection_cfg.get("sampler_seed") is None):
        if previous is not None:
            resolved_seed = (previous.eval.task_args_passed or {}).get("resolved_sampler_seed")
            if resolved_seed is None:
                resolved_seed = (previous.eval.metadata or {}).get("eigenbench", {}).get("resolved_sampler_seed")
            if type(resolved_seed) is not int:
                raise ValueError("Cannot resume: the log does not record the resolved sampler seed")
        elif has_output:
            resolved_seed = read_resolved_sampler_seed(evaluations_path)
        elif not preview_path(collection_cfg).exists():
            resolved_seed = randbits(64)
    ctx = _run_context_from_spec(spec, run_dir, models, resolved_sampler_seed=resolved_seed)
    resolved_seed = ctx["resolved_sampler_seed"]
    estimate = write_call_estimate(spec, run_dir, context=ctx, resolved_sampler_seed=resolved_seed)
    planned = estimate["rating_tasks"]
    # A completed run short-circuits only if it is this plan's collection; anything else
    # is refused with what differs.
    if has_output and previous is None:
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

    eval_kw = eval_kwargs(inspect_cfg, log_dir)
    if previous is not None:
        # An explicit resume is a request to fill the gaps, not omit them again.
        failure_policy = "strict"

    seen_logs: list = []
    planned_judgments = None
    before = log_names(log_dir)
    try:
        if phased:
            planned_judgments = planned_from_assignments(ctx["assignments"])
            judge_logs = _run_phased(spec_ref, spec_models, ctx, eval_kw, seen_logs)
            omissions: list[dict] = []
            records = records_from_logs(judge_logs, failure_policy=failure_policy, omissions=omissions)
            check_complete(planned=planned, records=records, omissions=omissions)
            write_evaluations_atomic(evaluations_path, records)
            write_omissions(evaluations_path, planned=planned,
                            records=records, omissions=omissions)
            # The full-panel score is primary; retain every inference log behind it.
            others = [log for log in seen_logs if all(log is not judge for judge in judge_logs)]
            panel_log = None
            if scoring:
                from ..metrics import finalize_panel_metrics
                panel_log = finalize_panel_metrics(ctx, logs=seen_logs, records=records,
                                                   planned=planned_judgments, eval_kwargs=eval_kw)
                if panel_log is not None:
                    seen_logs.append(panel_log)
            write_inspect_run_info(evaluations_path.parent,
                                   ([panel_log] if panel_log is not None else []) + judge_logs + others,
                                   evaluations=(evaluations_path, records))
            target = evaluations_path
        else:
            with use_run_context(spec_ref, models, ctx):
                task = eigenbench(spec_ref, models=models, panel_metrics=scoring,
                                  write_artifacts=bool(spec.get("training", {}).get("enabled", True)),
                                  resolved_sampler_seed=resolved_seed)
            planned_judgments = [sample.metadata for sample in task.dataset]
            if previous is not None:
                logs = _resume_judgments(previous, task, ctx, spec_ref, existing, eval_kw, seen_logs)
                if evaluations_sha256(evaluations_path) != before_resume:
                    raise RuntimeError("Saved judgments changed during the retry; nothing overwritten")
            else:
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
        if previous is not None and evaluations_path.exists():
            # A refused retry did not replace the existing export or its provenance.
            for log in seen_logs:
                print(f"Retry did not replace the saved output. Inspect log: {log.location}")
        else:
            record_refusal(evaluations_path.parent, "collection", exc, planned=planned_judgments,
                           logs=seen_logs, log_dir=log_dir, before=before,
                           failure_policy=failure_policy, evaluations_path=evaluations_path)
        raise
    record_collection(evaluations_path.parent, "collection", planned=planned_judgments,
                      logs=seen_logs, records=records, failure_policy=failure_policy,
                      evaluations_path=evaluations_path)
    write_fingerprint(evaluations_path, ctx, resolved_sampler_seed=resolved_seed)
    from ..metrics import record_analysis_status
    if scoring:
        analysis_log = panel_log if phased else logs[0]
        if analysis_log is not previous:
            record_analysis_status(analysis_log, analysis_status)

    print(f"Direct collection complete. {len(records)} ratings saved to {target}")
    return records
