"""Collect and export new judgments for an existing run."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.protocol.extension import (
    DIRECT, PAIRWISE, Plan, detect_mode, model_names, plan_additions, scenarios_of,
)

from ..phases import direct_rating_scorer, direct_rating_solver


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

    from ..context import _run_context
    from ..eigenbench import _samples_view
    from ..generation import ResponsePool

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


def _source_model_identities(source: Path, records: list[dict], mode: str, *,
                             log_dir: str | Path | None = None) -> dict[str, dict]:
    """Read known contributors without inferring identities from a historical spec."""
    from inspect_ai.log import read_eval_log

    from inspect_pipeline.coverage import REPORT_FILE, evaluations_sha256
    from inspect_pipeline.provenance import FINGERPRINT_FILE
    from ..export import panel_model_identities, records_fingerprint

    manifest = source.with_name("inspect_run.json")
    if manifest.is_file():
        provenance = json.loads(manifest.read_text(encoding="utf-8")).get("model_provenance", {})
        if source.name in provenance:
            entry = provenance[source.name]
            if entry["records_sha256"] != records_fingerprint(records):
                raise ValueError(f"source records no longer match their model provenance: {source}")
            return entry["models"]

    # Older collection reports link exported rows to their judgment logs.
    report_path = source.with_name(REPORT_FILE)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    linked = report is not None and report.get("evaluations_sha256") == evaluations_sha256(source)
    if linked:
        directories = [source.parent / "inspect_logs"]
        if log_dir:
            directories.insert(0, source.parent / log_dir)
        for collection in report.get("collections", []):
            if collection["kind"] != "collection" or collection["status"] != "exported":
                continue
            for entry in collection["logs"]:
                if entry["role"] != "judgments" or entry["status"] != "success":
                    continue
                for directory in dict.fromkeys(directories):
                    path = directory / Path(entry["file"]).name
                    if not path.is_file():
                        continue
                    log = read_eval_log(path, header_only=True)
                    meta = (log.eval.metadata or {}).get("eigenbench", {})
                    if log.status == "success" and meta.get("panel") and not meta.get("extension_source"):
                        return panel_model_identities(meta["panel"])

    # Older direct runs also stored the original panel in their collection fingerprint.
    fingerprint = source.with_name(FINGERPRINT_FILE)
    legacy_path = report is None and source.name == "evaluations.jsonl"
    if mode == DIRECT and fingerprint.is_file() and (linked or legacy_path):
        identity = json.loads(fingerprint.read_text(encoding="utf-8")).get("identity", {})
        return panel_model_identities(identity.get("models", []))
    return {}


def _prepare_extension(
    spec_ref, new_models=None, *, additional_scenarios=None, include_self=None, seed=42,
):
    from pipeline.config import load_run_spec
    from pipeline.records import load_records

    from inspect_pipeline.provenance import changed_scenarios, constitution_differs
    from ..context import load_selection, resolve_spec_ref
    from ..models import model_identity

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
    inspect_cfg = spec["collection"].get("inspect") or {}
    known = _source_model_identities(source, records, mode, log_dir=inspect_cfg.get("log_dir"))
    identities = {
        nick: json.loads(json.dumps(model_identity(model), default=str))
        for nick, model in models.items()
    }
    changed = [nick for nick in order if nick in known and known[nick] != identities[nick]]
    if changed:
        raise ValueError(f"extension model references changed for existing nicks: {', '.join(changed)}; "
                         "retain the source model or use a new nickname")
    selected, criteria = load_selection(spec, run_dir)
    if constitution_differs(records, criteria):
        raise ValueError("extension criteria differ from the source records")
    old_scenarios = scenarios_of(records)
    if changed_scenarios(records, selected):
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
    # Keep source evidence and record only models that contribute during this extension.
    active = {edge.judge for edge in plan.edges}
    for edge in plan.edges:
        for nick, response in ((edge.evaluee, edge.response), (edge.opponent, edge.opponent_response)):
            if nick is not None and (not isinstance(response, str) or not response.strip()):
                active.add(nick)
    return {
        "spec_ref": spec_ref, "spec": spec, "records": records, "models": models,
        "order": order, "plan": plan, "source": source, "output": output,
        "scenarios": {**old_scenarios, **extra},
        "model_identities": {nick: identities[nick] for nick in models if nick in known or nick in active},
    }


def _extension_task(prepared, *, context=None, task_label=None):
    plan = prepared["plan"]
    factory = eigenbench_extend
    if plan.mode == PAIRWISE:
        from ..pairwise import eigenbench_pairwise_extend
        factory = eigenbench_pairwise_extend
    result = factory(
        prepared["spec_ref"], new_model=task_label or plan.new_model, plan=plan,
        scenarios=prepared["scenarios"], order=prepared["order"],
        models=prepared["models"], context=context,
    )
    from ..export import records_fingerprint
    result.metadata["eigenbench"]["extension_source"] = str(prepared["source"])
    result.metadata["eigenbench"]["extension_source_sha256"] = records_fingerprint(prepared["records"])
    result.metadata["eigenbench"]["extension_model_identities"] = prepared["model_identities"]
    result.metadata["eigenbench"]["extension_expected_edges"] = len(plan.edges)
    result.metadata["eigenbench"]["extends_with"] = plan.new_models
    result.metadata["eigenbench"]["additional_scenarios"] = list(plan.new_scenarios)
    return result


def _use_phased_extension(prepared) -> bool:
    from ..context import has_local_models
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

    from inspect_ai import eval as inspect_eval

    from ..coverage import planned_from_edges, record_collection
    from ..export import records_from_log, write_evaluations_atomic, write_inspect_run_info
    from .direct_rating import eval_kwargs, log_names, record_refusal

    inspect_cfg = spec["collection"].get("inspect", {}) or {}
    log_dir = Path(inspect_cfg.get("log_dir") or "inspect_logs")
    if not log_dir.is_absolute():
        log_dir = prepared["output"].parent / log_dir
    run_dir = prepared["output"].parent
    planned = planned_from_edges(plan.edges)
    failure_policy = spec["collection"].get("failure_policy", "strict")
    seen_logs: list = []
    before = log_names(log_dir)
    try:
        if _use_phased_extension(prepared):
            from .extension_phased import collect_phased_extension
            new_records, logs = collect_phased_extension(
                prepared, eval_kwargs(inspect_cfg, log_dir), seen_logs,
            )
        else:
            from ..models import check_distinct_local_servers
            check_distinct_local_servers(prepared["models"])
            task = _extension_task(prepared)
            logs = inspect_eval(tasks=task, model=None, **eval_kwargs(inspect_cfg, log_dir))
            seen_logs.extend(logs)
            if plan.mode == DIRECT:
                new_records = records_from_log(logs[0])
            else:
                from ..pairwise import records_from_pairwise_log
                new_records = records_from_pairwise_log(
                    logs[0], task.metadata["eigenbench"]["criteria"]
                )
        if len(new_records) != len(plan.edges):
            raise RuntimeError("extension did not collect the complete plan; nothing appended")
        # Never overwrite concurrent changes to the source or an independently-created output.
        from pipeline.records import load_records
        if load_records(str(prepared["source"])) != prepared["records"]:
            raise RuntimeError("source records changed during collection; export the log manually")
        if prepared["output"] != prepared["source"] and prepared["output"].exists():
            raise RuntimeError("output appeared during collection; export the log manually")
        combined = prepared["records"] + new_records
        write_evaluations_atomic(prepared["output"], combined)
    except BaseException as exc:  # interrupts too: report what ran, then stop
        record_refusal(run_dir, "extension", exc, planned=planned, logs=seen_logs,
                       log_dir=log_dir, before=before, failure_policy=failure_policy,
                       evaluations_path=prepared["output"])
        raise
    # Judgment logs first: the first is the primary log the viewer opens.
    others = [log for log in seen_logs if all(log is not judged for judged in logs)]
    write_inspect_run_info(run_dir, list(logs) + others, append=True,
                           evaluations=(prepared["output"], combined))
    record_collection(run_dir, "extension", planned=planned, logs=seen_logs, records=new_records,
                      failure_policy=failure_policy, evaluations_path=prepared["output"])
    missing = [name for name in plan.new_models if name not in spec["models"]]
    if missing:
        print(f"Add these models to the spec before analysis: {missing}")
    print(f"Added {len(new_records)} judgments to {prepared['output']}")
    return {**summary, "collected": len(new_records)}
