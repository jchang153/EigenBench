"""Collection coverage: every planned judgment against every Inspect log and the export.

One versioned report per run (collection_report.json). Collection writes it from the
plan and every log; a run collected before the report existed gets one built from its
logs alone, with planned and unstarted unknown (null).
"""
from __future__ import annotations

import json
from pathlib import Path


REPORT_FILE = "collection_report.json"
SCHEMA_VERSION = 2
COUNTS = ("planned", "logged", "exported", "failed", "omitted", "unstarted")


def judgment_key(md: dict):
    """The judgment a sample, plan entry or edge stands for, or None if it is not one."""

    if "judge_nick" not in md:
        return None
    if "eval_nick" in md:
        return ("direct", int(md["scenario_index"]), md["judge_nick"], md["eval_nick"],
                int(md.get("sampling_round", 0)))
    if "eval1_nick" in md and "eval2_nick" in md:
        return ("pairwise", int(md["scenario_index"]), md["judge_nick"], md["eval1_nick"],
                md["eval2_nick"])
    return None


def _exported_key(row: dict):
    """The judgment an exported record holds, in judgment_key's form."""

    if row.get("record_type") == "direct_rating":
        return ("direct", int(row["scenario_index"]), row.get("judge", {}).get("name"),
                row.get("evaluee", {}).get("name"), int(row.get("sampling", {}).get("round", 0) or 0))
    if "eval1_name" in row and "eval2_name" in row:
        return ("pairwise", int(row["scenario_index"]), row.get("judge_name"),
                row["eval1_name"], row["eval2_name"])
    return None


def planned_from_assignments(assignments: list[dict]) -> list[dict]:
    """Plan entries, one per judgment, from direct-rating assignments."""

    return [
        {"scenario_index": a["scenario_index"], "judge_nick": a["judge_nick"], "eval_nick": nick,
         "sampling_round": a.get("sampling_round", 0)}
        for a in assignments
        for nick in a["eval_nicks"]
    ]


def planned_from_edges(edges) -> list[dict]:
    """Plan entries from extension edges; pairwise edges carry an opponent."""

    return [
        {"scenario_index": e.scenario_index, "judge_nick": e.judge, "eval1_nick": e.evaluee,
         "eval2_nick": e.opponent}
        if e.opponent is not None else
        {"scenario_index": e.scenario_index, "judge_nick": e.judge, "eval_nick": e.evaluee}
        for e in edges
    ]


def _log_entry(log) -> dict:
    task_md = getattr(log.eval, "metadata", None) or {}
    samples = log.samples or []
    responses = task_md.get("eigenbench_phase") == "response"
    return {
        "file": Path(str(log.location)).name,
        "role": "responses" if responses else "judgments",
        "model": task_md.get("evaluee") if responses else (task_md.get("eigenbench") or {}).get("judge"),
        "status": log.status,
        "samples": len(samples),
        "failed": sum(sample.error is not None for sample in samples),
    }


def collection_part(kind: str, *, planned: list[dict] | None, logs, records: list[dict],
                    status: str = "exported", error: str | None = None) -> dict:
    """Coverage of one collection or extension: its plan against every log it wrote."""

    exported_keys = {_exported_key(row) for row in records} - {None}
    logged: dict = {}
    for log in logs:
        log_file = Path(str(log.location)).name
        for sample in log.samples or []:
            key = judgment_key(sample.metadata or {})
            if key is None:
                continue  # Response samples are not judgments.
            if key in logged:
                raise ValueError("Repeated judgment keys in Inspect logs; coverage is ambiguous")
            logged[key] = (sample, log_file)

    omissions = []
    for key, (sample, log_file) in logged.items():
        if key in exported_keys:
            continue
        md = sample.metadata or {}
        omissions.append({
            "sample_id": str(sample.id),
            "log_file": log_file,
            "scenario_index": md.get("scenario_index"),
            "scenario": md.get("scenario", ""),
            "judge": md["judge_nick"],
            "models": list(key[3:4] if key[0] == "direct" else key[3:5]),
            "reason": "sample_error" if sample.error is not None else "not_exported",
            # Messages only: don't publish stack traces or request headers.
            "error": sample.error.message[:2000] if sample.error is not None else None,
        })

    unstarted = None
    if planned is not None:
        unstarted = []
        for entry in planned:
            key = judgment_key(entry)
            if key is not None and key not in logged:
                unstarted.append({
                    "scenario_index": key[1],
                    "judge": key[2],
                    "models": list(key[3:4] if key[0] == "direct" else key[3:5]),
                })

    return {
        "kind": kind,
        "status": status,
        "error": error[:2000] if error else None,
        "planned": len(planned) if planned is not None else None,
        "logged": len(logged),
        "exported": len(logged) - len(omissions),
        "failed": sum(sample.error is not None for sample, _ in logged.values()),
        "omitted": len(omissions),
        "unstarted": len(unstarted) if unstarted is not None else None,
        "logs": [_log_entry(log) for log in logs],
        "omissions": omissions,
        "unstarted_judgments": unstarted,
    }


def _total(parts: list[dict], name: str):
    values = [part[name] for part in parts]
    # A part with unknown coverage makes the run's total unknown, never zero.
    return None if any(value is None for value in values) else sum(values)


def build_report(parts: list[dict], *, failure_policy: str | None) -> dict:
    """The version 2 report for a run made of these collections and extensions."""

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "inspect_logs",
        "failure_policy": failure_policy,
        **{name: _total(parts, name) for name in COUNTS},
        "collections": parts,
    }


def write_collection_report(run_dir: str | Path, part: dict, *, failure_policy: str,
                            append: bool = False) -> Path:
    """Write the run's report, adding ``part`` to it when ``append`` (an extension)."""

    path = Path(run_dir) / REPORT_FILE
    parts = []
    if append:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if existing and existing.get("schema_version") == SCHEMA_VERSION:
            parts = existing["collections"]
        else:
            # The run predates this report, so what it collected first is unknown.
            parts = [{"kind": "collection", "status": "unknown", "error": None,
                      **{name: None for name in COUNTS}, "logs": [], "omissions": [],
                      "unstarted_judgments": None}]
    report = build_report(parts + [part], failure_policy=failure_policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    info_path = Path(run_dir) / "inspect_run.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info["collection_report_file"] = REPORT_FILE
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return path


def record_collection(run_dir: str | Path, kind: str, *, planned, logs, records, failure_policy: str,
                      status: str = "exported", error: str | None = None) -> Path | None:
    """Write the report, never letting a reporting problem hide the run's own outcome."""

    try:
        part = collection_part(kind, planned=planned, logs=logs, records=records,
                               status=status, error=error)
        path = write_collection_report(run_dir, part, failure_policy=failure_policy,
                                       append=kind == "extension")
    except Exception as exc:  # e.g. ambiguous keys; the run itself is unaffected
        print(f"Warning: collection report unavailable: {exc}", flush=True)
        return None
    print(f"Collection report: planned {part['planned']}, logged {part['logged']}, exported "
          f"{part['exported']}, failed {part['failed']}, omitted {part['omitted']}, unstarted "
          f"{part['unstarted']} -> {path.name}", flush=True)
    return path


def report_from_run_dir(run_dir: str | Path, evaluations_path: str | Path | None = None) -> dict:
    """A version 2 report for a run that has no collection_report.json.

    Reads every log in inspect_run.json (``log_files``, else ``log_file``) and the exported
    evaluations. Without the plan, planned and unstarted are unknown (null), never 0.
    """

    from inspect_ai.log import read_eval_log

    run_dir = Path(run_dir)
    info = json.loads((run_dir / "inspect_run.json").read_text(encoding="utf-8"))
    names = info.get("log_files") or ([info["log_file"]] if info.get("log_file") else [])
    if not names:
        raise ValueError("inspect_run.json names no logs")
    logs = [read_eval_log(str(run_dir / "inspect_logs" / name)) for name in names]
    evaluations = Path(evaluations_path or run_dir / "evaluations.jsonl")
    with evaluations.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    policies = {
        ((getattr(log.eval, "metadata", None) or {}).get("eigenbench") or {}).get("failure_policy")
        for log in logs
    } - {None}
    part = collection_part("collection", planned=None, logs=logs, records=records)
    return build_report([part], failure_policy=policies.pop() if len(policies) == 1 else None)
