"""Convert an Inspect eval log into the legacy ``evaluations.jsonl`` contract.

Record construction mirrors ``pipeline/eval/direct_rating.py`` verbatim so that
``pipeline.trust.direct_rating.aggregate_direct_records`` and the ValueArena
Space consume Inspect-collected output unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from inspect_ai.log import EvalLog, read_eval_log

from pipeline.eval.direct_rating import DIRECT_SAMPLER_ALL_TO_ALL, parse_direct_ratings

from .phases import (
    FAILURE_POLICIES,
    STORE_JUDGMENT_RAW,
    STORE_REFLECTION,
    STORE_RESPONSE,
    InvalidJudgmentError,
)


def records_fingerprint(records: list[dict]) -> str:
    """Hash logical records, independent of JSON whitespace/key ordering."""
    encoded = json.dumps(records, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_log(log: str | Path | EvalLog) -> EvalLog:
    if isinstance(log, EvalLog):
        return log
    return read_eval_log(str(log))


def eigenbench_metadata(log: EvalLog) -> dict:
    metadata = (log.eval.metadata or {}).get("eigenbench")
    if not metadata:
        raise ValueError(
            "log is not an EigenBench eval (no 'eigenbench' task metadata); "
            "was it produced by inspect_pipeline/eigenbench.py?"
        )
    return metadata


def is_invalid_judgment(error) -> bool:
    """Whether a sample error is a judgment still invalid after every attempt."""

    name = InvalidJudgmentError.__name__
    if error.message.startswith(f"{name}("):
        return True
    lines = [line for line in (error.traceback or "").splitlines() if line.strip()]
    return bool(lines) and name in lines[-1]


def _sample_label(sample) -> str:
    md = sample.metadata or {}
    return (
        f"scenario_index={md.get('scenario_index')} judge={md.get('judge_nick')} "
        f"evaluee={md.get('eval_nick')}"
    )


def _failure_message(failed, failure_policy: str) -> str:
    models = sorted({
        nick
        for sample in failed
        for nick in ((sample.metadata or {}).get("judge_nick"), (sample.metadata or {}).get("eval_nick"))
        if nick
    })
    shown = "\n  ".join(f"{_sample_label(sample)}: {sample.error.message}" for sample in failed[:10])
    more = f"\n  ... and {len(failed) - 10} more" if len(failed) > 10 else ""
    if failure_policy == "omit_invalid_judgments":
        what = f"{len(failed)} sample(s) failed for reasons other than an invalid judgment"
    else:
        what = f"{len(failed)} sample(s) failed (collection.failure_policy={failure_policy})"
    return (
        f"{what}; nothing exported. Models involved: {', '.join(models) or 'unknown'}\n"
        f"  {shown}{more}\nResume with: inspect eval-retry <log>"
    )


def _omission(sample) -> dict:
    md = sample.metadata or {}
    message = sample.error.message
    return {
        "scenario_index": md.get("scenario_index"),
        "judge": md.get("judge_nick"),
        "evaluee": md.get("eval_nick"),
        "stage": "reflection" if "(reflection " in message else "direct_rating",
        "reason": "invalid_response_after_retries" if is_invalid_judgment(sample.error) else "sample_error",
        "sample_id": str(sample.id),
        "error": message[:2000],
    }


def records_from_log(
    log: str | Path | EvalLog,
    *,
    strict: bool = True,
    failure_policy: str = "strict",
    omissions: list[dict] | None = None,
) -> list[dict]:
    """Build direct-rating records from an eval log, in plan (edge) order.

    With ``failure_policy="omit_invalid_judgments"`` a judgment that stayed invalid after
    every attempt is left out and added to ``omissions``; any other failed sample still
    refuses the export and names its models. ``strict=False`` skips every failed sample.
    """

    if failure_policy not in FAILURE_POLICIES:
        raise ValueError(f"Unknown failure_policy {failure_policy!r}")
    log = load_log(log)
    meta = eigenbench_metadata(log)
    criteria: list[str] = meta["criteria"]
    criteria_text = "\n".join(criteria)
    scale_min = int(meta.get("scale_min", 1))
    scale_max = int(meta.get("scale_max", 10))

    if strict and log.status != "success":
        message = log.error.message if log.error else f"status={log.status}"
        raise RuntimeError(f"eval log did not complete successfully: {message}")

    samples = log.samples or []
    if not samples:
        raise RuntimeError("eval log contains no samples")

    failed = [sample for sample in samples if sample.error is not None]
    if strict:
        fatal = [
            sample
            for sample in failed
            if not (failure_policy == "omit_invalid_judgments" and is_invalid_judgment(sample.error))
        ]
        if fatal:
            raise RuntimeError(_failure_message(fatal, failure_policy))
        if failed and len(failed) == len(samples):
            raise RuntimeError(
                f"every judgment in the log was invalid; nothing exported\n  "
                f"{_sample_label(failed[0])}: {failed[0].error.message}"
            )

    rows = []
    seen_edges: set[tuple[int, int, int]] = set()
    for sample in samples:
        if sample.error is not None:
            if omissions is not None:
                omissions.append(_omission(sample))
            continue
        md = sample.metadata or {}
        store = sample.store or {}
        response = store.get(STORE_RESPONSE)
        reflection = store.get(STORE_REFLECTION)
        raw_rating = store.get(STORE_JUDGMENT_RAW)
        if not all(isinstance(v, str) for v in (response, reflection, raw_rating)):
            if strict:
                raise RuntimeError(f"sample {sample.id} is missing generated content")
            continue

        parsed = parse_direct_ratings(
            raw_rating,
            num_criteria=len(criteria),
            scale_min=scale_min,
            scale_max=scale_max,
        )
        edge = (int(md["scenario_index"]), int(md["judge_idx"]), int(md["eval_idx"]))
        if edge in seen_edges:
            raise RuntimeError(f"duplicate directed edge in log: {edge}")
        seen_edges.add(edge)

        rows.append(
            (
                int(md.get("edge_index", len(rows))),
                {
                    "schema_version": 2,
                    "record_type": "direct_rating",
                    "constitution": criteria_text,
                    "scenario": md["scenario"],
                    "scenario_index": int(md["scenario_index"]),
                    "judge": {"index": int(md["judge_idx"]), "name": md["judge_nick"]},
                    "evaluee": {"index": int(md["eval_idx"]), "name": md["eval_nick"]},
                    "sampling": {
                        "mode": md.get("sampler_mode") or DIRECT_SAMPLER_ALL_TO_ALL,
                        "round": int(md.get("sampling_round", 0)),
                        "group_index": int(md.get("group_index", 0)),
                    },
                    "response": response,
                    "reflection": reflection,
                    "judgment_raw": raw_rating,
                    "ratings": [
                        {
                            "criterion_index": criterion_idx,
                            "criterion": criteria[criterion_idx],
                            "rating": value,
                        }
                        for criterion_idx, value in parsed.items()
                    ],
                },
            )
        )

    rows.sort(key=lambda row: row[0])
    return [record for _index, record in rows]


def records_from_logs(
    logs,
    *,
    strict: bool = True,
    failure_policy: str = "strict",
    omissions: list[dict] | None = None,
) -> list[dict]:
    """Merge the judge logs of a phased run back into one ordered record set."""

    records: list[dict] = []
    for log in logs:
        records.extend(
            records_from_log(log, strict=strict, failure_policy=failure_policy, omissions=omissions)
        )
    seen = set()
    for r in records:
        edge = (r["scenario_index"], r["judge"]["index"], r["evaluee"]["index"])
        if edge in seen:
            raise RuntimeError(f"duplicate directed edge across logs: {edge}")
        seen.add(edge)
    records.sort(
        key=lambda r: (r["scenario_index"], r["judge"]["index"], r["evaluee"]["index"])
    )
    return records


def responses_from_log(log: str | Path | EvalLog) -> list[dict]:
    """Per-scenario response maps, for the shared response cache."""

    log = load_log(log)
    meta = eigenbench_metadata(log)
    model_order = list(meta["model_order"])

    scenarios: dict[int, str] = {}
    responses: dict[int, dict[str, str]] = {}
    for sample in log.samples or []:
        if sample.error is not None:
            continue
        md = sample.metadata or {}
        content = (sample.store or {}).get(STORE_RESPONSE)
        if not isinstance(content, str):
            continue
        s_idx = int(md["scenario_index"])
        scenarios.setdefault(s_idx, md["scenario"])
        responses.setdefault(s_idx, {})[md["eval_nick"]] = content

    rows = []
    for s_idx in sorted(responses):
        complete = responses[s_idx]
        if all(nick in complete for nick in model_order):
            rows.append(
                {
                    "scenario": scenarios[s_idx],
                    "scenario_index": s_idx,
                    "responses": {nick: complete[nick] for nick in model_order},
                }
            )
    return rows


def write_inspect_run_info(run_dir: str | Path, log: EvalLog | list) -> Path:
    """Record which log(s) produced a run, so the upload can carry them."""

    path = Path(run_dir) / "inspect_run.json"
    info = {}
    if path.exists():
        info = json.loads(path.read_text(encoding="utf-8"))
    logs = log if isinstance(log, list) else [log]
    names = [Path(str(item.location)).name for item in logs]
    info["log_file"] = names[0]
    if len(names) > 1:
        info["log_files"] = names
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return path


def check_complete(*, planned: int, records: list[dict], omissions: list[dict]) -> None:
    """Refuse an export that does not account for every planned judgment."""

    if len(records) + len(omissions) != planned:
        raise RuntimeError(
            f"incomplete run: {planned} judgment(s) planned, but the logs hold {len(records)} "
            f"exported and {len(omissions)} omitted; nothing exported"
        )


def write_omissions(evaluations_path: str | Path, *, planned: int, records: list[dict],
                    omissions: list[dict]) -> Path:
    """Write omitted_samples.json beside the evaluations, in the native engine's format.

    Written for every policy, so a strict run reports its planned count too and replaces
    any file left by an earlier run.
    """

    path = Path(evaluations_path).with_name("omitted_samples.json")
    payload = {"planned": planned, "completed": len(records), "omitted": len(omissions),
               "samples": omissions}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Judgment coverage: {len(records)}/{planned}; omitted {len(omissions)}", flush=True)
    for row in omissions[:20]:
        print(f"  [Omitted] scenario_index={row['scenario_index']} judge={row['judge']} "
              f"evaluee={row['evaluee']} ({row['stage']}): {row['error'][:200]}", flush=True)
    if len(omissions) > 20:
        print(f"  ... and {len(omissions) - 20} more in {path.name}", flush=True)
    return path


def write_evaluations_atomic(path: str | Path, records: list[dict]) -> None:
    """Write jsonl with the same serialization as pipeline.utils.transcripts."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        os.replace(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def export_log(
    log: str | Path | EvalLog,
    *,
    evaluations_path: str | Path | None = None,
    cached_responses_path: str | Path | None = None,
    strict: bool = True,
    append: bool = False,
    failure_policy: str | None = None,
    planned: int | None = None,
) -> tuple[list[dict], Path]:
    """Export one eval log to evaluations.jsonl (+ optional response cache).

    ``append`` keeps whatever is already there, which is what a run extended
    with a new model needs. A pairwise log is written in the pairwise schema.
    ``failure_policy`` defaults to the run's collection.failure_policy. A direct-rating
    export always writes omitted_samples.json, with no omissions under ``strict``. Given
    ``planned``, a log that does not account for every judgment is refused.
    """

    log = load_log(log)
    meta = eigenbench_metadata(log)
    target = Path(evaluations_path or meta.get("evaluations_path") or "")
    if not str(target):
        raise ValueError(
            "no evaluations path: pass evaluations_path or set "
            "collection.evaluations_path in the run spec"
        )

    if meta.get("extension_phased"):
        raise ValueError(
            "a phased extension log contains only one judge; rerun scripts/extend_run.py "
            "to collect and combine the complete extension"
        )

    omissions: list[dict] = []
    direct = True
    # A pairwise extension log carries comparisons, not single ratings.
    if any("eval1_nick" in (s.metadata or {}) for s in (log.samples or [])):
        from .pairwise import records_from_pairwise_log

        records = records_from_pairwise_log(log, meta["criteria"], strict=strict)
        failure_policy = "strict"
        direct = False
    else:
        # Extensions are checked against their full plan below, so they stay strict.
        if failure_policy is None:
            failure_policy = "strict" if meta.get("extension_source") else meta.get("failure_policy", "strict")
        records = records_from_log(log, strict=strict, failure_policy=failure_policy, omissions=omissions)

    if meta.get("extension_source") and not append:
        raise ValueError("extension logs require --append to preserve the source records")
    if append:
        from pipeline.utils import load_records

        source = Path(meta["extension_source"]) if meta.get("extension_source") else None
        if source:
            if not source.is_file():
                raise ValueError(f"extension source is missing: {source}; nothing exported")
            source_records = load_records(str(source))
            fingerprint = meta.get("extension_source_sha256")
            if not fingerprint:
                raise ValueError("extension log has no source fingerprint; cannot verify baseline")
            if records_fingerprint(source_records) != fingerprint:
                raise ValueError("extension source changed since planning; nothing exported")
            expected = meta.get("extension_expected_edges")
            if strict and expected is not None and len(records) != expected:
                raise ValueError("extension log does not contain the complete plan; nothing exported")
            if target.resolve() != source.resolve() and target.exists():
                raise ValueError("extension output already exists; choose a new output path")
            existing = source_records
        else:
            existing = load_records(str(target)) if target.exists() else []
        def identity(r):
            if r.get("record_type") == "direct_rating":
                return (r["scenario_index"], r["judge"]["index"], r["evaluee"]["index"])
            return (r["scenario_index"], r["judge"], r["eval1"], r["eval2"])
        existing_ids = {identity(r) for r in existing}
        if any(identity(r) in existing_ids for r in records):
            raise ValueError("extension records already exist in the destination")
        write_evaluations_atomic(target, existing + records)
    else:
        if planned is not None:
            check_complete(planned=planned, records=records, omissions=omissions)
        write_evaluations_atomic(target, records)
        if direct:
            write_omissions(target, planned=planned or len(log.samples or []), records=records,
                            omissions=omissions)
    write_inspect_run_info(target.parent, log)

    cache_target = cached_responses_path or meta.get("cached_responses_path")
    if cache_target:
        from pipeline.utils import append_records, load_records

        existing = {
            int(row["scenario_index"]): row.get("responses")
            for row in load_records(str(cache_target))
            if isinstance(row, dict) and "scenario_index" in row
        }
        new_rows = [
            row
            for row in responses_from_log(log)
            if existing.get(row["scenario_index"]) != row["responses"]
        ]
        if new_rows:
            append_records(str(cache_target), new_rows)

    return records, target
