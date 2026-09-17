"""Compare a published Inspect log with the judgments actually exported."""
from __future__ import annotations

import json
from pathlib import Path


def _record_key(row: dict):
    if row.get("record_type") == "direct_rating":
        return ("direct", row.get("scenario_index"), row.get("judge", {}).get("name"),
                row.get("evaluee", {}).get("name"), row.get("sampling", {}).get("round", 0))
    if "eval1_name" in row and "eval2_name" in row:
        return ("pairwise", row.get("scenario_index"), row.get("judge_name"),
                row["eval1_name"], row["eval2_name"])
    return None


def report_from_samples(samples, records: list[dict], *, log_file: str) -> dict:
    exported_keys = {_record_key(row) for row in records} - {None}
    logged_keys = set()
    omitted = []
    for sample in samples:
        md = sample.metadata or {}
        if "judge_nick" not in md:
            continue  # Response-only phases are not judgment samples.
        if "eval_nick" in md:
            key = ("direct", md.get("scenario_index"), md["judge_nick"],
                   md["eval_nick"], md.get("sampling_round", 0))
            models = [md["eval_nick"]]
        elif "eval1_nick" in md and "eval2_nick" in md:
            key = ("pairwise", md.get("scenario_index"), md["judge_nick"],
                   md["eval1_nick"], md["eval2_nick"])
            models = [md["eval1_nick"], md["eval2_nick"]]
        else:
            continue
        if key in logged_keys:
            raise ValueError("Repeated judgment keys in Inspect log; omission report is ambiguous")
        logged_keys.add(key)
        if key not in exported_keys:
            omitted.append({
                "sample_id": str(sample.id),
                "scenario_index": md.get("scenario_index"),
                "scenario": md.get("scenario", ""),
                "judge": md["judge_nick"],
                "models": models,
                "reason": "sample_error" if sample.error is not None else "not_exported",
                # Messages only: don't publish stack traces or request headers.
                "error": sample.error.message[:2000] if sample.error is not None else None,
            })
    if not logged_keys:
        raise ValueError("No supported judgment samples in Inspect log")
    return {
        "schema_version": 1,
        "source": "inspect_log",
        "log_file": log_file,
        "logged_samples": len(logged_keys),
        "exported_samples": len(logged_keys & exported_keys),
        "omitted_samples": len(omitted),
        "failed_samples": sum(row["reason"] == "sample_error" for row in omitted),
        "omissions": omitted,
    }


def build_collection_report(log_path: Path, evaluations_path: Path) -> dict:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    with evaluations_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return report_from_samples(log.samples or [], records, log_file=log_path.name)
