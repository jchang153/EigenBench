"""Convert an Inspect eval log into the legacy ``evaluations.jsonl`` contract.

Record construction mirrors ``pipeline/eval/direct_rating.py`` verbatim so that
``pipeline.trust.direct_rating.aggregate_direct_records`` and the ValueArena
Space consume Inspect-collected output unchanged.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from inspect_ai.log import EvalLog, read_eval_log

from pipeline.eval.direct_rating import DIRECT_SAMPLER_ALL_TO_ALL, parse_direct_ratings

from .phases import STORE_JUDGMENT_RAW, STORE_REFLECTION, STORE_RESPONSE


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


def records_from_log(log: str | Path | EvalLog, *, strict: bool = True) -> list[dict]:
    """Build direct-rating records from an eval log, in plan (edge) order."""

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

    failures = [
        f"sample {sample.id}: {sample.error.message}"
        for sample in samples
        if sample.error is not None
    ]
    if failures and strict:
        shown = "\n  ".join(failures[:10])
        more = f"\n  ... and {len(failures) - 10} more" if len(failures) > 10 else ""
        raise RuntimeError(
            f"{len(failures)} sample(s) failed; nothing exported:\n  {shown}{more}\n"
            "Resume with: inspect eval-retry <log>"
        )

    rows = []
    seen_edges: set[tuple[int, int, int]] = set()
    for sample in samples:
        if sample.error is not None:
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


def write_inspect_run_info(run_dir: str | Path, log: EvalLog) -> Path:
    """Record which log produced a run, so the upload can carry it."""

    path = Path(run_dir) / "inspect_run.json"
    info = {}
    if path.exists():
        info = json.loads(path.read_text(encoding="utf-8"))
    info["log_file"] = Path(str(log.location)).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
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
) -> tuple[list[dict], Path]:
    """Export one eval log to evaluations.jsonl (+ optional response cache)."""

    log = load_log(log)
    meta = eigenbench_metadata(log)
    target = Path(evaluations_path or meta.get("evaluations_path") or "")
    if not str(target):
        raise ValueError(
            "no evaluations path: pass evaluations_path or set "
            "collection.evaluations_path in the run spec"
        )

    records = records_from_log(log, strict=strict)
    write_evaluations_atomic(target, records)
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
