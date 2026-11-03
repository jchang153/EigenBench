"""Prepare a complete EigenBench log for review without changing its evidence."""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import re

from inspect_ai.log import EvalLog, read_eval_log, write_eval_log


_PRIVATE_PATH = re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/]|/(?:Users|home|private|tmp|var|opt|workspace)/)[^\s\"'<>]*")
_CREDENTIAL = re.compile(r"(?:.*[_-])?(?:api[_-]?key|access[_-]?key(?:[_-]?id)?|token|authorization|password|secret)", re.I)
_PROTECTED = {"input", "output", "messages", "store", "record", "scenario", "criteria",
              "model_order", "model", "model_name", "base_model"}


def prepare_log(source: Path, destination: Path, *, redact: tuple[str, ...] = ()) -> Path:
    """Write a de-identified copy, refusing edits to benchmark content or labels."""
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError("Choose a new destination; source logs are retained")
    log = read_eval_log(source, resolve_attachments=True)
    if log.status != "success" or not log.samples or any(s.error for s in log.samples):
        raise ValueError("A publication log must have completed every sample successfully")
    panel = next((s for s in log.results.scores if s.name == "panel_scorer"), None) if log.results else None
    metrics = {key: metric.value for key, metric in panel.metrics.items()} if panel else {}
    if (metrics.get("ranking_available") != 1 or not metrics.get("coverage/planned")
            or metrics.get("coverage/scored") != metrics["coverage/planned"]
            or len(log.samples) != metrics["coverage/planned"]):
        raise ValueError("A publication log requires native rankings and complete planned coverage")
    source_commit = (log.eval.metadata or {}).get("source_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Record the full source SHA with --metadata source_commit=<40-character SHA>")

    private = set(redact) | {str(Path.home()), getpass.getuser()}
    private.update(value for key, value in os.environ.items()
                   if value and re.search(r"(?:API_KEY|ACCESS_TOKEN|HF_TOKEN|SPACE_SECRET)$", key))
    private.discard("")

    def scrub(value, pointer="", protected=False):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                child = f"{pointer}/{key}"
                if scrub(key, pointer) != key:
                    raise ValueError(f"Private identifier in a dictionary key at {pointer}; review manually")
                if not protected and (key in {"traceback", "traceback_ansi"}
                                      or (key == "call" and value.get("event") == "model")):
                    continue
                if not protected and _CREDENTIAL.fullmatch(key) and isinstance(item, str):
                    result[key] = "[redacted]"
                else:
                    result[key] = scrub(item, child, protected or key in _PROTECTED
                                        or (key == "name" and "/panel/" in child))
            return result
        if isinstance(value, list):
            return [scrub(item, f"{pointer}/{i}", protected) for i, item in enumerate(value)]
        if not isinstance(value, str):
            return value
        clean = value
        for marker in sorted(private, key=len, reverse=True):
            clean = clean.replace(marker, "[redacted]")
        clean = _PRIVATE_PATH.sub("[local-path]", clean)
        clean = re.sub(r"(https?://)[^/\s@]+@", r"\1", clean)
        clean = re.sub(r"([?&](?:api_key|token|key|secret)=)[^&\s]+", r"\1[redacted]", clean, flags=re.I)
        if protected and clean != value:
            raise ValueError(f"Redaction would change benchmark evidence at {pointer}; review manually")
        return clean

    data = log.model_dump(mode="json", exclude_none=True)
    # The resolved fields are authoritative; omit unused attachment-table copies.
    for sample in data["samples"]:
        sample.pop("attachments", None)
    public = EvalLog.model_validate(scrub(data))
    public.eval.metadata = {**(public.eval.metadata or {}), "eigenbench_public_copy": {
        "raw_api_calls_removed": True, "tracebacks_removed": True,
        "benchmark_content_preserved": True,
    }}
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(public, destination)
    saved = read_eval_log(destination)
    if [s.id for s in saved.samples] != [s.id for s in log.samples]:
        raise RuntimeError("Public log sample identities changed")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--redact", action="append", default=[], help="Additional private identifier to remove")
    args = parser.parse_args()
    print(prepare_log(args.source, args.destination, redact=tuple(args.redact)))


if __name__ == "__main__":
    main()
