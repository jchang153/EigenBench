"""Run-spec helpers for folder-based defaults."""

from __future__ import annotations

import copy
from pathlib import Path


def _resolve_path_for_run(path_value: str | None, run_dir: Path, default_name: str) -> str:
    if not path_value:
        return str((run_dir / default_name).resolve())
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return str(p)
    return str((run_dir / p).resolve())


def infer_run_name_and_dir(spec_module: str, module_file: str, spec: dict) -> tuple[str, Path]:
    module_path = Path(module_file).resolve()
    fallback_name = spec_module.split(".")[-1]
    run_name = str(spec.get("name") or fallback_name)

    # Package layout (recommended): runs/<name>/__init__.py or runs/<name>/config.py
    if module_path.name in {"__init__.py", "config.py"}:
        run_dir = module_path.parent
    else:
        # Legacy single-file layout: runs/<name>.py -> place outputs in runs/<name>/
        run_dir = module_path.parent / run_name

    return run_name, run_dir


def apply_run_defaults(spec_module: str, module_file: str, spec: dict) -> tuple[dict, Path]:
    """Return a normalized spec with run-folder defaults applied."""

    normalized = copy.deepcopy(spec)
    run_name, run_dir = infer_run_name_and_dir(spec_module, module_file, normalized)

    normalized["name"] = run_name

    collection = normalized.setdefault("collection", {})
    collection["evaluations_path"] = _resolve_path_for_run(
        collection.get("evaluations_path"),
        run_dir,
        "out/evaluations.jsonl",
    )
    collection["cached_responses_path"] = _resolve_path_for_run(
        collection.get("cached_responses_path"),
        run_dir,
        "out/cached_responses.jsonl",
    )

    training = normalized.setdefault("training", {})
    training["output_dir"] = _resolve_path_for_run(
        training.get("output_dir"),
        run_dir,
        "out/train",
    )

    return normalized, run_dir
