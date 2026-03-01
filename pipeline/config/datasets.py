"""Dataset loaders."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DATASET_PATHS = {
    "reddit": Path("data/scenarios/reddit_questions.json"),
    "oasst": Path("data/scenarios/oasst_questions.json"),
    "airisk": Path("data/scenarios/airiskdilemmas.json"),
}


def load_dataset_scenarios(dataset_id: str):
    key = dataset_id.strip().lower()
    if key not in _DATASET_PATHS:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")
    p = _REPO_ROOT / _DATASET_PATHS[key]
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_existing_path(path_value: str, run_dir: str | Path | None) -> Path:
    raw = Path(path_value).expanduser()
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        if run_dir is not None:
            candidates.append((Path(run_dir) / raw).resolve())
        candidates.append((_REPO_ROOT / raw).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve dataset path '{path_value}'. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def _normalize_scenarios(payload: Any) -> list[str]:
    if not isinstance(payload, list):
        raise ValueError("Scenario JSON must be a list.")

    if not payload:
        return []

    if all(isinstance(x, str) for x in payload):
        return payload

    # Also accept list[dict] with common prompt keys.
    out = []
    for item in payload:
        if isinstance(item, dict):
            if "scenario" in item and isinstance(item["scenario"], str):
                out.append(item["scenario"])
                continue
            if "prompt" in item and isinstance(item["prompt"], str):
                out.append(item["prompt"])
                continue
            if "question" in item and isinstance(item["question"], str):
                out.append(item["question"])
                continue
        raise ValueError(
            "Scenario JSON list items must be strings, or dicts containing "
            "'scenario', 'prompt', or 'question' string fields."
        )
    return out


def load_dataset_scenarios_from_spec(
    dataset_cfg: dict | str,
    *,
    run_dir: str | Path | None = None,
) -> list[str]:
    """Path-first dataset loader.

    Supports either:
    - {'path': '.../scenarios.json', 'start': ..., 'count': ...}
    - {'id': 'reddit' | 'oasst' | 'airisk', ...}
    - plain string path or id.

    For path-based scenario files, accepted payload formats are:
    - list[str]
    - list[dict] where each item has one of:
      - scenario
      - prompt
      - question

    Relative paths are resolved first against run_dir, then repo root.
    """

    if isinstance(dataset_cfg, str):
        # Try as path first if it looks like a file path.
        maybe_path = dataset_cfg.strip()
        if any(ch in maybe_path for ch in ("/", "\\")) or maybe_path.lower().endswith(".json"):
            p = _resolve_existing_path(maybe_path, run_dir=run_dir)
            with p.open("r", encoding="utf-8") as f:
                return _normalize_scenarios(json.load(f))
        return _normalize_scenarios(load_dataset_scenarios(maybe_path))

    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset config must be a dict or string")

    if "path" in dataset_cfg and dataset_cfg["path"]:
        p = _resolve_existing_path(str(dataset_cfg["path"]), run_dir=run_dir)
        with p.open("r", encoding="utf-8") as f:
            return _normalize_scenarios(json.load(f))

    if "id" in dataset_cfg and dataset_cfg["id"]:
        return _normalize_scenarios(load_dataset_scenarios(str(dataset_cfg["id"])))

    raise ValueError("dataset config must include either 'path' or 'id'")


def select_scenarios(
    scenarios: list[str],
    *,
    start: int = 0,
    count: int | None = None,
    shuffle: bool = False,
    shuffle_seed: int | None = None,
) -> list[tuple[int, str]]:
    """Select scenarios as ``(original_index, scenario_text)`` pairs.

    Behavior:
    - Builds indexed scenarios from the full input list.
    - Optionally shuffles before slicing.
    - Applies ``start`` and ``count`` after optional shuffle.
    - Preserves original indices so cache/evaluation keys remain stable.
    """

    if start < 0:
        raise ValueError("dataset.start must be >= 0")
    if count is not None and count < 0:
        raise ValueError("dataset.count must be >= 0 when provided")

    indexed = list(enumerate(scenarios))
    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(indexed)

    if count is None:
        return indexed[start:]
    return indexed[start : start + count]
