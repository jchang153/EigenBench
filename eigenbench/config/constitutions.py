"""Constitution/criteria loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
        f"Could not resolve constitution path '{path_value}'. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def _normalize_criteria(payload: Any) -> list[str]:
    if isinstance(payload, list):
        if not all(isinstance(x, str) for x in payload):
            raise ValueError("Constitution list payload must contain only strings.")
        return payload

    if isinstance(payload, dict):
        for key in ("criteria", "comparative_criteria", "comparativeCriteria"):
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                return value

        # Accept single-key dict containing criteria list.
        if len(payload) == 1:
            only_value = next(iter(payload.values()))
            if isinstance(only_value, list) and all(isinstance(x, str) for x in only_value):
                return only_value

        raise ValueError(
            "Constitution JSON dict must contain a list field named "
            "'criteria', 'comparative_criteria', or 'comparativeCriteria'."
        )

    raise ValueError("Constitution JSON must be a list or dict.")


def get_criteria_from_spec(
    constitution_cfg: dict | str,
    *,
    run_dir: str | Path | None = None,
) -> list[str]:
    """Path-first criteria loader.

    Supports either:
    - {'path': '.../constitution.json'}
    - plain string path.

    For path-based constitution files, accepted payload formats are:
    - list[str]
    - dict containing one list[str] field with key:
      - criteria
      - comparative_criteria
      - comparativeCriteria

    Relative paths are resolved first against run_dir, then repo root.
    """

    if isinstance(constitution_cfg, str):
        maybe_path = constitution_cfg.strip()
        p = _resolve_existing_path(maybe_path, run_dir=run_dir)
        with p.open("r", encoding="utf-8") as f:
            return _normalize_criteria(json.load(f))

    if not isinstance(constitution_cfg, dict):
        raise ValueError("constitution config must be a dict or string")

    if "path" in constitution_cfg and constitution_cfg["path"]:
        p = _resolve_existing_path(str(constitution_cfg["path"]), run_dir=run_dir)
        with p.open("r", encoding="utf-8") as f:
            return _normalize_criteria(json.load(f))

    raise ValueError("constitution config must include 'path'")
