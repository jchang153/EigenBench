"""Constitution/criteria accessors sourced from data/constitutions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRITERIA_FILE = _REPO_ROOT / "data/constitutions/criteria_map.json"


def _load_criteria_map() -> dict[str, list[str]]:
    with _CRITERIA_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {_CRITERIA_FILE}, got {type(data).__name__}")
    return data


def get_criteria(criteria_id: str):
    key = criteria_id.strip().lower()
    criteria_map = _load_criteria_map()

    if key in {"kindness", "universal_kindness"}:
        return criteria_map["kindness"]
    if key in {"ecology", "deep_ecology"}:
        return criteria_map["deep_ecology"]
    if key in {"conservatism", "conservatism_gpt"}:
        return criteria_map["conservatism"]
    if key == "openai":
        return criteria_map["openai"]
    if key == "claude":
        return criteria_map["claude"]
    raise ValueError(f"Unknown criteria_id: {criteria_id}")


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
    - {'criteria_id': 'kindness' | ...}
    - plain string path or criteria_id.
    """

    if isinstance(constitution_cfg, str):
        maybe_path = constitution_cfg.strip()
        if any(ch in maybe_path for ch in ("/", "\\")) or maybe_path.lower().endswith(".json"):
            p = _resolve_existing_path(maybe_path, run_dir=run_dir)
            with p.open("r", encoding="utf-8") as f:
                return _normalize_criteria(json.load(f))
        return _normalize_criteria(get_criteria(maybe_path))

    if not isinstance(constitution_cfg, dict):
        raise ValueError("constitution config must be a dict or string")

    if "path" in constitution_cfg and constitution_cfg["path"]:
        p = _resolve_existing_path(str(constitution_cfg["path"]), run_dir=run_dir)
        with p.open("r", encoding="utf-8") as f:
            return _normalize_criteria(json.load(f))

    if "criteria_id" in constitution_cfg and constitution_cfg["criteria_id"]:
        return _normalize_criteria(get_criteria(str(constitution_cfg["criteria_id"])))

    raise ValueError("constitution config must include either 'path' or 'criteria_id'")
