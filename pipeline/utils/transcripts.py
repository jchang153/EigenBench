"""Transcript read/write helpers.

JSONL is the canonical format for long-running append-heavy collection jobs.
JSON array format is still supported for backward compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_records(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []

    if p.suffix.lower() == ".jsonl":
        records: list[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected list payload in {p}, got {type(payload).__name__}")


def save_records(path: str | Path, records: Iterable[dict]) -> None:
    p = Path(path)
    _ensure_parent(p)
    items = list(records)

    if p.suffix.lower() == ".jsonl":
        with p.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
        return

    with p.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=4)


def append_records(path: str | Path, new_records: Iterable[dict]) -> None:
    p = Path(path)
    _ensure_parent(p)
    items = list(new_records)
    if not items:
        return

    if p.suffix.lower() == ".jsonl":
        with p.open("a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
        return

    existing = load_records(p)
    existing.extend(items)
    save_records(p, existing)
