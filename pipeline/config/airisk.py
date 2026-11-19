"""Load unique AIRiskDilemmas questions from a pinned Hugging Face snapshot."""

from __future__ import annotations

import json
from pathlib import Path

DATASET_ID = "kellycyy/AIRiskDilemmas"
DATASET_REVISION = "8674d1f5844c3909b05e06d9f30bbc2b7c753f39"


def paired_dilemmas(rows) -> list[str]:
    """Validate action pairs and retain each question once, in source order."""
    iterator = iter(rows)
    scenarios: list[str] = []
    seen: set[str] = set()
    while True:
        try:
            first = next(iterator)
        except StopIteration:
            break
        try:
            second = next(iterator)
        except StopIteration as exc:
            raise ValueError("AIRiskDilemmas contains an unpaired final action row") from exc
        first_value = first.get("dilemma") if isinstance(first, dict) else None
        second_value = second.get("dilemma") if isinstance(second, dict) else None
        if not isinstance(first_value, str) or not first_value.strip():
            raise ValueError("AIRiskDilemmas contains an empty or non-string dilemma")
        if first_value != second_value:
            raise ValueError("consecutive AIRiskDilemmas action rows have different dilemmas")
        if first_value not in seen:
            seen.add(first_value)
            scenarios.append(first_value)
    return scenarios


def load_airisk_scenarios() -> list[str]:
    """Download once, then read the pinned source from the HF cache offline.

    The legacy data/scenarios/airiskdilemmas.json is intentionally not the
    source for the built-in dataset ID: notebooks may have overwritten it
    with raw action rows or data from an unpinned revision.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    source = dict(
        repo_id=DATASET_ID,
        filename="model_eval.jsonl",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    try:
        source_path = hf_hub_download(**source, local_files_only=True)
    except LocalEntryNotFoundError:
        source_path = hf_hub_download(**source)
    with Path(source_path).open("r", encoding="utf-8") as handle:
        scenarios = paired_dilemmas(json.loads(line) for line in handle if line.strip())
    if not scenarios:
        raise ValueError("AIRiskDilemmas contains no scenarios")
    return scenarios
