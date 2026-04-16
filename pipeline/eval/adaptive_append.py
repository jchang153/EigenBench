"""Adaptive append: add new models to an existing evaluations.jsonl.

Given an existing evaluations file with N models and a spec with N+K models,
this module:
1. Loads prior evaluations and computes per-model judge/eval counts
2. Remaps old numeric indices to match the new (expanded) model list
3. Generates new assignments using adaptive_inverse_count sampling,
   which heavily favors the K new models (count=0) over existing ones
4. Runs collection for new assignments only
5. Appends new evaluations to the existing file

Spec config:
    "collection": {
        "adaptive_append": True,
        "evaluations_path": "evaluations.jsonl",
        ...
    }

The new models appear in spec["models"] alongside existing ones.
Existing evaluations are preserved; new evaluations are appended.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.utils import load_records


def load_prior_model_counts(
    evaluations_path: str | Path,
    model_nicks: list[str],
) -> tuple[list[int], list[int], set[str]]:
    """Load existing evaluations and compute judge/eval counts for the new model list.

    Returns:
        judge_counts: per-model judge appearance count (indexed by new model list)
        eval_counts: per-model evaluee appearance count (indexed by new model list)
        prior_nicks: set of model nicks found in existing evaluations
    """
    records = load_records(evaluations_path)
    if not records:
        return [0] * len(model_nicks), [0] * len(model_nicks), set()

    nick_to_idx = {nick: i for i, nick in enumerate(model_nicks)}
    num_models = len(model_nicks)

    judge_counts = [0] * num_models
    eval_counts = [0] * num_models
    prior_nicks = set()

    for record in records:
        judge_name = record.get("judge_name", "")
        eval1_name = record.get("eval1_name", "")
        eval2_name = record.get("eval2_name", "")

        if judge_name:
            prior_nicks.add(judge_name)
        if eval1_name:
            prior_nicks.add(eval1_name)
        if eval2_name:
            prior_nicks.add(eval2_name)

        if judge_name in nick_to_idx:
            judge_counts[nick_to_idx[judge_name]] += 1
        if eval1_name in nick_to_idx:
            eval_counts[nick_to_idx[eval1_name]] += 1
        if eval2_name in nick_to_idx:
            eval_counts[nick_to_idx[eval2_name]] += 1

    return judge_counts, eval_counts, prior_nicks


def remap_prior_evaluations(
    evaluations_path: str | Path,
    model_nicks: list[str],
) -> None:
    """Remap numeric indices in existing evaluations to match the new model list.

    When new models are added to the spec, the model list order may change,
    making old numeric indices (eval1, eval2, judge) incorrect. This function
    updates them based on the name fields (eval1_name, eval2_name, judge_name).

    Overwrites the file in place.
    """
    records = load_records(evaluations_path)
    if not records:
        return

    nick_to_idx = {nick: i for i, nick in enumerate(model_nicks)}
    changed = False

    for record in records:
        for idx_key, name_key in [
            ("eval1", "eval1_name"),
            ("eval2", "eval2_name"),
            ("judge", "judge_name"),
        ]:
            name = record.get(name_key, "")
            if name in nick_to_idx:
                new_idx = nick_to_idx[name]
                if record.get(idx_key) != new_idx:
                    record[idx_key] = new_idx
                    changed = True

    if changed:
        p = Path(evaluations_path)
        with p.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")


def detect_new_models(
    model_nicks: list[str],
    prior_nicks: set[str],
) -> list[str]:
    """Return nicks that are in the current spec but not in prior evaluations."""
    return [nick for nick in model_nicks if nick not in prior_nicks]


def print_adaptive_summary(
    model_nicks: list[str],
    judge_counts: list[int],
    eval_counts: list[int],
    new_models: list[str],
) -> None:
    """Print a summary of model coverage and which models are new."""
    print(f"\nAdaptive append: {len(model_nicks)} models in spec, "
          f"{len(new_models)} new")
    print(f"  New models: {new_models}")
    print(f"  Prior coverage:")
    for i, nick in enumerate(model_nicks):
        marker = " [NEW]" if nick in new_models else ""
        print(f"    {nick}: judge={judge_counts[i]}, eval={eval_counts[i]}{marker}")
    print()
