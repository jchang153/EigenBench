"""Freeze a native judge draw for review before making model calls."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from eigenbench.protocol.direct_rating import (
    DIRECT_SAMPLER_PAIR_BALANCED, direct_input_sha256, validate_direct_assignments,
)


def preview_path(collection):
    return Path(collection["evaluations_path"]).with_name("judge_assignments.json")


def judgment_rows(assignments):
    return [
        {"scenario_index": a["scenario_index"], "judge": a["judge_nick"],
         "evaluee": nick, "sampling_round": a["sampling_round"]}
        for a in assignments for nick in a["eval_nicks"]
    ]


def save_preview(path, inputs, seed, assignments):
    rows = judgment_rows(assignments)
    plan = {**inputs, "resolved_sampler_seed": seed, "judgments": rows}
    plan["preview_sha256"] = direct_input_sha256(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never expose a partial draw or overwrite one another reviewer already saw.
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        try:
            json.dump(plan, tmp, indent=2)
            tmp.write("\n")
            tmp.flush()
            os.link(tmp.name, path)
        finally:
            Path(tmp.name).unlink()


def read_preview(path, inputs, selected, criteria, models, include_self):
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.pop("preview_sha256", None) != direct_input_sha256(plan):
        raise ValueError(f"{path}: saved assignment preview was edited or is incomplete")
    for key, expected in inputs.items():
        if plan.get(key) != expected:
            raise ValueError(f"{path}: {key} differs from the preview; use a new output directory")
    assignments = validate_direct_assignments(plan, selected, criteria, models,
                                              include_self=include_self, path=path)
    for a in assignments:
        a.update(sampler_mode=DIRECT_SAMPLER_PAIR_BALANCED, group_index=a["eval_idxs"][0])
    return assignments, plan["resolved_sampler_seed"]
