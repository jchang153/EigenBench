"""Whether existing collection output belongs to the run being collected.

A finished run is reused rather than collected again, so reuse has to check what it
reuses. The records say which judgments, scenarios, models and constitution they hold;
``collection_fingerprint.json`` records what they do not: model references, sampling,
generation settings and the prompts. Anything that differs stops collection with an
explanation instead of returning another run's judgments.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from inspect_ai.model import Model

from pipeline.eval.direct_rating import (
    DIRECT_PROMPT_VERSION,
    build_direct_rating_prompt,
    build_direct_rating_user_prompt,
    build_direct_reflection_prompt,
    build_direct_reflection_user_prompt,
)

from .coverage import _exported_key, judgment_key, planned_from_assignments
from .export import records_fingerprint
from .extend import scenarios_of
from .model_mapping import to_inspect_model
from .phases import RESPONSE_SYSTEM_MESSAGE


FINGERPRINT_FILE = "collection_fingerprint.json"
FINGERPRINT_VERSION = 1


def constitution_differs(records: list[dict], criteria: list[str]) -> bool:
    """True when any record was judged against other criteria than these."""

    expected = "\n".join(criteria)
    return any(r.get("constitution") != expected for r in records)


def changed_scenarios(records: list[dict], selected) -> list[int]:
    """Scenario indices whose recorded text is not what the selection holds now."""

    current = dict(selected)
    return sorted(s for s, text in scenarios_of(records).items() if current.get(s) != text)


def _model_reference(value) -> dict:
    """What a panel entry points at, as the response cache identifies it."""

    if isinstance(value, Model):
        return {"name": str(value)}
    ref = to_inspect_model(value)
    return {"name": ref.name, "args": ref.model_args}


def _prompts_identity() -> str:
    """The prompts as they render, so editing any template changes the identity."""

    probe = ("{criteria}", "{scenario}", "{response}")
    prompts = [
        RESPONSE_SYSTEM_MESSAGE,
        build_direct_reflection_prompt(),
        build_direct_reflection_user_prompt(*probe),
        build_direct_rating_prompt(),
        build_direct_rating_user_prompt(*probe, "{reflection}"),
    ]
    return records_fingerprint([{"prompts": prompts}])


def collection_identity(ctx: dict) -> dict:
    """Everything that shapes a collection that its records do not carry."""

    return {
        "models": [
            {"nick": nick, **_model_reference(value)} for nick, value in ctx["models"].items()
        ],
        "sampling": {**ctx["sampling"], "include_self": ctx["include_self"]},
        "generation": ctx["generation"],
        "protocol": {
            "mode": "direct_rating",
            "prompt_version": DIRECT_PROMPT_VERSION,
            "prompts_sha256": _prompts_identity(),
            "scale": [ctx["scale_min"], ctx["scale_max"]],
        },
    }


def _normalized(identity: dict) -> dict:
    # Round-trip through JSON so a stored identity and a fresh one compare alike.
    return json.loads(json.dumps(identity, sort_keys=True, default=str))


def write_fingerprint(evaluations_path: str | Path, ctx: dict) -> Path:
    identity = _normalized(collection_identity(ctx))
    path = Path(evaluations_path).with_name(FINGERPRINT_FILE)
    payload = {"version": FINGERPRINT_VERSION, "sha256": records_fingerprint([identity]),
               "identity": identity}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _describe(key) -> str:
    _, s_idx, judge, evaluee, round_ = key
    return f"scenario {s_idx}: {judge} judging {evaluee}" + (f" (round {round_})" if round_ else "")


def _plan_problems(records: list[dict], ctx: dict, omitted: list[dict]) -> list[str]:
    planned = Counter(judgment_key(e) for e in planned_from_assignments(ctx["assignments"]))
    held = Counter(_exported_key(r) for r in records)
    problems = []
    extra = held - planned
    if extra:
        shown = ", ".join(_describe(k) for k in list(extra)[:3])
        problems.append(f"{sum(extra.values())} judgment(s) are not in this plan, e.g. {shown}")
    # Omitted judgments count toward the plan; their rows carry no sampling round.
    missing = planned - held
    for row in omitted:
        for key in missing:
            if missing[key] and key[1:4] == (int(row["scenario_index"]), row["judge"], row["evaluee"]):
                missing[key] -= 1
                break
    missing = +missing
    if missing:
        shown = ", ".join(_describe(k) for k in list(missing)[:3])
        problems.append(f"{sum(missing.values())} planned judgment(s) are missing, e.g. {shown}")
    return problems


def _model_problems(records: list[dict], ctx: dict) -> list[str]:
    order = list(ctx["models"])
    wrong = sorted({
        f"{side['name']} at index {side['index']}"
        for r in records if r.get("record_type") == "direct_rating"
        for side in (r["judge"], r["evaluee"])
        if side["name"] not in order or order.index(side["name"]) != int(side["index"])
    })
    if not wrong:
        return []
    return [f"model names or positions differ from the panel {order}: {', '.join(wrong[:5])}"]


def _fingerprint_problems(path: Path, ctx: dict) -> list[str]:
    stored = json.loads(path.read_text(encoding="utf-8")).get("identity", {})
    current = _normalized(collection_identity(ctx))
    problems = []
    if stored.get("models") != current["models"]:
        before = {m["nick"]: m for m in stored.get("models", [])}
        changed = [m["nick"] for m in current["models"] if before.get(m["nick"]) != m]
        dropped = [nick for nick in before if nick not in {m["nick"] for m in current["models"]}]
        detail = ", ".join(changed + [f"{nick} (removed)" for nick in dropped]) or "panel order"
        problems.append(f"model references changed: {detail}")
    for section, label in (("sampling", "sampling settings"), ("generation", "generation settings")):
        if stored.get(section) != current[section]:
            keys = sorted(k for k in set(stored.get(section, {})) | set(current[section])
                          if stored.get(section, {}).get(k) != current[section].get(k))
            problems.append(f"{label} changed: {', '.join(keys)}")
    if stored.get("protocol") != current["protocol"]:
        problems.append("the judging protocol or its prompts changed")
    return problems


def check_reusable(evaluations_path: str | Path, records: list[dict], ctx: dict,
                   omitted: list[dict]) -> None:
    """Stop unless ``records`` are exactly this plan's collection, explaining any difference.

    A run collected before fingerprints were written is checked against its records
    alone, with a warning that its settings could not be verified.
    """

    path = Path(evaluations_path)
    problems = _plan_problems(records, ctx, omitted) + _model_problems(records, ctx)
    changed = changed_scenarios(records, ctx["selected"])
    if changed:
        problems.append(f"scenario text changed at index {', '.join(map(str, changed[:10]))}")
    if constitution_differs(records, ctx["criteria"]):
        problems.append("the constitution's criteria differ from the ones these records were judged on")
    fingerprint = path.with_name(FINGERPRINT_FILE)
    if fingerprint.exists():
        problems += _fingerprint_problems(fingerprint, ctx)
    if problems:
        listed = "\n  - ".join(problems)
        raise RuntimeError(
            f"{path} holds a different collection than this plan:\n  - {listed}\n"
            "Move or delete it (and its collection files) to collect again."
        )
    if not fingerprint.exists():
        print(f"warning: {path.name} has no {FINGERPRINT_FILE}; its judgments match this plan, "
              "but its model references, sampling and generation settings could not be verified")
