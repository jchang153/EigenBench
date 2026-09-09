"""Programmatic driver: build the task, run it, export evaluations.jsonl.

Equivalent to ``inspect eval inspect_pipeline/eigenbench.py -T spec=...``
followed by ``scripts/export_evaluations.py``; used by scripts/run_inspect.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import eval as inspect_eval

from pipeline.config import load_run_spec
from pipeline.eval.direct_rating import (
    build_direct_assignments,
    count_cached_responses,
    estimate_direct_calls,
    resolve_direct_sampling_settings,
)
from pipeline.model_refs import is_hf_local_model
from pipeline.utils import load_records

from .eigenbench import eigenbench, load_selection
from .export import export_log


def write_call_estimate(spec: dict, run_dir: Path) -> dict:
    """Exact planned call counts, mirroring scripts/run_collect.py."""

    models = spec["models"]
    collection_cfg = spec.get("collection", {})
    evaluation_cfg = spec.get("evaluation", {})
    include_self = bool(
        (evaluation_cfg.get("direct_rating", {}) or {}).get("include_self", True)
    )
    selected, _criteria = load_selection(spec, run_dir)
    sampling = resolve_direct_sampling_settings(
        collection_cfg, num_models=len(models), include_self=include_self
    )
    assignments = build_direct_assignments(
        selected, models, include_self=include_self, **sampling
    )
    openrouter_nicks = {
        nick for nick, value in models.items() if not is_hf_local_model(value)
    }
    cached_total, cached_remote = count_cached_responses(
        collection_cfg.get("cached_responses_path"),
        scenario_indices={int(item[0]) for item in selected},
        model_nicks=set(models),
        openrouter_nicks=openrouter_nicks,
    )
    estimate = {
        "mode": "direct_rating",
        **estimate_direct_calls(
            num_scenarios=len(selected),
            num_models=len(models),
            num_openrouter_models=len(openrouter_nicks),
            include_self=include_self,
            cached_responses=cached_total,
            cached_openrouter_responses=cached_remote,
            assignments=assignments,
            openrouter_model_indices={
                idx for idx, nick in enumerate(models) if nick in openrouter_nicks
            },
            **sampling,
        ),
    }
    estimate_path = Path(collection_cfg["evaluations_path"]).with_name(
        "direct_call_estimate.json"
    )
    estimate_path.parent.mkdir(parents=True, exist_ok=True)
    estimate_path.write_text(json.dumps(estimate, indent=2) + "\n", encoding="utf-8")
    return estimate


def eval_kwargs(inspect_cfg: dict, log_dir: Path) -> dict:
    kwargs: dict = {"log_dir": str(log_dir), "fail_on_error": False}
    for key in ("max_connections", "max_samples", "retry_on_error"):
        if inspect_cfg.get(key) is not None:
            kwargs[key] = int(inspect_cfg[key])
    if inspect_cfg.get("display") is not None:
        kwargs["display"] = inspect_cfg["display"]
    return kwargs


def collect_direct_ratings_inspect(
    spec_ref: str,
    *,
    models: dict[str, object] | None = None,
) -> list[dict]:
    """Run collection for a run spec and write evaluations.jsonl."""

    spec, run_dir = load_run_spec(spec_ref)
    collection_cfg = spec.get("collection", {})
    evaluations_path = Path(collection_cfg["evaluations_path"])
    inspect_cfg = collection_cfg.get("inspect", {}) or {}

    log_dir = Path(inspect_cfg.get("log_dir") or "inspect_logs")
    if not log_dir.is_absolute():
        log_dir = evaluations_path.parent / log_dir

    # A completed run short-circuits; a foreign file is refused.
    estimate = write_call_estimate(spec, run_dir)
    expected = estimate["rating_tasks"]
    if evaluations_path.exists() and evaluations_path.stat().st_size > 0:
        existing = load_records(str(evaluations_path))
        if len(existing) == expected:
            print(f"Direct collection already complete: {len(existing)} ratings")
            return existing
        raise RuntimeError(
            f"{evaluations_path} exists with {len(existing)} records but this plan "
            f"expects {expected}; move or delete it before collecting"
        )

    print(
        f"Direct-rating plan: {estimate['total_logical_generations']} logical "
        f"generations ({estimate['response_tasks']} responses, "
        f"{estimate['reflection_tasks']} reflections, {estimate['rating_tasks']} ratings)"
    )

    task = eigenbench(spec_ref, models=models) if models else eigenbench(spec_ref)
    logs = inspect_eval(tasks=task, model=None, **eval_kwargs(inspect_cfg, log_dir))

    records, target = export_log(
        logs[0],
        evaluations_path=str(evaluations_path),
        cached_responses_path=collection_cfg.get("cached_responses_path"),
    )
    print(f"Direct collection complete. {len(records)} ratings saved to {target}")
    return records
