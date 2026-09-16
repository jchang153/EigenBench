"""Programmatic driver: build the task, run it, export evaluations.jsonl.

Equivalent to ``inspect eval inspect_pipeline/eigenbench.py -T spec=...``
followed by ``scripts/export_evaluations.py``; used by scripts/run_inspect.py.
"""

from __future__ import annotations

import json
from collections import defaultdict
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

from .eigenbench import eigenbench, has_local_models, load_selection
from .phased import eigenbench_judge, eigenbench_responses
from .export import (
    export_log,
    records_from_logs,
    write_evaluations_atomic,
    write_inspect_run_info,
)
from .phases import STORE_RESPONSE


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


async def _close_model(nick: str, resolve_model) -> None:
    """Terminate a local model's vLLM server so the next one can have the GPU."""

    try:
        api = resolve_model(nick).api
    except Exception:
        return
    aclose = getattr(api, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception as exc:  # a stuck server must not sink the run
            print(f"  warning: could not close {nick}: {exc}")


def _run_phased(spec_ref: str, models, ctx, eval_kw: dict) -> list[dict]:
    """Responses then judgments, one model resident at a time.

    A judgment needs both its evaluee and its judge, so an edge-per-sample run
    keeps every model live at once. That is fine for hosted models and fatal on
    one GPU, so local runs phase the work the way the legacy collector did.
    """

    import anyio

    from inspect_ai import eval as inspect_eval

    resolve_model = ctx["resolve_model"]
    nicks = list(models)
    responses: dict[int, dict[str, str]] = defaultdict(dict)
    for (s_idx, nick), text in ctx["seed"].items():
        responses[s_idx][nick] = text

    judge_logs = []

    def close(nick: str) -> None:
        anyio.run(_close_model, nick, resolve_model)

    print(f"Phase 1/2: responses, {len(nicks)} model(s), one at a time")
    for nick in nicks:
        missing = [s for s, _ in ctx["selected"] if nick not in responses[int(s)]]
        if not missing:
            continue
        logs = inspect_eval(
            tasks=eigenbench_responses(spec_ref, model_nick=nick, models=models),
            model=None,
            max_tasks=1,
            **eval_kw,
        )
        for sample in logs[0].samples or []:
            content = (sample.store or {}).get(STORE_RESPONSE)
            if isinstance(content, str) and content.strip():
                responses[int(sample.metadata["scenario_index"])][nick] = content
        close(nick)

    edges_by_judge: dict[str, list[dict]] = defaultdict(list)
    for a in ctx["assignments"]:
        s_idx = int(a["scenario_index"])
        for eval_idx, eval_nick in zip(a["eval_idxs"], a["eval_nicks"]):
            edges_by_judge[a["judge_nick"]].append(
                {
                    "edge_index": len(edges_by_judge[a["judge_nick"]]),
                    "scenario_index": s_idx,
                    "scenario": a["scenario"],
                    "judge_idx": int(a["judge_idx"]),
                    "judge_nick": a["judge_nick"],
                    "eval_idx": int(eval_idx),
                    "eval_nick": eval_nick,
                    "sampler_mode": a.get("sampler_mode"),
                    "sampling_round": int(a.get("sampling_round", 0)),
                    "group_index": int(a.get("group_index", 0)),
                    "response": responses[s_idx][eval_nick],
                }
            )

    print(f"Phase 2/2: judgments, {len(edges_by_judge)} judge(s), one at a time")
    for nick, edges in edges_by_judge.items():
        logs = inspect_eval(
            tasks=eigenbench_judge(
                spec_ref, judge_nick=nick, edges=edges, models=models
            ),
            model=None,
            max_tasks=1,
            **eval_kw,
        )
        judge_logs.append(logs[0])
        close(nick)

    return judge_logs


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

    eval_kw = eval_kwargs(inspect_cfg, log_dir)
    spec_models = models if models is not None else spec["models"]
    phased = bool(inspect_cfg.get("phased", has_local_models(spec_models)))

    if phased:
        from .eigenbench import _run_context

        ctx = _run_context(spec_ref, models)
        judge_logs = _run_phased(spec_ref, spec_models, ctx, eval_kw)
        records = records_from_logs(judge_logs)
        write_evaluations_atomic(evaluations_path, records)
        write_inspect_run_info(evaluations_path.parent, judge_logs)
        target = evaluations_path
    else:
        task = eigenbench(spec_ref, models=models) if models else eigenbench(spec_ref)
        logs = inspect_eval(tasks=task, model=None, **eval_kw)
        records, target = export_log(
            logs[0],
            evaluations_path=str(evaluations_path),
            cached_responses_path=collection_cfg.get("cached_responses_path"),
        )

    print(f"Direct collection complete. {len(records)} ratings saved to {target}")
    return records
