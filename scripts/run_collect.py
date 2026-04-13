"""Internal collection stage for pipeline orchestration.

This module is intended to be invoked by ``scripts/run.py``.
"""

from __future__ import annotations

from pipeline.config import (
    load_run_spec,
    load_dataset_scenarios_from_spec,
    select_scenarios,
    get_criteria_from_spec,
)
from pipeline.eval import collect_core_evaluations
from pipeline.utils import append_records, load_records


def _build_cached_index(cached_records):
    index = {}
    for entry in cached_records:
        if isinstance(entry, dict) and "scenario_index" in entry and "responses" in entry:
            index[entry["scenario_index"]] = entry
    return index


def _has_local_models(models: dict[str, str]) -> bool:
    return any(v.startswith("hf_local:") for v in models.values())


def main(spec_ref: str):
    spec, run_dir = load_run_spec(spec_ref)
    verbose = bool(spec.get("verbose", False))

    models = spec["models"]
    ds = spec["dataset"]
    constitution = spec["constitution"]
    cfg = spec["collection"]

    if not bool(cfg.get("enabled", True)):
        if verbose:
            print("Collection disabled in run spec (collection.enabled=False). Skipping run_collect.")
        return

    scenarios = load_dataset_scenarios_from_spec(ds, run_dir=run_dir)
    start = int(ds.get("start", 0))
    count = ds.get("count")
    count = None if count is None else int(count)
    shuffle = bool(ds.get("shuffle", False))
    shuffle_seed = ds.get("shuffle_seed")
    shuffle_seed = None if shuffle_seed is None else int(shuffle_seed)
    selected = select_scenarios(
        scenarios,
        start=start,
        count=count,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
    )

    if "num_criteria" not in constitution:
        raise SystemExit(
            "Set constitution.num_criteria in your run spec. "
            "This controls criterion truncation during collection."
        )
    requested_num_criteria = int(constitution["num_criteria"])
    if requested_num_criteria <= 0:
        raise SystemExit("constitution.num_criteria must be a positive integer.")

    criteria = get_criteria_from_spec(constitution, run_dir=run_dir)
    if requested_num_criteria < len(criteria):
        if verbose:
            print(
                f"Truncating constitution criteria from {len(criteria)} to "
                f"{requested_num_criteria} based on constitution.num_criteria."
            )
        criteria = criteria[:requested_num_criteria]
    elif requested_num_criteria > len(criteria):
        raise SystemExit(
            f"constitution.num_criteria={requested_num_criteria} exceeds "
            f"criteria found in constitution file ({len(criteria)})."
        )

    evaluations_path = cfg.get("evaluations_path")
    if not evaluations_path:
        raise SystemExit("Set collection.evaluations_path in your run spec.")

    sampler_mode = (cfg.get("sampler_mode", "random_judge_group")).strip().lower()

    # Route: mixed collection (hf_local models or all_to_all mode)
    if _has_local_models(models) or sampler_mode == "all_to_all":
        from pipeline.eval.mixed_collect import collect_mixed_evaluations

        if verbose:
            print(f"Run: {spec['name']}")
            print(f"Run folder: {run_dir}")
            print(f"Evaluations file: {evaluations_path}")
            print(
                "Scenario selection: "
                f"total={len(scenarios)}, selected={len(selected)}, start={start}, "
                f"count={'all' if count is None else count}, shuffle={shuffle}, shuffle_seed={shuffle_seed}"
            )

        collect_mixed_evaluations(
            models=models,
            selected_scenarios=selected,
            criteria=criteria,
            collection_cfg=cfg,
            evaluations_path=evaluations_path,
            verbose=verbose,
            model_system_prompts=spec.get("model_system_prompts"),
        )
        return

    # Route: OpenRouter-only collection (original path)
    cached_responses_path = cfg.get("cached_responses_path")
    cached_index = None
    if cached_responses_path:
        cached_records = load_records(cached_responses_path)
        cached_index = _build_cached_index(cached_records)

    if verbose:
        print(f"Run: {spec['name']}")
        print(f"Run folder: {run_dir}")
        print(f"Evaluations file: {evaluations_path}")
        print(f"Cached responses file: {cached_responses_path}")
        print(
            "Scenario selection: "
            f"total={len(scenarios)}, selected={len(selected)}, start={start}, "
            f"count={'all' if count is None else count}, shuffle={shuffle}, shuffle_seed={shuffle_seed}"
        )

    for scenario_index, scenario in selected:
        existing = load_records(evaluations_path)
        new_evals = collect_core_evaluations(
            criteria=criteria,
            scenario=scenario,
            scenario_index=scenario_index,
            models=models,
            evaluations=existing,
            sampler_mode=cfg.get("sampler_mode", "random_judge_group"),
            allow_ties=bool(cfg.get("allow_ties", True)),
            group_size=int(cfg.get("group_size", 4)),
            groups=int(cfg.get("groups", 1)),
            alpha=float(cfg.get("alpha", 2.0)),
            cached_responses_by_scenario=cached_index,
            verbose=verbose,
        )
        append_records(evaluations_path, new_evals)
        if verbose:
            print(
                f"Wrote {len(new_evals)} new evaluations for scenario_index={scenario_index} "
                f"-> {evaluations_path}"
            )


if __name__ == "__main__":
    raise SystemExit(
        "run_collect.py is an internal stage. "
        "Use: python scripts/run.py <spec_module_or_path>"
    )
