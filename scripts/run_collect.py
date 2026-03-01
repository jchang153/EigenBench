"""Run collection from a Python run spec.

Usage:
    python scripts/run_collect.py runs.example_custom
"""

from __future__ import annotations

import sys

from pipeline.config import (
    load_run_spec,
    load_dataset_scenarios_from_spec,
    get_criteria_from_spec,
)
from pipeline.eval import collect_core_evaluations
from pipeline.io import append_records, load_records


def _build_cached_index(cached_records):
    index = {}
    for entry in cached_records:
        if isinstance(entry, dict) and "scenario_index" in entry and "responses" in entry:
            index[entry["scenario_index"]] = entry
    return index


def main(spec_ref: str):
    spec, run_dir = load_run_spec(spec_ref)

    models = spec["models"]
    ds = spec["dataset"]
    constitution = spec["constitution"]
    cfg = spec["collection"]

    scenarios = load_dataset_scenarios_from_spec(ds, run_dir=run_dir)
    start = int(ds.get("start", 0))
    count = int(ds.get("count", len(scenarios) - start))
    selected = scenarios[start : start + count]

    criteria = get_criteria_from_spec(constitution, run_dir=run_dir)

    evaluations_path = cfg.get("evaluations_path")
    if not evaluations_path:
        raise SystemExit("Set collection.evaluations_path in your run spec.")

    cached_responses_path = cfg.get("cached_responses_path")
    cached_index = None
    if cached_responses_path:
        cached_records = load_records(cached_responses_path)
        cached_index = _build_cached_index(cached_records)

    print(f"Run: {spec['name']}")
    print(f"Run folder: {run_dir}")
    print(f"Evaluations file: {evaluations_path}")
    print(f"Cached responses file: {cached_responses_path}")

    for offset, scenario in enumerate(selected):
        scenario_index = start + offset
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
        )
        append_records(evaluations_path, new_evals)
        print(
            f"Wrote {len(new_evals)} new evaluations for scenario_index={scenario_index} "
            f"-> {evaluations_path}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/run_collect.py <spec_module_or_path>")
    main(sys.argv[1])
