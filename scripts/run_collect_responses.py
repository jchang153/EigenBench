"""Internal response-cache collection stage for pipeline orchestration.

This module is intended to be invoked by ``scripts/run.py``.
"""

from __future__ import annotations

from pipeline.config import load_run_spec, load_dataset_scenarios_from_spec, select_scenarios
from pipeline.eval.flows import collect_responses_only
from pipeline.utils import append_records, load_records


def _build_cached_index(cached_records):
    index = {}
    for entry in cached_records:
        if isinstance(entry, dict) and "scenario_index" in entry and "responses" in entry:
            index[entry["scenario_index"]] = entry
    return index


def main(spec_ref: str):
    spec, run_dir = load_run_spec(spec_ref)
    verbose = bool(spec.get("verbose", False))

    models = spec["models"]
    ds = spec["dataset"]
    cfg = spec["collection"]

    cache_path = cfg.get("cached_responses_path")
    if not cache_path:
        raise SystemExit(
            "Set collection.cached_responses_path in your run spec "
            "(recommended: data/responses/<cache_name>.jsonl)."
        )

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

    overwrite = bool(cfg.get("overwrite_cached_responses", False))
    max_tokens = int(cfg.get("max_tokens", 4096))

    existing = load_records(cache_path)
    cached_index = _build_cached_index(existing)

    if verbose:
        print(f"Run: {spec['name']}")
        print(f"Run folder: {run_dir}")
        print(f"Cached responses file: {cache_path}")
        print(
            "Scenario selection: "
            f"total={len(scenarios)}, selected={len(selected)}, start={start}, "
            f"count={'all' if count is None else count}, shuffle={shuffle}, shuffle_seed={shuffle_seed}"
        )

    for scenario_index, scenario in selected:

        if (not overwrite) and (scenario_index in cached_index):
            if verbose:
                print(f"Skipping scenario_index={scenario_index}; cached response set already exists.")
            continue

        rows = collect_responses_only(
            scenario=scenario,
            scenario_index=scenario_index,
            models=models,
            max_tokens=max_tokens,
            cached_responses_by_scenario=None,
            verbose=verbose,
        )
        append_records(cache_path, rows)
        cached_index[scenario_index] = rows[0]
        if verbose:
            print(f"Wrote cached responses for scenario_index={scenario_index} -> {cache_path}")


if __name__ == "__main__":
    raise SystemExit(
        "run_collect_responses.py is an internal stage. "
        "Use: python scripts/run.py <spec_module_or_path>"
    )
