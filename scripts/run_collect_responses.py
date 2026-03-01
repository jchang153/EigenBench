"""Run response-only collection from a Python run spec.

Usage:
    python scripts/run_collect_responses.py runs.example_custom
"""

from __future__ import annotations

import importlib
import sys

from pipeline.config import apply_run_defaults, load_dataset_scenarios_from_spec
from pipeline.eval.flows import collect_responses_only
from pipeline.io import append_records, load_records


def _build_cached_index(cached_records):
    index = {}
    for entry in cached_records:
        if isinstance(entry, dict) and "scenario_index" in entry and "responses" in entry:
            index[entry["scenario_index"]] = entry
    return index


def main(spec_module: str):
    mod = importlib.import_module(spec_module)
    spec, run_dir = apply_run_defaults(spec_module, mod.__file__, mod.RUN_SPEC)

    models = spec["models"]
    ds = spec["dataset"]
    cfg = spec["collection"]

    cache_path = cfg.get("cached_responses_path")
    if not cache_path:
        raise SystemExit("Set collection.cached_responses_path in your run spec.")

    scenarios = load_dataset_scenarios_from_spec(ds, run_dir=run_dir)
    start = int(ds.get("start", 0))
    count = int(ds.get("count", len(scenarios) - start))
    selected = scenarios[start : start + count]

    overwrite = bool(cfg.get("overwrite_cached_responses", False))
    max_tokens = int(cfg.get("max_tokens", 4096))

    existing = load_records(cache_path)
    cached_index = _build_cached_index(existing)

    print(f"Run: {spec['name']}")
    print(f"Run folder: {run_dir}")
    print(f"Cached responses file: {cache_path}")

    for offset, scenario in enumerate(selected):
        scenario_index = start + offset

        if (not overwrite) and (scenario_index in cached_index):
            print(f"Skipping scenario_index={scenario_index}; cached response set already exists.")
            continue

        rows = collect_responses_only(
            scenario=scenario,
            scenario_index=scenario_index,
            models=models,
            max_tokens=max_tokens,
            cached_responses_by_scenario=None,
        )
        append_records(cache_path, rows)
        cached_index[scenario_index] = rows[0]
        print(f"Wrote cached responses for scenario_index={scenario_index} -> {cache_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/run_collect_responses.py <spec_module>")
    main(sys.argv[1])
