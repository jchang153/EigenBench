"""Extend the completed example_inspect run with two models and two scenarios.

First collect runs/example_inspect/spec.py. Then:
    python scripts/extend_run.py runs/example_extension/spec.py --dry-run
    python scripts/extend_run.py runs/example_extension/spec.py
    python scripts/run.py runs/example_extension/spec.py --collection-enabled false
"""
from copy import deepcopy
from pathlib import Path

from pipeline.config import load_run_spec

# Reuse the original model/evaluation settings. Redirect run-relative dataset
# and output paths below; the constitution path is relative to the repository.
_ROOT = Path(__file__).resolve().parents[2]
_base, _ = load_run_spec(str(_ROOT / "runs/example_inspect/spec.py"))
RUN_SPEC = deepcopy(_base)
RUN_SPEC["name"] = "example_extension"
RUN_SPEC["models"].update({
    "Claude Sonnet 4.5": "anthropic/claude-sonnet-4-5",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro",
})
RUN_SPEC["extension"] = {
    "from_evaluations": "../example_inspect/evaluations.jsonl",
    "additional_scenarios": 2,
}
# The source example selects the first four scenarios; expose two more here.
RUN_SPEC["dataset"] = {
    "path": "../example_inspect/scenarios.json",
    "start": 0,
    "count": 6,
}
RUN_SPEC["collection"]["evaluations_path"] = "evaluations.jsonl"
RUN_SPEC["collection"]["inspect"]["log_dir"] = "inspect_logs"
RUN_SPEC["training"]["output_dir"] = "."
