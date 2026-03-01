"""Run collection then training from a Python run spec.

Usage:
    python scripts/run.py runs.example.spec
    python scripts/run.py runs/example/spec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow "python scripts/run.py ..." to import top-level packages (e.g. pipeline).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.config import load_run_spec


def main(spec_ref: str):
    spec, _ = load_run_spec(spec_ref)
    collection_cfg = spec.get("collection", {})
    training_cfg = spec.get("training", {})
    cached_responses_path = collection_cfg.get("cached_responses_path")

    if cached_responses_path:
        print("Stage: collect responses cache")
        from run_collect_responses import main as run_collect_responses_main

        run_collect_responses_main(spec_ref)
    else:
        print("Stage: collect responses cache (skipped; collection.cached_responses_path is not set)")

    if bool(collection_cfg.get("enabled", True)):
        print("Stage: collect evaluations")
        from run_collect import main as run_collect_main

        run_collect_main(spec_ref)
    else:
        print("Stage: collect evaluations (skipped; collection.enabled=False)")

    if bool(training_cfg.get("enabled", True)):
        print("Stage: train + eigentrust")
        from run_train import main as run_train_main

        run_train_main(spec_ref)
    else:
        print("Stage: train + eigentrust (skipped; training.enabled=False)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/run.py <spec_module_or_path>")
    main(sys.argv[1])
