"""Run collection then training from a Python run spec.

Usage:
    python scripts/run_pipeline.py runs.example.spec
    python scripts/run_pipeline.py runs/example/spec.py
"""

from __future__ import annotations

import sys

from run_collect import main as run_collect_main
from run_train import main as run_train_main


def main(spec_ref: str):
    run_collect_main(spec_ref)
    run_train_main(spec_ref)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/run_pipeline.py <spec_module_or_path>")
    main(sys.argv[1])
