"""Run collection then training from a Python run spec.

Usage:
    python scripts/run_pipeline.py runs.example_custom
"""

from __future__ import annotations

import sys

from run_collect import main as run_collect_main
from run_train import main as run_train_main


def main(spec_module: str):
    run_collect_main(spec_module)
    run_train_main(spec_module)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/run_pipeline.py <spec_module>")
    main(sys.argv[1])
