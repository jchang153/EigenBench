"""Extend from RUN_SPEC.extension and the full RUN_SPEC.models population.

    python scripts/extend_run.py runs/expanded/spec.py --dry-run
    python scripts/extend_run.py runs/expanded/spec.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inspect_pipeline.extend import extend_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec")
    parser.add_argument("--additional-scenarios", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    extend_run(args.spec, additional_scenarios=args.additional_scenarios,
               seed=args.seed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
