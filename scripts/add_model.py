"""Add one model to a finished run, collecting only the inference it needs.

    python scripts/add_model.py runs/my_run/spec.py \
        --model "New Model" --id anthropic/claude-sonnet-4-5 [--dry-run]

Existing responses are reused where the new model acts as judge, so the cost is
roughly one response plus two judgments per new edge rather than a rebuild.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inspect_pipeline.extend import extend_run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("spec", help="Run spec of the existing run")
    p.add_argument("--model", required=True, help="Display name for the new model")
    p.add_argument("--id", required=True, help="Model reference, as written in a spec")
    p.add_argument("--no-self", action="store_true", help="Do not let it rate itself")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Print the plan and stop")
    a = p.parse_args()

    extend_run(a.spec, a.model, a.id, include_self=not a.no_self,
               seed=a.seed, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
