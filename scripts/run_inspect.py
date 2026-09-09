"""Run an EigenBench direct-rating spec end to end on the Inspect AI engine.

    python scripts/run_inspect.py runs/my_run/spec.py [--estimate-calls]

Convenience wrapper around the native Inspect workflow:

    inspect eval inspect_pipeline/eigenbench.py -T spec=runs/my_run/spec.py
    python scripts/export_evaluations.py <log> -o runs/my_run/evaluations.jsonl
    # then training / upload

Everything downstream of collection is delegated to scripts/run.py, which
consumes the identical evaluations.jsonl.
"""

from __future__ import annotations

import argparse
import os
import pprint
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pipeline.config import load_run_spec  # noqa: E402


def main(spec_ref: str, collection_enabled: bool | None = None) -> None:
    spec, _run_dir = load_run_spec(spec_ref)
    if spec.get("evaluation", {}).get("mode") != "direct_rating":
        raise SystemExit(
            "run_inspect.py supports evaluation.mode='direct_rating' only; "
            "use scripts/run.py for pairwise BTD runs."
        )

    collection_cfg = spec.get("collection", {})
    if collection_enabled is not None:
        collection_cfg["enabled"] = collection_enabled

    # Fail fast on a missing Space secret, mirroring scripts/run.py.
    upload_cfg = spec.get("upload", {})
    if bool(upload_cfg.get("enabled", False)) and not (
        upload_cfg.get("secret") or os.environ.get("SPACE_SECRET", "")
    ):
        raise SystemExit("Set upload.secret in spec or SPACE_SECRET env var")

    if bool(collection_cfg.get("enabled", True)):
        print("Stage: collect evaluations (Inspect engine)")
        from inspect_pipeline.collect import collect_direct_ratings_inspect

        collect_direct_ratings_inspect(spec_ref)
    else:
        print("Stage: collect evaluations (skipped; collection.enabled=False)")

    # Training / aggregation / upload run on the legacy stages.
    from run import main as legacy_main

    legacy_main(spec_ref, collection_enabled=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EigenBench direct-rating runner on Inspect AI"
    )
    parser.add_argument("spec", help="Run spec module or path, e.g. runs/my_run/spec.py")
    parser.add_argument(
        "--estimate-calls",
        action="store_true",
        help="Print planned API call counts and exit without collecting",
    )
    parser.add_argument(
        "--collection-enabled",
        choices=["true", "false"],
        default=None,
        help="Override collection.enabled from the spec",
    )
    args = parser.parse_args()

    if args.estimate_calls:
        from run import estimate_calls

        print(pprint.pformat(estimate_calls(args.spec), sort_dicts=False))
    else:
        override = None
        if args.collection_enabled is not None:
            override = args.collection_enabled == "true"
        main(args.spec, collection_enabled=override)
