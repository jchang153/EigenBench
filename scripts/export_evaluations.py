"""Export an Inspect eval log to the legacy evaluations.jsonl contract.

    python scripts/export_evaluations.py logs/2026-08-31_eigenbench.eval
    python scripts/export_evaluations.py <log> -o runs/my_run/evaluations.jsonl

With no -o, the destination comes from collection.evaluations_path recorded in
the log's task metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inspect_ai.log import list_eval_logs  # noqa: E402

from inspect_pipeline.export import export_log  # noqa: E402


def resolve_log(log_ref: str) -> str:
    path = Path(log_ref)
    if path.is_dir():
        logs = list_eval_logs(str(path))
        if not logs:
            raise SystemExit(f"no eval logs found in {path}")
        newest = max(logs, key=lambda info: info.mtime or 0)
        return newest.name
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="Eval log file, or a log directory (newest is used)")
    parser.add_argument("-o", "--output", default=None, help="evaluations.jsonl path")
    parser.add_argument(
        "--cached-responses",
        default=None,
        help="Append per-scenario responses to this shared cache jsonl",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Export successful samples even if some failed (default: refuse)",
    )
    args = parser.parse_args()

    records, target = export_log(
        resolve_log(args.log),
        evaluations_path=args.output,
        cached_responses_path=args.cached_responses,
        strict=not args.allow_incomplete,
    )
    print(f"Exported {len(records)} direct-rating records to {target}")


if __name__ == "__main__":
    main()
