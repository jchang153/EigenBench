"""Bundle a run's Inspect logs into a static viewer and record its URL.

    # local bundle (inspect it before publishing anything)
    python scripts/publish_inspect_bundle.py runs/my_run --output-dir runs/my_run/inspect_bundle

    # publish to a HuggingFace Space and record the URL for ValueArena
    python scripts/publish_inspect_bundle.py runs/my_run \
        --output-dir hf/<org>/<space> \
        --url https://<org>-<space>.static.hf.space

A static Space is served from the `.static.hf.space` subdomain -- the plain
`.hf.space` host 404s -- and is created private, so make it public before
linking it.

The recorded URL lands in <run_dir>/inspect_run.json, which
scripts/upload_results.py copies into meta.json as `meta.inspect`. ValueArena
shows its "Open in Inspect" button only when that block is present.

Note: the viewer app is ~11MB, so prefer one bundle holding many runs' logs
over one bundle per run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inspect_ai.log import bundle_log_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run folder containing inspect_logs/")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Logs to bundle (default: <run_dir>/inspect_logs)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Bundle destination; an 'hf/<org>/<name>' path uploads to a HF Space",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-bundle",
        action="store_true",
        help="Only record --url, without rebuilding the bundle",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_dir = Path(args.log_dir) if args.log_dir else run_dir / "inspect_logs"
    if not args.skip_bundle:
        if not log_dir.is_dir():
            raise SystemExit(f"no Inspect logs at {log_dir}")
        bundle_log_dir(
            log_dir=str(log_dir), output_dir=args.output_dir, overwrite=args.overwrite
        )
        print(f"Bundled {log_dir} -> {args.output_dir}")

    info_path = run_dir / "inspect_run.json"
    info = {}
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    if not info.get("log_file"):
        # Fall back to the newest log when collection did not record one.
        logs = sorted(log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime)
        if not logs:
            raise SystemExit(f"no .eval logs in {log_dir} to reference")
        info["log_file"] = logs[-1].name

    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {info_path}: {info}")


if __name__ == "__main__":
    main()
