#!/usr/bin/env python3
"""Run all prompted constitution experiments: collect locally, then submit to Space.

Phase 1 (GPU): Run collection for all 11 constitutions sequentially.
Phase 2 (fire-and-forget): One nohup background script that:
  - Submits all 11 runs to Space sequentially (each waits for completion)
  - Builds the character-train matrix from HF results
  - Uploads matrix_view.png to HF

After Phase 1 completes, the GPU machine can be shut down.
Phase 2 runs in background — needs a machine with internet (no GPU).

Usage:
    export SPACE_SECRET="..."
    python scripts/run_prompted.py
    python scripts/run_prompted.py --skip-collection
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.config import load_run_spec
from run import SPACES, DEFAULT_SPACE

CONSTITUTIONS = [
    "goodness",
    "humor",
    "impulsiveness",
    "loving",
    "mathematical",
    "misalignment",
    "nonchalance",
    "poeticism",
    "remorse",
    "sarcasm",
    "sycophancy",
]


def run_collection(spec_ref: str):
    """Run collection stage only."""
    spec, _ = load_run_spec(spec_ref)
    collection_cfg = spec.get("collection", {})

    cached_responses_path = collection_cfg.get("cached_responses_path")
    if cached_responses_path:
        from run_collect_responses import main as run_collect_responses_main
        run_collect_responses_main(spec_ref)

    if bool(collection_cfg.get("enabled", True)):
        from run_collect import main as run_collect_main
        run_collect_main(spec_ref)


def build_submit_script(group: str, specs_dir: Path, space: str = DEFAULT_SPACE) -> str:
    """Build a self-contained Python script that submits all runs + matrix."""
    space_url = SPACES.get(space, SPACES[DEFAULT_SPACE])
    space_secret = os.environ.get("SPACE_SECRET", "")

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:
        git_commit = ""

    # Collect all run info
    runs = []
    for c in CONSTITUTIONS:
        spec_ref = str(specs_dir / c / "spec.py")
        spec, _ = load_run_spec(spec_ref)
        upload_cfg = spec.get("upload", {})
        collection_cfg = spec.get("collection", {})
        runs.append({
            "constitution": c,
            "eval_path": str(Path(collection_cfg.get("evaluations_path", "")).resolve()),
            "spec_path": str(Path(spec_ref).resolve()),
            "run_name": upload_cfg.get("name", ""),
            "run_group": upload_cfg.get("group", group),
            "run_note": upload_cfg.get("note", ""),
        })

    return f'''
import os, sys, json, tempfile
for k in list(os.environ):
    if k.lower() in ("all_proxy", "ftp_proxy", "grpc_proxy", "rsync_proxy"):
        os.environ.pop(k, None)

from gradio_client import Client, handle_file

SPACE_SECRET = {space_secret!r}
GIT_COMMIT = {git_commit!r}
GROUP = {group!r}
REPO_ROOT = {str(_REPO_ROOT)!r}
SPACE_URL = {space_url!r}
RUNS = {json.dumps(runs, indent=2)}

# --- Submit all runs to Space ---
print("Submitting", len(RUNS), "runs to Space:", SPACE_URL)
client = Client(SPACE_URL)

for run in RUNS:
    name = run["run_name"]
    print(f"  Submitting {{name}}...")
    try:
        result = client.predict(
            SPACE_SECRET,
            handle_file(run["eval_path"]),
            handle_file(run["spec_path"]),
            run["run_name"],
            run["run_group"],
            run["run_note"],
            GIT_COMMIT,
        )
        print(f"  {{name}}: OK ({{result[0] if result else result}})")
    except Exception as e:
        print(f"  {{name}}: FAILED ({{e}})")

print("All runs submitted.")

# --- Build and upload matrix ---
print("\\nBuilding character-train matrix from HF...")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
os.chdir(REPO_ROOT)

from build_matrix import find_model_nick, plot_matrix, save_csv, BASE_NICK, REF_ANCHOR
from upload_matrix import fetch_summary_from_hf, build_matrix_from_hf, upload_matrix_to_hf

CONSTITUTIONS = {json.dumps(CONSTITUTIONS)}

summaries = {{}}
for c in CONSTITUTIONS:
    bs = fetch_summary_from_hf(GROUP, c)
    if bs:
        summaries[c] = bs
        print(f"  {{c}}: OK")
    else:
        print(f"  {{c}}: not found on HF")

if len(summaries) >= 2:
    A_mean, A_std, consts, col_labels = build_matrix_from_hf(summaries)
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        staging = Path(tmpdir)
        plot_matrix(A_mean, A_std, consts, staging / "matrix_view.png",
                    col_labels=col_labels,
                    title=f"Character-Train Matrix — {{GROUP}} (Elo, API avg = {{REF_ANCHOR}})")
        save_csv(A_mean, consts, staging / "matrix_view.csv", col_labels=col_labels)
        upload_matrix_to_hf(GROUP, staging)
    print("Matrix uploaded!")
else:
    print("Not enough summaries for matrix.")

print("\\nALL DONE")
'''


def main():
    parser = argparse.ArgumentParser(description="Run all prompted constitution experiments")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip collection (already done), just submit to Space")
    parser.add_argument("--group", default="prompted", help="Run group name")
    parser.add_argument("--space", default=DEFAULT_SPACE, choices=list(SPACES.keys()),
                        help=f"Which HF Space to use for upload (default: {DEFAULT_SPACE})")
    args = parser.parse_args()

    specs_dir = _REPO_ROOT / "runs" / args.group

    # Phase 1: Collection (GPU)
    if not args.skip_collection:
        print("=" * 60)
        print("PHASE 1: Collection (GPU)")
        print("=" * 60)
        for c in CONSTITUTIONS:
            spec_ref = str(specs_dir / c / "spec.py")
            print(f"\n--- {c} ---")
            run_collection(spec_ref)
        print("\nCollection complete for all constitutions.")
    else:
        print("Skipping collection (--skip-collection)")

    # Phase 2: Fire-and-forget background script
    print("\n" + "=" * 60)
    print("PHASE 2: Fire-and-forget (Space submit + matrix)")
    print("=" * 60)

    script_content = build_submit_script(args.group, specs_dir, space=args.space)
    script_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="va_prompted_"
    )
    script_file.write(script_content)
    script_file.close()

    log_file = script_file.name.replace(".py", ".log")
    subprocess.Popen(
        f"nohup {sys.executable} -u {script_file.name} > {log_file} 2>&1 &",
        shell=True,
    )

    space_url = SPACES.get(args.space, SPACES[DEFAULT_SPACE])
    print(f"Background job launched!")
    print(f"  Space:  {space_url}")
    print(f"  Script: {script_file.name}")
    print(f"  Log:    {log_file}")
    print(f"\n{'=' * 60}")
    print("GPU can be turned off now.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
