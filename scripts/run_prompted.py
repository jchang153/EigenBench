#!/usr/bin/env python3
"""Run all prompted constitution experiments: collect locally, then submit to Space.

Phase 1 (GPU): Run collection for all 11 constitutions sequentially.
Phase 2 (fire-and-forget): One nohup background script that:
  - Submits all 11 runs to the Space queue (non-blocking, uses .submit())
  - Polls HF every 60s for completed summaries
  - Builds the character-train matrix (Elo + CI) from HF results
  - Uploads matrix_view.png + matrix_ci.png to HF

The Space queue (max_size=20, concurrency=1) processes jobs serially on its own.
The client doesn't wait for individual results — it polls HF for completion.
This avoids SSE connection timeouts that broke long sequential waits.

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
from run import space_url, DEFAULT_SPACE

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
    surl = space_url(space)
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
import os, sys, json, tempfile, time, threading
for k in list(os.environ):
    if k.lower() in ("all_proxy", "ftp_proxy", "grpc_proxy", "rsync_proxy"):
        os.environ.pop(k, None)

from gradio_client import Client, handle_file

SPACE_SECRET = {space_secret!r}
GIT_COMMIT = {git_commit!r}
GROUP = {group!r}
REPO_ROOT = {str(_REPO_ROOT)!r}
SPACE_URL = {surl!r}
RUNS = {json.dumps(runs, indent=2)}

# --- Phase A: Submit all runs to Space queue (fire-and-forget, parallel) ---
# Each submission uses a fresh client + .submit() (non-blocking).
# The Space queue (max_size=20, concurrency=1) processes them serially on its side.
# We don't wait for results — we'll poll HF for completion instead.
print("Submitting", len(RUNS), "runs to Space:", SPACE_URL)

submitted = []
def _submit_one(run):
    name = run["run_name"]
    try:
        # Fresh client per submission so a broken connection doesn't poison others
        c = Client(SPACE_URL)
        # .submit() returns a Job object immediately without waiting for result
        job = c.submit(
            SPACE_SECRET,
            handle_file(run["eval_path"]),
            handle_file(run["spec_path"]),
            run["run_name"],
            run["run_group"],
            run["run_note"],
            GIT_COMMIT,
        )
        print(f"  Queued: {{name}}", flush=True)
        return (name, job)
    except Exception as e:
        print(f"  Queue FAILED: {{name}} ({{e}})", flush=True)
        return (name, None)

# Submit serially (one connection at a time) but don't wait for results.
# .submit() returns immediately, so all 11 land in the queue within seconds.
for run in RUNS:
    submitted.append(_submit_one(run))

print(f"All {{len(submitted)}} submissions queued. Space will process serially.", flush=True)

# --- Phase B: Poll HF for completion ---
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
os.chdir(REPO_ROOT)

from upload_matrix import fetch_summary_from_hf, build_matrix_from_hf, upload_matrix_to_hf
from build_matrix import plot_matrix, plot_ci_matrix, save_csv, REF_ANCHOR

CONSTITUTIONS = {json.dumps(CONSTITUTIONS)}
POLL_INTERVAL = 60  # seconds between polls
MAX_WAIT_MIN = 240  # give up after 4 hours

print(f"\\nPolling HF every {{POLL_INTERVAL}}s for completion (max {{MAX_WAIT_MIN}} min)...", flush=True)

start_time = time.time()
seen = set()
while True:
    elapsed_min = (time.time() - start_time) / 60
    if elapsed_min > MAX_WAIT_MIN:
        print(f"Timeout after {{MAX_WAIT_MIN}} min. Found {{len(seen)}}/{{len(CONSTITUTIONS)}}.", flush=True)
        break

    new_found = []
    for c in CONSTITUTIONS:
        if c in seen:
            continue
        bs = fetch_summary_from_hf(GROUP, c)
        if bs:
            seen.add(c)
            new_found.append(c)

    if new_found:
        print(f"  [{{elapsed_min:.0f}}min] Found: {{new_found}} ({{len(seen)}}/{{len(CONSTITUTIONS)}})", flush=True)

    if len(seen) == len(CONSTITUTIONS):
        print(f"All {{len(CONSTITUTIONS)}} runs complete after {{elapsed_min:.0f}} min.", flush=True)
        break

    time.sleep(POLL_INTERVAL)

# --- Phase C: Build and upload matrix ---
print("\\nBuilding character-train matrix from HF...", flush=True)
summaries = {{}}
for c in CONSTITUTIONS:
    bs = fetch_summary_from_hf(GROUP, c)
    if bs:
        summaries[c] = bs

if len(summaries) >= 2:
    A_mean, A_std, consts, col_labels = build_matrix_from_hf(summaries)
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        staging = Path(tmpdir)
        plot_matrix(A_mean, A_std, consts, staging / "matrix_view.png",
                    col_labels=col_labels,
                    title=f"Character-Train Matrix — {{GROUP}} (Elo, API avg = {{REF_ANCHOR}})")
        plot_ci_matrix(A_std, consts, staging / "matrix_ci.png",
                       col_labels=col_labels,
                       title=f"Character-Train Matrix — {{GROUP}} (CI Width)")
        save_csv(A_mean, consts, staging / "matrix_view.csv", col_labels=col_labels)
        upload_matrix_to_hf(GROUP, staging)
    print("Matrix uploaded!", flush=True)
else:
    print(f"Not enough summaries for matrix ({{len(summaries)}}/{{len(CONSTITUTIONS)}}).", flush=True)

print("\\nALL DONE", flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description="Run all prompted constitution experiments")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip collection (already done), just submit to Space")
    parser.add_argument("--group", default="prompted", help="Run group name")
    parser.add_argument("--space", default=DEFAULT_SPACE,
                        help=f"Space number: 1=valuearena, 2+=valuearena-N (default: {DEFAULT_SPACE})")
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

    print(f"Background job launched!")
    print(f"  Space:  {space_url(args.space)}")
    print(f"  Script: {script_file.name}")
    print(f"  Log:    {log_file}")
    print(f"\n{'=' * 60}")
    print("GPU can be turned off now.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
