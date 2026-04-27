#!/usr/bin/env python3
"""Run all prompted constitution experiments: collect locally, then submit to Space.

Phase 1 (GPU): Run collection for all 11 constitutions sequentially.
Phase 2 (fire-and-forget): One nohup background script that:
  - Submits runs to the Space ONE AT A TIME (sequential)
  - After each submission, polls HF until summary.json appears
  - Only then submits the next run
  - After all runs complete, builds and uploads the character-train matrix

Sequential submission avoids the Space crash-on-queue problem: the Space
tends to process one job then restart, losing queued jobs.

Usage:
    export SPACE_SECRET="..."
    python scripts/run_prompted.py
    python scripts/run_prompted.py --skip-collection
    python scripts/run_prompted.py --only sarcasm,sycophancy,remorse
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
    """Build a self-contained Python script that submits runs sequentially."""
    surl = space_url(space)
    space_secret = os.environ.get("SPACE_SECRET", "")

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:
        git_commit = ""

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
import os, sys, json, tempfile, time
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

POLL_INTERVAL = 90
MAX_WAIT_PER_RUN = 7200  # 2 hours per run
MAX_RETRIES = 3

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
os.chdir(REPO_ROOT)

from upload_matrix import fetch_summary_from_hf, build_matrix_from_hf, upload_matrix_to_hf
from build_matrix import plot_matrix, plot_ci_matrix, save_csv, REF_ANCHOR

# --- Check which runs already exist on HF (skip them) ---
already_done = set()
for run in RUNS:
    bs = fetch_summary_from_hf(GROUP, run["constitution"])
    if bs:
        already_done.add(run["constitution"])
        print(f"  Already on HF: {{run['constitution']}}", flush=True)

pending = [r for r in RUNS if r["constitution"] not in already_done]
print(f"\\n{{len(already_done)}} already done, {{len(pending)}} to submit.", flush=True)

# --- Sequential submit: one at a time, poll until done, then next ---
completed = list(already_done)
failed = []

for idx, run in enumerate(pending):
    name = run["run_name"]
    constitution = run["constitution"]
    print(f"\\n[{{idx+1}}/{{len(pending)}}] Submitting: {{name}}", flush=True)

    success = False
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            c = Client(SPACE_URL)
            job = c.submit(
                SPACE_SECRET,
                handle_file(run["eval_path"]),
                handle_file(run["spec_path"]),
                run["run_name"],
                run["run_group"],
                run["run_note"],
                GIT_COMMIT,
            )
            print(f"  Submitted (attempt {{attempt}}). Polling HF...", flush=True)
        except Exception as e:
            print(f"  Submit failed (attempt {{attempt}}): {{e}}", flush=True)
            if attempt < MAX_RETRIES:
                time.sleep(30)
                continue
            else:
                failed.append(constitution)
                break

        # Poll HF until this specific run's summary.json appears
        start = time.time()
        while time.time() - start < MAX_WAIT_PER_RUN:
            time.sleep(POLL_INTERVAL)
            elapsed = (time.time() - start) / 60
            bs = fetch_summary_from_hf(GROUP, constitution)
            if bs:
                print(f"  Done! {{constitution}} appeared on HF after {{elapsed:.0f}} min.", flush=True)
                completed.append(constitution)
                success = True
                break
            print(f"  [{{elapsed:.0f}}min] Waiting for {{constitution}}...", flush=True)

        if success:
            break

        # Timed out — retry
        print(f"  Timed out after {{MAX_WAIT_PER_RUN//60}} min (attempt {{attempt}}).", flush=True)
        if attempt < MAX_RETRIES:
            print(f"  Retrying...", flush=True)
            time.sleep(10)

    if not success and constitution not in [f for f in failed]:
        failed.append(constitution)

print(f"\\n--- Results ---", flush=True)
print(f"Completed: {{completed}}", flush=True)
if failed:
    print(f"Failed: {{failed}}", flush=True)

# --- Build and upload matrix ---
CONSTITUTIONS = {json.dumps(CONSTITUTIONS)}

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
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of constitutions to run (e.g., 'sarcasm,sycophancy')")
    args = parser.parse_args()

    # Filter constitutions if --only is provided
    global CONSTITUTIONS
    if args.only:
        wanted = [c.strip() for c in args.only.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in CONSTITUTIONS]
        if unknown:
            raise SystemExit(f"Unknown constitutions: {unknown}. Valid: {CONSTITUTIONS}")
        CONSTITUTIONS = wanted
        print(f"Running only: {CONSTITUTIONS}")

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
