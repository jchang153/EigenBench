#!/usr/bin/env python3
"""Train BTD + bootstrap locally for missing constitutions, then upload to HF.

Skips constitutions already on HF. Trains the rest locally (CPU), then uploads
each to HF one at a time (so a crash on run N doesn't lose runs 1..N-1).

With --parallel N, runs up to N training jobs concurrently (subprocesses).
Uploads are always serialized afterward to avoid HF API rate limits.

Usage:
    python scripts/run_local_train_upload.py --group openchar
    python scripts/run_local_train_upload.py --group openchar --parallel 4
    python scripts/run_local_train_upload.py --group prompted --only sarcasm,sycophancy
    python scripts/run_local_train_upload.py --group openchar --dry-run
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Strip SOCKS proxy env vars that break huggingface_hub (requires socksio package).
for _k in list(os.environ):
    if _k.lower() in ("all_proxy", "ftp_proxy", "grpc_proxy", "rsync_proxy"):
        os.environ.pop(_k, None)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


def summary_exists_on_hf(group: str, constitution: str) -> bool:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(
            repo_id="invi-bhagyesh/ValueArena",
            repo_type="dataset",
        )
        target = f"runs/{group}/{constitution}/summary.json"
        return target in files
    except Exception as e:
        print(f"  HF check failed for {group}/{constitution}: {e}")
        return False


def train_one(spec_path: Path) -> bool:
    """Train BTD + bootstrap locally. Returns True on success."""
    try:
        # Ensure scripts/ is importable for run_train
        scripts_dir = str(_REPO_ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from run_train import main as run_train_main
        run_train_main(str(spec_path))
        return True
    except Exception as e:
        print(f"  Training FAILED: {e}")
        traceback.print_exc()
        return False


def train_one_subprocess(spec_path: Path, log_path: Path, python_exe: str) -> tuple[str, int]:
    """Train via subprocess (for parallel execution). Returns (spec_name, returncode)."""
    # Invoke via `python scripts/run_train.py <spec>` — but run_train is a module,
    # so use a tiny inline wrapper.
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r});"
        f"sys.path.insert(0, {str(_REPO_ROOT / 'scripts')!r});"
        "from run_train import main;"
        f"main({str(spec_path)!r})"
    )
    with open(log_path, "w") as logf:
        logf.write(f"# Training {spec_path}\n")
        logf.flush()
        proc = subprocess.run(
            [python_exe, "-u", "-c", code],
            stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(_REPO_ROOT),
        )
    return (spec_path.parent.name, proc.returncode)


def upload_one(run_dir: Path, slug: str, note: str | None) -> bool:
    """Upload a single trained run to HF. Returns True on success."""
    try:
        scripts_dir = str(_REPO_ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from upload_results import upload_run
        upload_run(slug, run_dir, "invi-bhagyesh/ValueArena")
        return True
    except Exception as e:
        print(f"  Upload FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Train locally + upload missing constitutions")
    parser.add_argument("--group", required=True, choices=["openchar", "prompted"])
    parser.add_argument("--only", default=None, help="Comma-separated subset")
    parser.add_argument("--dry-run", action="store_true", help="Show plan, don't run")
    parser.add_argument("--force", action="store_true", help="Re-train + re-upload even if on HF")
    parser.add_argument("--no-upload", action="store_true", help="Train only, skip upload")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel train workers (default: 1, sequential)")
    parser.add_argument("--skip-hf-check", action="store_true",
                        help="Skip HF pre-check (treat all as pending, trust local btd_d2 presence)")
    args = parser.parse_args()

    constitutions = CONSTITUTIONS
    if args.only:
        wanted = [c.strip() for c in args.only.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in CONSTITUTIONS]
        if unknown:
            raise SystemExit(f"Unknown: {unknown}")
        constitutions = wanted

    specs_dir = _REPO_ROOT / "runs" / args.group

    # Determine which need training
    print(f"Checking {'local only' if args.skip_hf_check else 'HF'} for existing {args.group}/* runs...")
    pending = []
    for c in constitutions:
        spec_path = specs_dir / c / "spec.py"
        if not spec_path.exists():
            print(f"  {c}: SKIP (no spec.py)")
            continue

        eval_path = specs_dir / c / "evaluations.jsonl"
        if not eval_path.exists():
            print(f"  {c}: SKIP (no evaluations.jsonl)")
            continue

        if not args.force and not args.skip_hf_check and summary_exists_on_hf(args.group, c):
            print(f"  {c}: SKIP (already on HF)")
            continue

        local_btd = specs_dir / c / "btd_d2"
        has_local = (local_btd / "bootstrap" / "summary.json").exists()
        status = "needs upload (trained)" if has_local else "needs train + upload"
        print(f"  {c}: {status}")
        pending.append((c, spec_path, has_local))

    if not pending:
        print("\nNothing to do.")
        return

    print(f"\n{len(pending)} runs pending.")
    if args.parallel > 1:
        print(f"Training with {args.parallel} parallel workers.")
    if args.dry_run:
        return

    # Split into "needs train" vs "just needs upload"
    needs_train = [(c, sp) for c, sp, has_local in pending if not has_local]
    needs_upload_only = [(c, sp) for c, sp, has_local in pending if has_local]

    trained_ok = []
    trained_fail = []

    # ── Training phase ──
    if needs_train:
        logs_dir = _REPO_ROOT / "logs" / f"{args.group}_{int(time.time())}"
        logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nTraining logs: {logs_dir}")

        python_exe = sys.executable
        t_start = time.time()

        if args.parallel > 1:
            # Parallel subprocess fan-out
            print(f"\n{'=' * 60}")
            print(f"Training {len(needs_train)} runs with {args.parallel} workers")
            print(f"{'=' * 60}")

            with mp.Pool(processes=args.parallel) as pool:
                async_results = []
                for c, sp in needs_train:
                    log_path = logs_dir / f"{c}.log"
                    print(f"  Queued: {c} (log: {log_path})")
                    async_results.append((
                        c,
                        pool.apply_async(train_one_subprocess, (sp, log_path, python_exe)),
                    ))

                for c, async_res in async_results:
                    try:
                        name, rc = async_res.get()
                        if rc == 0:
                            print(f"  DONE: {c}")
                            trained_ok.append(c)
                        else:
                            print(f"  FAILED: {c} (rc={rc}, see logs/{c}.log)")
                            trained_fail.append(c)
                    except Exception as e:
                        print(f"  FAILED: {c} ({e})")
                        trained_fail.append(c)
        else:
            # Sequential in-process
            for idx, (c, sp) in enumerate(needs_train, 1):
                print(f"\n[{idx}/{len(needs_train)}] Training {args.group}/{c}")
                t0 = time.time()
                ok = train_one(sp)
                dt = (time.time() - t0) / 60
                if ok:
                    print(f"  Done ({dt:.1f}min)")
                    trained_ok.append(c)
                else:
                    print(f"  FAILED ({dt:.1f}min)")
                    trained_fail.append(c)

        dt_total = (time.time() - t_start) / 60
        print(f"\nTraining phase done: {len(trained_ok)}/{len(needs_train)} ok ({dt_total:.1f}min)")

    # ── Upload phase (always serial) ──
    if args.no_upload:
        print("\nSkipping uploads (--no-upload).")
        print(f"\nCompleted ({len(trained_ok)}): {trained_ok}")
        if trained_fail:
            print(f"Train failed ({len(trained_fail)}): {trained_fail}")
        return

    to_upload = [(c, specs_dir / c) for c in trained_ok]
    to_upload += [(c, specs_dir / c) for c, _ in needs_upload_only]

    if not to_upload:
        print("\nNothing to upload.")
        return

    print(f"\n{'=' * 60}")
    print(f"Uploading {len(to_upload)} runs to HF (serial)")
    print(f"{'=' * 60}")

    uploaded_ok = []
    uploaded_fail = []
    for idx, (c, run_dir) in enumerate(to_upload, 1):
        slug = f"{args.group}/{c}"
        print(f"\n[{idx}/{len(to_upload)}] Uploading {slug}")
        t0 = time.time()
        ok = upload_one(run_dir, slug, None)
        dt = (time.time() - t0) / 60
        if ok:
            print(f"  Done ({dt:.1f}min)")
            uploaded_ok.append(c)
        else:
            print(f"  FAILED ({dt:.1f}min)")
            uploaded_fail.append(c)

    print(f"\n{'=' * 60}")
    print(f"Summary for {args.group}:")
    print(f"  Uploaded: {uploaded_ok}")
    if trained_fail:
        print(f"  Train failed: {trained_fail}")
    if uploaded_fail:
        print(f"  Upload failed: {uploaded_fail}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
