#!/usr/bin/env python3
"""Build and upload the character-train matrix from HF-hosted run summaries.

Polls HuggingFace for all per-constitution summary.json files under a given
group prefix (e.g., 'prompted'), builds the Elo-vs-base heatmap, and uploads
the matrix_view.png + matrix_view.csv to the group folder on HF.

Usage:
    python scripts/upload_matrix.py prompted
    python scripts/upload_matrix.py prompted --poll          # wait until all 11 are ready
    python scripts/upload_matrix.py matrix --base-nick base
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from build_matrix import (
    BASE_NICK,
    CONSTITUTIONS,
    REF_NICKS,
    find_model_nick,
    plot_matrix,
    save_csv,
)

HF_REPO = "invi-bhagyesh/ValueArena"


def fetch_summary_from_hf(group: str, constitution: str, repo_id: str = HF_REPO) -> dict | None:
    """Fetch summary.json for a run from HuggingFace."""
    from huggingface_hub import hf_hub_download
    slug = f"{group}/{constitution}"
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"runs/{slug}/summary.json",
            repo_type="dataset",
            force_download=True,
        )
        with open(path) as f:
            data = json.load(f)
        # summary.json is a list of {model_name, elo_mean, elo_std, ...}
        # Convert to dict keyed by model_name
        return {entry["model_name"]: entry for entry in data}
    except Exception:
        return None


def poll_all_summaries(
    group: str, repo_id: str = HF_REPO, interval: int = 120, max_wait: int = 7200,
) -> dict[str, dict]:
    """Poll HF until all constitution summaries are available."""
    elapsed = 0
    while elapsed < max_wait:
        summaries = {}
        missing = []
        for c in CONSTITUTIONS:
            bs = fetch_summary_from_hf(group, c, repo_id)
            if bs:
                summaries[c] = bs
            else:
                missing.append(c)

        if not missing:
            print(f"All {len(CONSTITUTIONS)} summaries ready.")
            return summaries

        print(f"  {len(summaries)}/{len(CONSTITUTIONS)} ready. Missing: {', '.join(missing)}")
        if elapsed + interval >= max_wait:
            break
        print(f"  Retrying in {interval}s...")
        time.sleep(interval)
        elapsed += interval

    print(f"Timed out after {max_wait}s. Got {len(summaries)}/{len(CONSTITUTIONS)}.")
    if not summaries:
        sys.exit(1)
    return summaries


def build_matrix_from_hf(
    summaries: dict[str, dict], nick_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build matrix from HF-fetched summaries."""
    # Auto-detect nick prefix
    if nick_prefix is None:
        for c, bs in summaries.items():
            nicks = [n for n in bs if n != BASE_NICK and c in n.lower()]
            if nicks:
                nick = nicks[0]
                nick_prefix = nick[:nick.lower().index(c)]
                break
        if nick_prefix is None:
            print("Could not auto-detect nick prefix. Use --nick-prefix.")
            sys.exit(1)
    print(f"Nick prefix: '{nick_prefix}'")

    constitutions = [c for c in CONSTITUTIONS if c in summaries]
    N = len(constitutions)

    A_mean = np.full((N, N), np.nan)
    A_std = np.full((N, N), np.nan)

    for i, ci in enumerate(constitutions):
        bs = summaries[ci]
        ref_elos = [bs[r]["elo_mean"] for r in REF_NICKS if r in bs]
        if not ref_elos:
            if BASE_NICK in bs:
                ref_elos = [bs[BASE_NICK]["elo_mean"]]
            else:
                print(f"  {ci}: no reference models — skipping row")
                continue
        ref_mean = sum(ref_elos) / len(ref_elos)

        for j, cj in enumerate(constitutions):
            nick = find_model_nick(bs, cj, nick_prefix)
            if nick and nick in bs:
                A_mean[i, j] = bs[nick]["elo_mean"] - ref_mean
                A_std[i, j] = bs[nick]["elo_std"]

    return A_mean, A_std, constitutions


def upload_matrix_to_hf(group: str, staging_dir: Path, repo_id: str = HF_REPO):
    """Upload matrix files to HF dataset repo."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    files = sorted(f for f in staging_dir.rglob("*") if f.is_file())
    operations = []
    for fpath in files:
        repo_path = f"runs/{group}/{fpath.name}"
        print(f"  Uploading {repo_path}")
        operations.append(CommitOperationAdd(
            path_in_repo=repo_path,
            path_or_fileobj=fpath.read_bytes(),
        ))

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Add character-train matrix for {group}",
    )
    print(f"Done! https://huggingface.co/datasets/{repo_id}/tree/main/runs/{group}")


def main():
    parser = argparse.ArgumentParser(description="Build and upload character-train matrix from HF")
    parser.add_argument("group", help="Run group prefix (e.g., 'prompted', 'matrix')")
    parser.add_argument("--nick-prefix", default=None, help="Model nick prefix (auto-detected if omitted)")
    parser.add_argument("--poll", action="store_true", help="Poll until all summaries are ready")
    parser.add_argument("--poll-interval", type=int, default=120, help="Poll interval in seconds (default: 120)")
    parser.add_argument("--max-wait", type=int, default=7200, help="Max poll wait in seconds (default: 7200)")
    parser.add_argument("--repo", default=HF_REPO, help="HF dataset repo")
    parser.add_argument("--no-upload", action="store_true", help="Build locally only, don't upload")
    args = parser.parse_args()

    if args.poll:
        summaries = poll_all_summaries(args.group, args.repo, args.poll_interval, args.max_wait)
    else:
        summaries = {}
        for c in CONSTITUTIONS:
            bs = fetch_summary_from_hf(args.group, c, args.repo)
            if bs:
                summaries[c] = bs
            else:
                print(f"  {c}: not found on HF — skipping")
        if not summaries:
            print("No summaries found. Runs may not have finished yet. Use --poll to wait.")
            sys.exit(1)

    print(f"\nBuilding matrix from {len(summaries)} constitutions...")
    A_mean, A_std, constitutions = build_matrix_from_hf(summaries, args.nick_prefix)

    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir)
        plot_matrix(A_mean, A_std, constitutions, staging / "matrix_view.png",
                    title=f"Character-Train Matrix — {args.group} (Elo vs Base)")
        save_csv(A_mean, constitutions, staging / "matrix_view.csv")

        if not args.no_upload:
            upload_matrix_to_hf(args.group, staging, args.repo)
        else:
            # Copy to local runs dir
            out_dir = _REPO_ROOT / "runs" / args.group
            out_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            for f in staging.iterdir():
                shutil.copy2(f, out_dir / f.name)
            print(f"Saved locally to {out_dir}")


if __name__ == "__main__":
    main()
