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
    REF_ANCHOR,
    REF_NICKS,
    find_model_nick,
    plot_ci_matrix,
    plot_grouped_matrix,
    plot_matrix,
    save_csv,
)


HF_REPO = "invi-bhagyesh/ValueArena"


def fetch_summary_from_hf(
    group: str,
    constitution: str,
    repo_id: str = HF_REPO,
    slug_template: str = "{group}/{c}",
) -> dict | None:
    """Fetch summary.json for a run from HuggingFace.

    slug_template controls how (group, constitution) map to the HF run slug.
    Default "{group}/{c}" matches the standard layout (e.g. prompted/goodness).
    Use e.g. "loving-cross-{c}" for groups whose runs are top-level slugs.
    """
    from huggingface_hub import hf_hub_download
    slug = slug_template.format(group=group, c=constitution)
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"runs/{slug}/summary.json",
            repo_type="dataset",
            force_download=True,
        )
        with open(path) as f:
            data = json.load(f)
        return {entry["model_name"]: entry for entry in data}
    except Exception:
        return None


def poll_all_summaries(
    group: str, repo_id: str = HF_REPO, interval: int = 120, max_wait: int = 7200,
    slug_template: str = "{group}/{c}",
    constitutions: list[str] | None = None,
) -> dict[str, dict]:
    """Poll HF until all constitution summaries are available."""
    cs = constitutions or CONSTITUTIONS
    elapsed = 0
    while elapsed < max_wait:
        summaries = {}
        missing = []
        for c in cs:
            bs = fetch_summary_from_hf(group, c, repo_id, slug_template)
            if bs:
                summaries[c] = bs
            else:
                missing.append(c)

        if not missing:
            print(f"All {len(cs)} summaries ready.")
            return summaries

        print(f"  {len(summaries)}/{len(cs)} ready. Missing: {', '.join(missing)}")
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

    # Preserve summaries' key order so users can pass --constitutions in
    # whatever order they want the rows displayed.
    constitutions = list(summaries.keys())
    N = len(constitutions)

    col_labels = constitutions + [BASE_NICK]
    M = len(col_labels)
    A_mean = np.full((N, M), np.nan)
    A_std = np.full((N, M), np.nan)

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
        offset = REF_ANCHOR - ref_mean

        for j, cj in enumerate(constitutions):
            nick = find_model_nick(bs, cj, nick_prefix)
            if nick and nick in bs:
                A_mean[i, j] = bs[nick]["elo_mean"] + offset
                A_std[i, j] = bs[nick]["elo_std"]

        if BASE_NICK in bs:
            A_mean[i, N] = bs[BASE_NICK]["elo_mean"] + offset
            A_std[i, N] = bs[BASE_NICK]["elo_std"]

    return A_mean, A_std, constitutions, col_labels


def build_all_models_matrix_from_hf(
    summaries: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build matrix with ALL non-reference models as columns.

    Rows: constitutions (eval dimension), in the order they appear in
    ``summaries`` (i.e. the order the caller passed via --constitutions).
    Cols: every unique model nick across all summaries, excluding REF_NICKS.
    """
    # Preserve summaries' key order so users can pass --constitutions in
    # whatever order they want the rows displayed.
    constitutions = list(summaries.keys())
    N = len(constitutions)

    # Collect all unique non-reference model nicks across summaries.
    # Treat any gemini variant (flash/pro) as a reference — some runs use flash, others pro.
    def _is_ref(nick: str) -> bool:
        if nick in REF_NICKS:
            return True
        if not nick:
            return False
        first = nick.lower().replace("-", " ").split()[0]
        return first in {"gemini", "gpt", "claude"}

    all_nicks: list[str] = []
    seen = set()
    for c in constitutions:
        for nick in summaries[c]:
            if _is_ref(nick) or nick in seen:
                continue
            seen.add(nick)
            all_nicks.append(nick)

    # Stable order: base first, then dpo-*, introspection-*, prompted_*, then rest
    def _sort_key(nick: str) -> tuple[int, str]:
        if nick == BASE_NICK:
            return (0, nick)
        if nick.startswith("dpo"):
            return (1, nick)
        if nick.startswith("introspection"):
            return (2, nick)
        if nick.startswith("prompted"):
            return (3, nick)
        return (4, nick)

    col_labels = sorted(all_nicks, key=_sort_key)
    M = len(col_labels)
    A_mean = np.full((N, M), np.nan)
    A_std = np.full((N, M), np.nan)

    for i, ci in enumerate(constitutions):
        bs = summaries[ci]
        ref_elos = [bs[n]["elo_mean"] for n in bs if _is_ref(n)]
        if not ref_elos:
            if BASE_NICK in bs:
                ref_elos = [bs[BASE_NICK]["elo_mean"]]
            else:
                print(f"  {ci}: no reference models — skipping row")
                continue
        ref_mean = sum(ref_elos) / len(ref_elos)
        offset = REF_ANCHOR - ref_mean

        for j, nick in enumerate(col_labels):
            if nick in bs:
                A_mean[i, j] = bs[nick]["elo_mean"] + offset
                A_std[i, j] = bs[nick]["elo_std"]

    return A_mean, A_std, constitutions, col_labels


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
    parser.add_argument("--all-models", action="store_true",
                        help="Include all non-reference models as columns (not just per-constitution variants)")
    parser.add_argument("--slug-template", default="{group}/{c}",
                        help="Slug template, e.g. 'loving-cross-{c}' for cross-constitution. Default: '{group}/{c}'")
    parser.add_argument("--upload-as", default=None,
                        help="Group folder to upload matrix files under. Defaults to <group>.")
    parser.add_argument("--constitutions", default=None,
                        help="Comma-separated subset of constitutions (default: all 11)")
    parser.add_argument("--column-filter", default=None,
                        help="Comma-separated substrings; keep only columns whose label contains "
                             "ANY of these substrings. 'base' is always kept. Used with --all-models.")
    parser.add_argument("--filename-suffix", default="",
                        help="Suffix added to output filenames (e.g. '_loving' -> matrix_view_loving.png). "
                             "Useful when uploading multiple subset matrices into the same group.")
    parser.add_argument("--column-groups", default=None,
                        help="Pipe-separated groups; each group is 'name=substr1,substr2,...'. "
                             "Renders one combined PNG with sub-matrices side-by-side and whitespace "
                             "gaps. 'base' becomes its own leading group. Example: "
                             "'prompted=prompted_|trained_loving=trained_loving|trained_misalignment="
                             "trained_misalignment|trained_sarcasm=trained_sarcasm'.")
    args = parser.parse_args()

    cs_list = [c.strip() for c in args.constitutions.split(",")] if args.constitutions else CONSTITUTIONS

    if args.poll:
        summaries = poll_all_summaries(
            args.group, args.repo, args.poll_interval, args.max_wait,
            slug_template=args.slug_template, constitutions=cs_list,
        )
    else:
        summaries = {}
        for c in cs_list:
            bs = fetch_summary_from_hf(args.group, c, args.repo, args.slug_template)
            if bs:
                summaries[c] = bs
            else:
                print(f"  {c}: not found on HF — skipping")
        if not summaries:
            print("No summaries found. Runs may not have finished yet. Use --poll to wait.")
            sys.exit(1)

    print(f"\nBuilding matrix from {len(summaries)} constitutions...")
    if args.all_models:
        A_mean, A_std, constitutions, col_labels = build_all_models_matrix_from_hf(summaries)
        print(f"All-models columns: {col_labels}")
    else:
        A_mean, A_std, constitutions, col_labels = build_matrix_from_hf(summaries, args.nick_prefix)

    if args.column_filter:
        substrings = [s.strip() for s in args.column_filter.split(",") if s.strip()]
        keep_idx = [
            j for j, label in enumerate(col_labels)
            if label == BASE_NICK or any(s in label for s in substrings)
        ]
        if not keep_idx:
            print("No columns match --column-filter; aborting.")
            sys.exit(1)
        A_mean = A_mean[:, keep_idx]
        A_std = A_std[:, keep_idx]
        col_labels = [col_labels[j] for j in keep_idx]
        print(f"After --column-filter: kept {len(col_labels)} columns: {col_labels}")

    suffix = args.filename_suffix
    title_tag = f" ({suffix.strip('_')})" if suffix else ""

    upload_group = args.upload_as or args.group
    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir)
        plot_matrix(A_mean, A_std, constitutions, staging / f"matrix_view{suffix}.png",
                    col_labels=col_labels,
                    title=f"Character-Train Matrix — {upload_group}{title_tag} (Elo, API avg = {REF_ANCHOR})")
        plot_ci_matrix(A_std, constitutions, staging / f"matrix_ci{suffix}.png",
                       col_labels=col_labels,
                       title=f"Character-Train Matrix — {upload_group}{title_tag} (CI Width)")
        save_csv(A_mean, constitutions, staging / f"matrix_view{suffix}.csv", col_labels=col_labels)

        # Optional grouped variant: split columns into named blocks side-by-side
        if args.column_groups:
            groups: list[tuple[str, np.ndarray, np.ndarray, list[str]]] = []
            if BASE_NICK in col_labels:
                bj = col_labels.index(BASE_NICK)
                groups.append((BASE_NICK,
                               A_mean[:, bj:bj + 1], A_std[:, bj:bj + 1],
                               [BASE_NICK]))

            for spec in args.column_groups.split("|"):
                spec = spec.strip()
                if not spec:
                    continue
                if "=" not in spec:
                    print(f"Skipping malformed group spec: {spec!r}")
                    continue
                g_name, substr_csv = spec.split("=", 1)
                g_name = g_name.strip()
                substrs = [s.strip() for s in substr_csv.split(",") if s.strip()]
                idxs = [
                    j for j, label in enumerate(col_labels)
                    if label != BASE_NICK and any(s in label for s in substrs)
                ]
                if not idxs:
                    print(f"  Group '{g_name}': no matching columns — skipping")
                    continue
                # Strip the longest matching substring from each label so the
                # group's column ticks aren't redundant with the group title.
                # E.g. inside group "trained_loving" with substr "trained_loving",
                # "trained_loving__prompt_sarcasm" -> "prompt_sarcasm" -> "sarcasm".
                def _short_label(full: str) -> str:
                    out = full
                    for s in sorted(substrs, key=len, reverse=True):
                        if out.startswith(s):
                            out = out[len(s):]
                        elif s in out:
                            out = out.replace(s, "", 1)
                    out = out.strip("_- ")
                    if out.startswith("prompt_"):
                        out = out[len("prompt_"):]
                    return out or full

                short_labels = [_short_label(col_labels[j]) for j in idxs]
                groups.append((
                    g_name,
                    A_mean[:, idxs],
                    A_std[:, idxs],
                    short_labels,
                ))

            if groups:
                plot_grouped_matrix(
                    groups, constitutions,
                    staging / f"matrix_view_grouped{suffix}.png",
                    title=f"Character-Train Matrix — {upload_group}{title_tag}",
                )

        if not args.no_upload:
            upload_matrix_to_hf(upload_group, staging, args.repo)
        else:
            # Copy to local runs dir
            out_dir = _REPO_ROOT / "runs" / upload_group
            out_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            for f in staging.iterdir():
                shutil.copy2(f, out_dir / f.name)
            print(f"Saved locally to {out_dir}")


if __name__ == "__main__":
    main()
