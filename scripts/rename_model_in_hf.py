#!/usr/bin/env python3
"""Rename a model nick across all files in the HF ValueArena dataset.

Walks the HF dataset, downloads every relevant file (summary.json, meta.json,
evaluations.jsonl), replaces the old model name with the new one, and uploads
back. Idempotent — running twice is safe.

Usage:
    python scripts/rename_model_in_hf.py --old gemini-2.5-pro --new gemini-2.5-flash
    python scripts/rename_model_in_hf.py --old gemini-2.5-pro --new gemini-2.5-flash --dry-run
    python scripts/rename_model_in_hf.py --old gemini-2.5-pro --new gemini-2.5-flash --group prompted
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HF_REPO = "invi-bhagyesh/ValueArena"


def list_run_slugs(group_filter: str | None = None) -> list[str]:
    """List all run slugs (group/constitution) on HF."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    slugs = set()
    for f in files:
        # Match runs/<group>/<constitution>/...
        parts = f.split("/")
        if len(parts) >= 4 and parts[0] == "runs":
            slug = f"{parts[1]}/{parts[2]}"
            if group_filter and not slug.startswith(group_filter + "/"):
                continue
            slugs.add(slug)
    return sorted(slugs)


def fix_summary_json(content: bytes, old: str, new: str) -> tuple[bytes, int]:
    """Fix model_name in summary.json. Returns (new_content, num_changes)."""
    data = json.loads(content)
    n = 0
    for entry in data:
        if entry.get("model_name") == old:
            entry["model_name"] = new
            n += 1
    return json.dumps(data, indent=2).encode("utf-8"), n


def fix_meta_json(content: bytes, old: str, new: str) -> tuple[bytes, int]:
    """Fix model nicks in meta.json (spec.models dict + any other refs)."""
    data = json.loads(content)
    n = 0

    # Top-level models dict
    if isinstance(data.get("models"), dict) and old in data["models"]:
        data["models"][new] = data["models"].pop(old)
        n += 1

    # spec.models dict
    spec = data.get("spec", {})
    if isinstance(spec.get("models"), dict) and old in spec["models"]:
        spec["models"][new] = spec["models"].pop(old)
        n += 1

    return json.dumps(data, indent=2).encode("utf-8"), n


def fix_evaluations_jsonl(content: bytes, old: str, new: str) -> tuple[bytes, int]:
    """Fix model names in evaluations.jsonl. Each line is a JSON record with
    eval1_name, eval2_name, judge_name fields."""
    n = 0
    out_lines = []
    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            out_lines.append(line)
            continue
        rec = json.loads(line)
        for k in ("eval1_name", "eval2_name", "judge_name"):
            if rec.get(k) == old:
                rec[k] = new
                n += 1
        out_lines.append(json.dumps(rec, ensure_ascii=True))
    return ("\n".join(out_lines) + "\n").encode("utf-8"), n


FILE_HANDLERS = {
    "summary.json": fix_summary_json,
    "meta.json": fix_meta_json,
    "evaluations.jsonl": fix_evaluations_jsonl,
}


def process_run(slug: str, old: str, new: str, dry_run: bool) -> dict:
    """Download, fix, and re-upload all files for a run. Returns stats."""
    from huggingface_hub import hf_hub_download, upload_file

    stats = {"slug": slug, "files_changed": 0, "items_changed": 0}

    for filename, handler in FILE_HANDLERS.items():
        path_in_repo = f"runs/{slug}/{filename}"
        try:
            local_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=path_in_repo,
                repo_type="dataset",
                force_download=True,
            )
        except Exception as e:
            # File doesn't exist for this run — skip silently
            if "404" in str(e) or "EntryNotFoundError" in type(e).__name__:
                continue
            print(f"  {slug}/{filename}: download failed ({e})")
            continue

        with open(local_path, "rb") as f:
            content = f.read()

        try:
            new_content, n_items = handler(content, old, new)
        except Exception as e:
            print(f"  {slug}/{filename}: parse failed ({e})")
            continue

        if n_items == 0:
            continue

        stats["files_changed"] += 1
        stats["items_changed"] += n_items
        print(f"  {slug}/{filename}: {n_items} occurrences")

        if dry_run:
            continue

        # Upload back
        with tempfile.NamedTemporaryFile(mode="wb", suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(new_content)
            tmp_path = tmp.name

        try:
            upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=path_in_repo,
                repo_id=HF_REPO,
                repo_type="dataset",
                commit_message=f"rename {old} → {new} in {slug}/{filename}",
            )
        finally:
            os.unlink(tmp_path)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Rename a model nick across all HF dataset files")
    parser.add_argument("--old", required=True, help="Old model name (e.g., gemini-2.5-pro)")
    parser.add_argument("--new", required=True, help="New model name (e.g., gemini-2.5-flash)")
    parser.add_argument("--group", default=None, help="Limit to one group (e.g., prompted)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, don't upload")
    args = parser.parse_args()

    print(f"Renaming '{args.old}' → '{args.new}' in HF dataset {HF_REPO}")
    if args.dry_run:
        print("(DRY RUN — no uploads)")
    if args.group:
        print(f"(restricted to group: {args.group})")
    print()

    slugs = list_run_slugs(group_filter=args.group)
    print(f"Found {len(slugs)} run slugs to scan")
    print()

    total_files = 0
    total_items = 0
    for slug in slugs:
        s = process_run(slug, args.old, args.new, args.dry_run)
        total_files += s["files_changed"]
        total_items += s["items_changed"]

    print()
    print(f"DONE. Modified {total_files} files, {total_items} occurrences.")
    if args.dry_run:
        print("(dry run — re-run without --dry-run to apply)")


if __name__ == "__main__":
    main()
