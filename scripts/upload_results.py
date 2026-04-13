"""
Upload EigenBench run results to HuggingFace dataset repo for ValueArena.

Usage:
    python scripts/upload_results.py --name "persona-goodness" --run-dir runs/matrix_new/goodness/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_spec(spec_path: Path) -> dict:
    """Parse a spec.py file and return the RUN_SPEC dict."""
    namespace = {"min": min, "max": max, "bool": bool, "True": True, "False": False}
    with open(spec_path) as f:
        exec(f.read(), namespace)
    return namespace["RUN_SPEC"]


def parse_log_train(log_path: Path) -> dict:
    """Parse log_train.txt into a dict of numeric values."""
    result = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                try:
                    result[key] = int(val)
                except ValueError:
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val
    return result


def parse_eigentrust(et_path: Path) -> list[float]:
    """Parse eigentrust.txt into a list of floats."""
    text = et_path.read_text()
    numbers = re.findall(r"[\d.]+(?:e[+-]?\d+)?", text)
    return [float(x) for x in numbers]


def detect_model_type(model_id: str) -> dict:
    """Detect model type from spec model ID string."""
    if model_id.startswith("hf_local:"):
        hf_path = model_id[len("hf_local:"):]
        parts = hf_path.split("/")
        if len(parts) >= 3:
            # hf_local:org/repo/subfolder -> LoRA
            base_repo = "/".join(parts[:2])
            adapter = hf_path
            return {"id": model_id, "type": "lora", "base_model": base_repo, "adapter": adapter}
        else:
            # hf_local:org/repo -> base model
            return {"id": model_id, "type": "base", "base_model": hf_path, "adapter": None}
    else:
        # provider/model -> API model
        return {"id": model_id, "type": "api", "base_model": None, "adapter": None}


def get_git_info(repo_dir: Path) -> tuple[str | None, str | None]:
    """Get current git commit hash and remote URL."""
    commit = None
    repo_url = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=repo_dir, text=True
        ).strip()
        # Convert SSH to HTTPS format
        if remote.startswith("git@"):
            remote = remote.replace(":", "/").replace("git@", "https://")
        repo_url = remote.removesuffix(".git")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return commit, repo_url


def build_summary_from_eigentrust(et_scores: list[float], model_names: list[str]) -> list[dict]:
    """Build a summary.json-compatible list from eigentrust scores (no bootstrap CI)."""
    import math
    n = len(model_names)
    rows = []
    for i, name in enumerate(model_names):
        trust = et_scores[i] if i < len(et_scores) else 0.0
        elo = 1500.0 + 400.0 * math.log10(max(n * trust, 1e-12))
        rows.append({
            "model_index": i,
            "model_name": name,
            "elo_mean": elo,
            "elo_std": 0.0,
            "elo_ci_lower": elo,
            "elo_ci_upper": elo,
        })
    rows.sort(key=lambda r: r["elo_mean"], reverse=True)
    return rows


def find_btd_dir(run_dir: Path) -> Path | None:
    """Find the btd_d* output directory (picks first match)."""
    candidates = sorted(run_dir.glob("btd_d*"))
    return candidates[0] if candidates else None


def build_meta(
    name: str,
    spec: dict,
    log: dict,
    eigentrust: list[float],
    git_commit: str | None,
    git_repo: str | None,
) -> dict:
    """Build the meta.json dict from parsed components."""
    models = {}
    for model_name, model_id in spec.get("models", {}).items():
        models[model_name] = detect_model_type(model_id)

    meta = {
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_repo": git_repo,
        "models": models,
        "dataset": spec.get("dataset", {}),
        "constitution": spec.get("constitution", {}),
        "training": {
            k: v for k, v in spec.get("training", {}).items()
            if k != "bootstrap" and k != "enabled"
        },
        "collection": {
            k: v for k, v in spec.get("collection", {}).items()
            if k not in ("enabled", "evaluations_path", "cached_responses_path")
        },
        "bootstrap": spec.get("training", {}).get("bootstrap", {}),
        "log": log,
        "eigentrust": eigentrust,
    }
    return meta


def stage_run(name: str, run_dir: Path, staging_dir: Path) -> tuple[dict, Path]:
    """Stage a single run's files into a local directory for upload.

    Returns (meta_dict, summary_path).
    """
    spec_path = run_dir / "spec.py"
    if not spec_path.exists():
        raise FileNotFoundError(f"{spec_path} not found")

    btd_dir = find_btd_dir(run_dir)
    if btd_dir is None:
        raise FileNotFoundError(f"No btd_d* directory found in {run_dir}")

    print(f"  Parsing {run_dir.name}")
    spec = parse_spec(spec_path)

    log_path = btd_dir / "log_train.txt"
    log = parse_log_train(log_path) if log_path.exists() else {}

    et_path = btd_dir / "eigentrust.txt"
    eigentrust = parse_eigentrust(et_path) if et_path.exists() else []

    summary_path = btd_dir / "bootstrap" / "summary.json"
    has_bootstrap = summary_path.exists()

    git_commit, git_repo = get_git_info(run_dir)
    meta = build_meta(name, spec, log, eigentrust, git_commit, git_repo)

    # Stage files
    dest = staging_dir / "runs" / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "images").mkdir(exist_ok=True)

    with open(dest / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    import shutil
    if has_bootstrap:
        shutil.copy2(summary_path, dest / "summary.json")
    else:
        # Build summary from eigentrust scores
        model_names = list(spec.get("models", {}).keys())
        summary_data = build_summary_from_eigentrust(eigentrust, model_names)
        with open(dest / "summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)
        summary_path = dest / "summary.json"

    image_files = {
        "eigenbench.png": btd_dir / "eigenbench.png",
        "training_loss.png": btd_dir / "training_loss.png",
        "uv_embeddings_pca.png": btd_dir / "uv_embeddings_pca.png",
        "bootstrap_elo.png": btd_dir / "bootstrap" / "bootstrap_elo.png",
    }
    for img_name, img_path in image_files.items():
        if img_path.exists():
            shutil.copy2(img_path, dest / "images" / img_name)

    # Evaluations
    eval_path = run_dir / "evaluations.jsonl"
    if eval_path.exists():
        shutil.copy2(eval_path, dest / "evaluations.jsonl")

    return meta, summary_path


def upload_run(name: str, run_dir: Path, repo_id: str, token: str | None = None):
    """Upload a single run's results to HuggingFace."""
    import tempfile
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir)
        meta, summary_path = stage_run(name, run_dir, staging)

        print(f"Uploading {name}...")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add run: {name}",
        )

    # Update index
    update_index(name, name, meta, summary_path, repo_id, api)
    print(f"Done! https://huggingface.co/datasets/{repo_id}/tree/main/runs/{name}")


def build_index_entry(name: str, meta: dict, summary_path: Path, group: str | None = None, note: str | None = None) -> dict:
    """Build a single index.json entry with all spec details."""
    with open(summary_path) as f:
        summary = json.load(f)

    top = summary[0] if summary else {}
    constitution_path = meta.get("constitution", {}).get("path", "")
    constitution_name = Path(constitution_path).stem if constitution_path else ""
    constitution_name = constitution_name.removeprefix("oct_")
    dataset_path = meta.get("dataset", {}).get("path", "")
    scenario_name = Path(dataset_path).stem if dataset_path else ""
    scenario_name = scenario_name.removeprefix("oct_")

    ds = meta.get("dataset", {})
    start = ds.get("start", 0)
    count = ds.get("count", 0)
    scenario_range = f"{scenario_name} [{start}-{start + count}]" if scenario_name else ""

    return {
        "slug": name,
        "name": name,
        "group": group,
        "note": note,
        "timestamp": meta["timestamp"],
        "git_commit": meta.get("git_commit"),
        "models_count": len(meta.get("models", {})),
        "constitution": constitution_name,
        "scenario": scenario_range,
        "sampler_mode": meta.get("collection", {}).get("sampler_mode"),
        "btd_model": meta.get("training", {}).get("model"),
        "dims": meta.get("training", {}).get("dims"),
        "top_model": top.get("model_name", ""),
        "top_elo": round(top.get("elo_mean", 0), 1),
        "test_loss": meta.get("log", {}).get("test_loss"),
    }


def upload_batch(batch_dir: Path, prefix: str, repo_id: str, token: str | None = None, note: str | None = None):
    """Upload all sub-runs in a directory as a single HF commit.

    Each sub-run is named as {prefix}/{subfolder} (e.g., matrix/goodness).
    """
    import tempfile
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Find all sub-dirs with spec.py
    sub_runs = sorted([d for d in batch_dir.iterdir() if d.is_dir() and (d / "spec.py").exists()])
    if not sub_runs:
        print(f"No runs found in {batch_dir}")
        sys.exit(1)

    print(f"Found {len(sub_runs)} runs in {batch_dir.name} (prefix: {prefix})")

    all_metas = []
    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir)

        for sub_dir in sub_runs:
            name = f"{prefix}/{sub_dir.name}"
            try:
                meta, summary_path = stage_run(name, sub_dir, staging)
                all_metas.append((name, meta, summary_path))
            except FileNotFoundError as e:
                print(f"  Skipping {name}: {e}")

        if not all_metas:
            print("No valid runs to upload")
            sys.exit(1)

        # Build index.json in staging
        try:
            from huggingface_hub import hf_hub_download
            index_path = hf_hub_download(repo_id=repo_id, filename="index.json", repo_type="dataset")
            with open(index_path) as f:
                index = json.load(f)
        except Exception:
            index = {"last_updated": None, "runs": []}

        for name, meta, summary_path in all_metas:
            entry = build_index_entry(name, meta, summary_path, group=prefix, note=note)
            index["runs"] = [r for r in index["runs"] if r["slug"] != name]
            index["runs"].append(entry)

        index["runs"].sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        index["last_updated"] = datetime.now(timezone.utc).isoformat()

        with open(staging / "index.json", "w") as f:
            json.dump(index, f, indent=2)

        # Upload all staged files via create_commit (single commit, no xet)
        from huggingface_hub import CommitOperationAdd
        print(f"Uploading {len(all_metas)} runs in single commit...")
        staged_files = sorted(f for f in staging.rglob("*") if f.is_file())
        operations = []
        for fpath in staged_files:
            rel = fpath.relative_to(staging)
            print(f"  Staging {rel}")
            operations.append(CommitOperationAdd(
                path_in_repo=str(rel),
                path_or_fileobj=fpath.read_bytes(),
            ))
        print(f"Committing {len(operations)} files...")
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add {len(all_metas)} runs from {batch_dir.name}",
        )

    print(f"Done! https://huggingface.co/datasets/{repo_id}/tree/main/runs")


def update_index(
    name: str, slug: str, meta: dict, summary_path: Path, repo_id: str, api: Any, note: str | None = None
):
    """Update the global index.json with this run's entry."""
    # Try to fetch existing index
    try:
        from huggingface_hub import hf_hub_download
        index_path = hf_hub_download(repo_id=repo_id, filename="index.json", repo_type="dataset")
        with open(index_path) as f:
            index = json.load(f)
    except Exception:
        index = {"last_updated": None, "runs": []}

    entry = build_index_entry(name, meta, summary_path, group=None, note=note)
    runs = [r for r in index["runs"] if r["slug"] != slug]
    runs.append(entry)
    # Sort by timestamp descending
    runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    index["runs"] = runs
    index["last_updated"] = datetime.now(timezone.utc).isoformat()

    print("Updating index.json")
    api.upload_file(
        path_or_fileobj=json.dumps(index, indent=2).encode(),
        path_in_repo="index.json",
        repo_id=repo_id,
        repo_type="dataset",
    )


def main():
    parser = argparse.ArgumentParser(description="Upload EigenBench results to HuggingFace")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", help="Path to a single run directory containing spec.py")
    group.add_argument("--batch-dir", help="Path to directory with multiple sub-run folders")
    parser.add_argument("--name", required=True,
                        help="Run name. For --run-dir: used as-is. For --batch-dir: used as prefix (e.g., 'matrix' -> 'matrix/goodness')")
    parser.add_argument("--repo", default="invi-bhagyesh/ValueArena",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--note", default=None, help="Note visible in the table (e.g., 'with API models')")
    parser.add_argument("--token", default=None, help="HF token (defaults to cached login)")
    args = parser.parse_args()

    if args.batch_dir:
        batch_dir = Path(args.batch_dir).resolve()
        if not batch_dir.exists():
            print(f"Error: {batch_dir} does not exist")
            sys.exit(1)
        upload_batch(batch_dir, args.name, args.repo, args.token, note=args.note)
    else:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            print(f"Error: {run_dir} does not exist")
            sys.exit(1)
        upload_run(args.name, run_dir, args.repo, args.token)


if __name__ == "__main__":
    main()
