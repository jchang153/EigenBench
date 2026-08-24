"""
Upload EigenBench run results to HuggingFace dataset repo for ValueArena.

Usage:
    python scripts/upload_results.py --name "my-run" --run-dir runs/my_run/
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import re
import shutil
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


def evaluation_mode(spec: dict) -> str:
    """Return the normalized evaluation protocol for a run specification."""

    value = str(spec.get("evaluation", {}).get("mode", "pairwise_btd")).strip().lower()
    aliases = {"pairwise": "pairwise_btd", "btd": "pairwise_btd", "direct": "direct_rating"}
    mode = aliases.get(value, value)
    if mode not in {"pairwise_btd", "direct_rating"}:
        raise ValueError(f"Unsupported evaluation mode: {value!r}")
    return mode


def detect_model_type(model_id: object) -> dict:
    """Normalize legacy strings and structured local model references for metadata."""

    if isinstance(model_id, Mapping):
        provider = model_id.get("provider")
        if provider != "hf_local":
            return {
                "id": str(model_id.get("id") or model_id.get("repo_id") or provider or ""),
                "type": str(model_id.get("kind") or "api"),
                "base_model": model_id.get("base_model_id"),
                "adapter": None,
            }
        repo_id = model_id.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError("Structured hf_local model reference must include repo_id")
        kind = str(model_id.get("kind") or "base")
        subfolder = model_id.get("subfolder")
        adapter = repo_id if kind == "lora" else None
        if isinstance(subfolder, str) and subfolder:
            adapter = f"{repo_id}/{subfolder}"
        return {
            "id": f"hf_local:{adapter or repo_id}",
            "type": kind,
            "base_model": model_id.get("base_model_id") or (repo_id if kind == "base" else None),
            "adapter": adapter,
            "revision": model_id.get("revision"),
            "base_revision": model_id.get("base_revision"),
        }

    if not isinstance(model_id, str):
        raise ValueError(f"Unsupported model reference: {model_id!r}")
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


def find_direct_dir(run_dir: Path, spec: dict) -> Path | None:
    """Find the direct-rating analysis directory for a run."""

    candidates = [run_dir / "direct_rating"]
    configured = spec.get("training", {}).get("output_dir")
    if configured:
        root = Path(str(configured)).expanduser()
        if not root.is_absolute():
            root = run_dir / root
        candidates.append(root / "direct_rating")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _point_summary(rows: list[dict], eigentrust: list[float], model_names: list[str]) -> list[dict]:
    """Convert direct point-estimate rows to the public summary schema."""

    if rows and all("elo_mean" in row for row in rows):
        return rows
    if rows and all("eigenbench_elo" in row for row in rows):
        converted = []
        for row in rows:
            elo = float(row["eigenbench_elo"])
            converted.append(
                {
                    "model_index": int(row["model_index"]),
                    "model_name": str(row["model_name"]),
                    "elo_mean": elo,
                    "elo_std": 0.0,
                    "elo_ci_lower": elo,
                    "elo_ci_upper": elo,
                }
            )
        converted.sort(key=lambda row: row["elo_mean"], reverse=True)
        return converted
    return build_summary_from_eigentrust(eigentrust, model_names)


def build_meta(
    name: str,
    spec: dict,
    log: dict,
    eigentrust: list[float],
    git_commit: str | None,
    git_repo: str | None,
    *,
    analysis: dict | None = None,
    artifacts: dict | None = None,
) -> dict:
    """Build the meta.json dict from parsed components."""
    mode = evaluation_mode(spec)
    models = {}
    for model_name, model_id in spec.get("models", {}).items():
        models[model_name] = detect_model_type(model_id)

    evaluation = dict(spec.get("evaluation", {}))
    evaluation["mode"] = mode
    bootstrap = dict(spec.get("training", {}).get("bootstrap", {}))
    bootstrap.setdefault("unit", "scenario" if mode == "direct_rating" else "judgment")

    meta = {
        "schema_version": 2,
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_repo": git_repo,
        "evaluation_mode": mode,
        "evaluation": evaluation,
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
        "bootstrap": bootstrap,
        "analysis": analysis or {},
        "artifacts": artifacts or {"images": [], "data": []},
        "log": log,
        "eigentrust": eigentrust,
    }

    if mode == "direct_rating":
        meta.pop("training", None)

    return meta


def stage_run(name: str, run_dir: Path, staging_dir: Path) -> tuple[dict, list[dict]]:
    """Stage a single run's files into a local directory for upload.

    Returns ``(meta_dict, summary_rows)``.
    """
    spec_path = run_dir / "spec.py"
    if not spec_path.exists():
        raise FileNotFoundError(f"{spec_path} not found")

    print(f"  Parsing {run_dir.name}")
    spec = parse_spec(spec_path)
    mode = evaluation_mode(spec)
    if mode == "direct_rating":
        analysis_dir = find_direct_dir(run_dir, spec)
        if analysis_dir is None:
            raise FileNotFoundError(f"No direct_rating directory found in {run_dir}")
        analysis_config = _load_json(analysis_dir / "analysis_config.json", {})
        log = dict(analysis_config)
    else:
        analysis_dir = find_btd_dir(run_dir)
        if analysis_dir is None:
            raise FileNotFoundError(f"No btd_d* directory found in {run_dir}")
        log_path = analysis_dir / "log_train.txt"
        log = parse_log_train(log_path) if log_path.exists() else {}
        analysis_config = {
            "kind": "pairwise_btd_eigentrust",
            "bootstrap_unit": "judgment",
        }

    analysis_config.setdefault(
        "kind", "direct_eigentrust" if mode == "direct_rating" else "pairwise_btd_eigentrust"
    )
    analysis_config.setdefault("bootstrap_unit", "scenario" if mode == "direct_rating" else "judgment")
    et_path = analysis_dir / "eigentrust.txt"
    eigentrust = parse_eigentrust(et_path) if et_path.exists() else []

    # Stage files
    dest = staging_dir / "runs" / name
    dest.mkdir(parents=True, exist_ok=True)
    images_dest = dest / "images"
    images_dest.mkdir(exist_ok=True)
    data_dest = dest / "data"

    bootstrap_summary_path = analysis_dir / "bootstrap" / "summary.json"
    if bootstrap_summary_path.exists():
        summary_data = _load_json(bootstrap_summary_path, [])
    else:
        model_names = list(spec.get("models", {}).keys())
        point_rows = _load_json(analysis_dir / "summary.json", []) if mode == "direct_rating" else []
        summary_data = _point_summary(point_rows, eigentrust, model_names)
    (dest / "summary.json").write_text(json.dumps(summary_data, indent=2) + "\n", encoding="utf-8")

    image_files = {
        "eigenbench.png": analysis_dir / "eigenbench.png",
        "bootstrap_elo.png": analysis_dir / "bootstrap" / "bootstrap_elo.png",
    }
    if mode == "direct_rating":
        image_files["trust_matrix.png"] = analysis_dir / "trust_matrix.png"
    else:
        image_files.update(
            {
                "training_loss.png": analysis_dir / "training_loss.png",
                "uv_embeddings_pca.png": analysis_dir / "uv_embeddings_pca.png",
            }
        )
    staged_images = []
    for img_name, img_path in image_files.items():
        if img_path.exists():
            shutil.copy2(img_path, images_dest / img_name)
            staged_images.append(img_name)

    staged_data = []
    if mode == "direct_rating":
        direct_files = [
            "analysis_config.json",
            "raw_mean_scores.csv",
            "normalization_intermediate.csv",
            "trust_matrix.csv",
            "observation_counts.csv",
        ]
        for filename in direct_files:
            source = analysis_dir / filename
            if source.exists():
                data_dest.mkdir(exist_ok=True)
                shutil.copy2(source, data_dest / filename)
                staged_data.append(f"data/{filename}")
        criteria_source = analysis_dir / "criteria"
        if criteria_source.is_dir():
            criteria_dest = data_dest / "criteria"
            shutil.copytree(criteria_source, criteria_dest, dirs_exist_ok=True)
            staged_data.extend(
                f"data/criteria/{path.name}" for path in sorted(criteria_dest.glob("*.csv"))
            )
        samples_source = analysis_dir / "bootstrap" / "samples.json"
        if samples_source.exists():
            data_dest.mkdir(exist_ok=True)
            shutil.copy2(samples_source, data_dest / "bootstrap_samples.json")
            staged_data.append("data/bootstrap_samples.json")

    # Evaluations
    eval_path = run_dir / "evaluations.jsonl"
    if eval_path.exists():
        shutil.copy2(eval_path, dest / "evaluations.jsonl")

    git_commit, git_repo = get_git_info(run_dir)
    meta = build_meta(
        name,
        spec,
        log,
        eigentrust,
        git_commit,
        git_repo,
        analysis=analysis_config,
        artifacts={"images": staged_images, "data": staged_data},
    )
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta, summary_data


def upload_run(
    name: str,
    run_dir: Path,
    repo_id: str,
    token: str | None = None,
    *,
    group: str | None = None,
    note: str | None = None,
):
    """Upload a single run's results to HuggingFace."""
    import tempfile
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir)
        meta, summary = stage_run(name, run_dir, staging)

        print(f"Uploading {name}...")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add run: {name}",
        )

    # Update index
    update_index(name, name, meta, summary, repo_id, api, group=group, note=note)
    print(f"Done! https://huggingface.co/datasets/{repo_id}/tree/main/runs/{name}")


def build_index_entry(name: str, meta: dict, summary: list[dict], group: str | None = None, note: str | None = None) -> dict:
    """Build a single index.json entry with all spec details."""
    top = max(summary, key=lambda row: row.get("elo_mean", float("-inf"))) if summary else {}
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

    mode = str(meta.get("evaluation_mode") or "pairwise_btd")
    direct_cfg = meta.get("evaluation", {}).get("direct_rating", {})
    entry = {
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
        "evaluation_mode": mode,
        "normalization": direct_cfg.get("normalization") if mode == "direct_rating" else None,
        "bootstrap_unit": meta.get("bootstrap", {}).get(
            "unit", "scenario" if mode == "direct_rating" else "judgment"
        ),
        "btd_model": meta.get("training", {}).get("model"),
        "dims": meta.get("training", {}).get("dims"),
        "top_model": top.get("model_name", ""),
        "top_elo": round(top.get("elo_mean", 0), 1),
        "test_loss": meta.get("log", {}).get("test_loss"),
    }
    return entry


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
                meta, summary = stage_run(name, sub_dir, staging)
                all_metas.append((name, meta, summary))
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

        for name, meta, summary in all_metas:
            entry = build_index_entry(name, meta, summary, group=prefix, note=note)
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
    name: str,
    slug: str,
    meta: dict,
    summary: list[dict],
    repo_id: str,
    api: Any,
    group: str | None = None,
    note: str | None = None,
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

    entry = build_index_entry(name, meta, summary, group=group, note=note)
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
    parser.add_argument("--group", default=None, help="Optional website grouping for a single run")
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
        upload_run(args.name, run_dir, args.repo, args.token, group=args.group, note=args.note)


if __name__ == "__main__":
    main()
