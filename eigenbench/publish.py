"""Publish runs to ValueArena or a Hugging Face dataset."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pprint
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eigenbench.analysis.elo import elo_scores
from eigenbench.config.run_spec import _resolve_path_for_run
from eigenbench.records import load_records, save_records


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _space_safe_model_id(model_ref: object) -> str:
    """Convert a structured local model reference to Space metadata syntax.

    ValueArena trains only from the collected comparisons, so it does not load
    these models. Its current metadata parser nevertheless expects legacy
    string model IDs.
    """

    if isinstance(model_ref, str):
        return model_ref
    if not isinstance(model_ref, Mapping) or model_ref.get("provider") != "hf_local":
        raise ValueError(f"Unsupported model reference for ValueArena: {model_ref!r}")

    repo_id = model_ref.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("Structured local model reference must include repo_id")
    subfolder = model_ref.get("subfolder")
    if subfolder:
        return f"hf_local:{repo_id}/{subfolder}"
    if model_ref.get("kind") == "lora":
        # The Space uses the third path component only to classify metadata;
        # model artifacts are never loaded during the upload-stage training.
        return f"hf_local:{repo_id}/adapter"
    return f"hf_local:{repo_id}"


def write_space_safe_spec(spec: dict) -> str:
    """Materialize a standalone spec accepted by the current ValueArena Space."""

    space_spec = copy.deepcopy(spec)
    space_spec["models"] = {
        name: _space_safe_model_id(model_ref)
        for name, model_ref in spec.get("models", {}).items()
    }
    if isinstance(space_spec.get("upload"), dict):
        space_spec["upload"].pop("secret", None)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix="va_spec_",
    )
    handle.write("RUN_SPEC = ")
    pprint.pprint(space_spec, stream=handle, sort_dicts=False)
    handle.close()
    return handle.name


def resolve_upload_backend(upload_cfg: dict) -> str:
    value = str(upload_cfg.get("backend", "valuearena_space")).strip().lower()
    aliases = {
        "space": "valuearena_space",
        "valuearena": "valuearena_space",
        "hf": "huggingface_dataset",
        "hf_dataset": "huggingface_dataset",
        "huggingface": "huggingface_dataset",
    }
    backend = aliases.get(value, value)
    if backend not in {"valuearena_space", "huggingface_dataset"}:
        raise ValueError(
            "upload.backend must be 'valuearena_space' or 'huggingface_dataset'; "
            f"got {value!r}"
        )
    return backend


def submit_to_space(spec: dict, space_spec_path: str, space_secret: str) -> None:
    """Submit collected records for analysis and publication on ValueArena."""

    collection_cfg = spec.get("collection", {})
    upload_cfg = spec.get("upload", {})
    # Auto-capture git commit
    git_commit = upload_cfg.get("git_commit", "")
    if not git_commit:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
            ).strip()
        except Exception:
            git_commit = ""

    eval_path = collection_cfg.get("evaluations_path", "")
    spec_path = space_spec_path

    run_name = upload_cfg.get("name", spec.get("name", ""))
    run_group = upload_cfg.get("group", "")
    run_note = upload_cfg.get("note", "")

    try:
        eval_mb = os.path.getsize(eval_path) / 1e6
    except OSError:
        eval_mb = 0.0
    upload_timeout = float(
        upload_cfg.get("timeout_seconds", max(600.0, eval_mb * 20.0))
    )

    # Write a standalone script and run it detached via nohup
    script_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="va_submit_")
    script_file.write(f"""
import os
# Remove SOCKS proxies (cause socksio import error) but keep HTTP/HTTPS proxy for DNS
for k in list(os.environ):
    kl = k.lower()
    if kl in ("all_proxy", "ftp_proxy", "grpc_proxy", "rsync_proxy"):
        os.environ.pop(k, None)
import httpx
from gradio_client import Client, handle_file
c = Client(
    "https://invi-bhagyesh-valuearena.hf.space/",
    httpx_kwargs={{"timeout": httpx.Timeout({upload_timeout!r})}},
)
try:
    secret = os.environ["SPACE_SECRET"]
    result = c.predict(secret, handle_file({eval_path!r}), handle_file({spec_path!r}), {run_name!r}, {run_group!r}, {run_note!r}, {git_commit!r})
    print("Done!", result[0] if result else result)
except Exception as e:
    print("Error:", e)
finally:
    for path in (__file__, {spec_path!r}):
        try:
            os.unlink(path)
        except OSError:
            pass
""")
    script_file.close()

    log_file = script_file.name.replace(".py", ".log")
    submit_env = os.environ.copy()
    submit_env["SPACE_SECRET"] = space_secret
    subprocess.Popen(
        f"nohup {sys.executable} -u {script_file.name} > {log_file} 2>&1 &",
        shell=True,
        env=submit_env,
    )
    print(f"Handed off to background uploader ({eval_mb:.1f} MB, timeout {upload_timeout:.0f}s).")
    print(f"  Result (check this): {log_file}")
    print(f"  Track: https://huggingface.co/spaces/invi-bhagyesh/ValueArena")


def parse_spec(spec_path: Path) -> dict:
    """Parse a spec.py file and return the RUN_SPEC dict."""
    namespace = {
        "__file__": str(spec_path),
        "min": min, "max": max, "bool": bool, "True": True, "False": False,
    }
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
    values = text.strip().removeprefix("EigenTrust scores:").strip()
    values = values.removeprefix("[").removesuffix("]")
    return [float(token) for token in values.replace(",", " ").split()]


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


def build_summary_from_eigentrust(
    et_scores: list[float], model_names: list[str], *, direct_rating: bool = False,
) -> list[dict]:
    """Build a point summary using the same Elo conversion as analysis."""
    elos = elo_scores(et_scores, len(model_names), direct_rating=direct_rating)
    rows = []
    for i, (name, elo) in enumerate(zip(model_names, elos, strict=True)):
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


def btd_dir_name(spec: dict) -> str:
    """The published BTD dimension: the first in training.dims, as analysis defaults it."""

    dims = list(spec.get("training", {}).get("dims", [2]))
    if not dims:
        raise ValueError("training.dims is empty; nothing to publish")
    return f"btd_d{int(dims[0])}"


def find_btd_dir(run_dir: Path, spec: dict) -> Path | None:
    """Find the configured BTD dimension's output directory, never another dimension's.

    A configured training.output_dir is the only place looked, so an older folder in the
    run directory cannot stand in for missing results.
    """

    configured = spec.get("training", {}).get("output_dir")
    candidate = Path(_resolve_path_for_run(configured, run_dir, ".")) / btd_dir_name(spec)
    return candidate if candidate.is_dir() else None


def find_direct_dir(run_dir: Path, spec: dict) -> Path | None:
    """Find the direct-rating output in the configured training directory."""

    configured = spec.get("training", {}).get("output_dir")
    candidate = Path(_resolve_path_for_run(configured, run_dir, ".")) / "direct_rating"
    return candidate if candidate.is_dir() else None


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _point_summary(rows: list[dict], trust_summary: list[dict]) -> list[dict]:
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
    return trust_summary


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
    inspect_info: dict | None = None,
) -> dict:
    """Build the meta.json dict from parsed components."""
    mode = evaluation_mode(spec)
    models = {}
    for model_name, model_id in spec.get("models", {}).items():
        models[model_name] = detect_model_type(model_id)

    evaluation = dict(spec.get("evaluation", {}))
    evaluation["mode"] = mode
    bootstrap = dict(spec.get("training", {}).get("bootstrap") or {})
    bootstrap["enabled"] = bool(bootstrap.get("enabled", False))
    if analysis and "bootstrap_available" in analysis:
        bootstrap["available"] = bootstrap["enabled"] and analysis["bootstrap_available"]
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
    # Only Inspect-collected runs have a viewer to link to.
    if inspect_info:
        meta["inspect"] = inspect_info
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
    collection = spec.get("collection", {})
    configured_evaluations = collection.get("evaluations_path")
    eval_path = Path(_resolve_path_for_run(configured_evaluations, run_dir, "evaluations.jsonl"))
    if configured_evaluations and not eval_path.is_file():
        raise FileNotFoundError(f"Configured evaluations file not found: {eval_path}")

    training = spec.get("training", {})
    bootstrap_enabled = bool((training.get("bootstrap") or {}).get("enabled", False))
    mode = evaluation_mode(spec)
    if mode == "direct_rating":
        analysis_dir = find_direct_dir(run_dir, spec)
        if analysis_dir is None:
            raise FileNotFoundError(f"No direct_rating directory found in {run_dir}")
        analysis_config = _load_json(analysis_dir / "analysis_config.json", {})
        log = dict(analysis_config)
    else:
        analysis_dir = find_btd_dir(run_dir, spec)
        if analysis_dir is None:
            raise FileNotFoundError(f"No {btd_dir_name(spec)} directory found in {run_dir}")
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
    eigentrust = []
    trust_summary = []
    if et_path.exists():
        eigentrust = parse_eigentrust(et_path)
        trust_summary = build_summary_from_eigentrust(
            eigentrust, list(spec["models"]), direct_rating=mode == "direct_rating",
        )

    # Stage files
    dest = staging_dir / "runs" / name
    dest.mkdir(parents=True, exist_ok=True)
    images_dest = dest / "images"
    images_dest.mkdir(exist_ok=True)
    data_dest = dest / "data"

    bootstrap_unavailable = bootstrap_enabled and analysis_config.get("bootstrap_available") is False
    stage_bootstrap = bootstrap_enabled and not bootstrap_unavailable
    bootstrap_dir = analysis_dir / "bootstrap"
    bootstrap_summary_path = bootstrap_dir / "summary.json"
    if stage_bootstrap and bootstrap_summary_path.exists():
        summary_data = _load_json(bootstrap_summary_path, [])
    else:
        point_rows = _load_json(analysis_dir / "summary.json", []) if mode == "direct_rating" else []
        summary_data = _point_summary(point_rows, trust_summary)
        if bootstrap_unavailable:
            for row in summary_data:
                for field in ("elo_std", "elo_ci_lower", "elo_ci_upper"):
                    row.pop(field, None)
    (dest / "summary.json").write_text(json.dumps(summary_data, indent=2) + "\n", encoding="utf-8")

    image_files = {"eigenbench.png": analysis_dir / "eigenbench.png"}
    if stage_bootstrap:
        image_files["bootstrap_elo.png"] = bootstrap_dir / "bootstrap_elo.png"
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
        samples_source = bootstrap_dir / "samples.json"
        if stage_bootstrap and samples_source.exists():
            data_dest.mkdir(exist_ok=True)
            shutil.copy2(samples_source, data_dest / "bootstrap_samples.json")
            staged_data.append("data/bootstrap_samples.json")

    # Evaluations
    staged_eval_path = dest / "evaluations.jsonl"
    is_jsonl_export = eval_path.suffix.lower() == ".jsonl"
    if eval_path.is_file():
        if is_jsonl_export:
            shutil.copy2(eval_path, staged_eval_path)
        else:
            save_records(staged_eval_path, load_records(eval_path))

    git_commit, git_repo = get_git_info(run_dir)
    # The Inspect logs travel with the run, so the viewer can read them straight
    # from the dataset by URL instead of needing a bundle published elsewhere.
    inspect_info = None
    collection_dir = eval_path.parent
    inspect_info_path = collection_dir / "inspect_run.json"
    loaded = json.loads(inspect_info_path.read_text(encoding="utf-8")) if inspect_info_path.exists() else {}
    names = loaded.get("log_files") or ([loaded["log_file"]] if loaded.get("log_file") else [])
    inspect_cfg = collection.get("inspect") or {}
    log_dir = collection_dir / (inspect_cfg.get("log_dir") or "inspect_logs")
    present = [log_dir / name for name in names if (log_dir / name).is_file()]
    if present:
        report_name = "collection_report.json"
        report_path = collection_dir / report_name
        try:
            from eigenbench.inspect.coverage import evaluations_sha256, report_from_run_dir

            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
            staged_hash = evaluations_sha256(staged_eval_path)
            if staged_hash is None:
                raise ValueError("no staged evaluations to link to Inspect logs")
            source_hash = staged_hash if is_jsonl_export else evaluations_sha256(eval_path)
            reported_logs = {entry["file"] for part in (report or {}).get("collections", [])
                             for entry in part.get("logs", [])}
            if (not report or report.get("schema_version") != 2
                    or report.get("evaluations_sha256") != source_hash
                    or reported_logs != set(names)):
                # Missing or stale report. Rebuild from the current logs and export,
                # with the original plan unknown.
                report = report_from_run_dir(collection_dir, staged_eval_path, log_dir=log_dir,
                                             verify_records=True)
            report["evaluations_sha256"] = staged_hash
            (dest / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"Warning: Inspect evidence unavailable: {exc}", file=sys.stderr)
        else:
            for source in present:
                shutil.copy2(source, dest / source.name)
            published = [source.name for source in present]
            primary = loaded.get("log_file") if loaded.get("log_file") in published else published[0]
            inspect_info = {"log_file": primary, "log_files": published,
                            "collection_report_file": report_name}
    meta = build_meta(
        name,
        spec,
        log,
        eigentrust,
        git_commit,
        git_repo,
        analysis=analysis_config,
        artifacts={"images": staged_images, "data": staged_data},
        inspect_info=inspect_info,
    )
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta, summary_data


def _load_index(repo_id: str, *, revision: str, token: str | None) -> dict:
    """The dataset's catalogue. Only a missing index.json starts a new one.

    Any other failure (offline, authentication, timeout, unreadable JSON) aborts, so a
    failed read can never replace the catalogue with only the runs being uploaded. The
    read is always fresh: after a timeout the Hub client would otherwise return a stale
    cached copy, and writing it back would drop runs added since.
    """

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    try:
        index_path = hf_hub_download(repo_id=repo_id, filename="index.json", repo_type="dataset",
                                     revision=revision, token=token, force_download=True)
    except LocalEntryNotFoundError:
        raise  # the Hub was unreachable, not the file absent
    except EntryNotFoundError:
        return {"last_updated": None, "runs": []}
    with open(index_path) as f:
        index = json.load(f)
    if not isinstance(index, dict) or not isinstance(index.get("runs"), list):
        raise ValueError(f"index.json in {repo_id} is not a run catalogue; not overwriting it")
    return index


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
        uploaded = api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add run: {name}",
        )

    # Update index
    update_index(name, name, meta, summary, repo_id, api, group=group, note=note, revision=uploaded.oid)
    print(f"Done! https://huggingface.co/datasets/{repo_id}/tree/main/runs/{name}")


def build_index_entry(name: str, meta: dict, summary: list[dict], group: str | None = None, note: str | None = None) -> dict:
    """Build a single index.json entry with all spec details."""
    top = max(summary, key=lambda row: row.get("elo_mean", float("-inf"))) if summary else {}
    constitution_path = meta.get("constitution", {}).get("path", "")
    constitution_name = Path(constitution_path).stem if constitution_path else ""
    constitution_name = constitution_name.removeprefix("oct_")
    ds = meta.get("dataset", {})
    dataset_path = ds.get("path", "")
    scenario_name = Path(dataset_path).stem if dataset_path else str(ds.get("id") or "")
    scenario_name = scenario_name.removeprefix("oct_")
    start = int(ds.get("start", 0) or 0)
    count_value = ds.get("count")
    scenario_range = scenario_name
    if scenario_name and count_value is not None:
        count = int(count_value)
        if count > 0:
            scenario_range = f"{scenario_name} [{start}-{start + count - 1}]"

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


def _merge_index_entries(index: dict, replacements: dict[str, dict]) -> dict:
    runs = [run for run in index["runs"] if run["slug"] not in replacements]
    runs.extend(replacements.values())
    runs.sort(key=lambda run: run.get("timestamp", ""), reverse=True)
    return {**index, "runs": runs, "last_updated": datetime.now(timezone.utc).isoformat()}


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

        revision = api.dataset_info(repo_id).sha
        index = _load_index(repo_id, revision=revision, token=api.token)
        index = _merge_index_entries(index, {
            name: build_index_entry(name, meta, summary, group=prefix, note=note)
            for name, meta, summary in all_metas
        })

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
            parent_commit=revision,
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
    *,
    revision: str | None = None,
):
    """Update the global index.json with this run's entry."""
    revision = revision or api.dataset_info(repo_id).sha
    index = _load_index(repo_id, revision=revision, token=api.token)
    entry = build_index_entry(name, meta, summary, group=group, note=note)
    index = _merge_index_entries(index, {slug: entry})

    print("Updating index.json")
    api.upload_file(
        path_or_fileobj=json.dumps(index, indent=2).encode(),
        path_in_repo="index.json",
        parent_commit=revision,
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
