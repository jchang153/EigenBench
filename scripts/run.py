"""Run collection on Inspect, then shared analysis and upload from a Python run spec.

Usage:
    python scripts/run.py runs.example.spec
    python scripts/run.py runs/example/spec.py
"""

from __future__ import annotations

import copy
import os
import pprint
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

# Allow "python scripts/run.py ..." to import top-level packages (e.g. pipeline).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.config import load_run_spec


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


def _write_space_safe_spec(spec: dict) -> str:
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


def estimate_calls(spec_ref: str) -> dict:
    """Use the collection planner without executing models or writing output."""

    spec, run_dir = load_run_spec(spec_ref)
    if spec.get("extension"):
        from inspect_pipeline.extend import extend_run
        return extend_run(spec_ref, dry_run=True)
    if spec["evaluation"]["mode"] == "pairwise_btd":
        from inspect_pipeline.pairwise_collect import estimate_pairwise_calls
        return estimate_pairwise_calls(spec_ref)
    from inspect_pipeline.collect import estimate_direct_calls_for_spec
    return estimate_direct_calls_for_spec(spec, run_dir)


def _resolve_upload_backend(upload_cfg: dict) -> str:
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


def main(spec_ref: str, collection_enabled: bool | None = None):
    spec, run_dir = load_run_spec(spec_ref)
    collection_cfg = spec.get("collection", {})
    training_cfg = spec.get("training", {})
    upload_cfg = spec.get("upload", {})
    upload_enabled = bool(upload_cfg.get("enabled", False))
    upload_backend = _resolve_upload_backend(upload_cfg)
    upload_to_space = upload_enabled and upload_backend == "valuearena_space"
    upload_to_dataset = upload_enabled and upload_backend == "huggingface_dataset"
    evaluation_mode = spec.get("evaluation", {}).get("mode", "pairwise_btd")
    space_secret = upload_cfg.get("secret") or os.environ.get("SPACE_SECRET", "")
    space_spec_path = None
    if upload_to_space:
        if not space_secret:
            raise SystemExit("Set upload.secret in spec or SPACE_SECRET env var")
        # Validate and materialize this before any expensive collection calls.
        space_spec_path = _write_space_safe_spec(spec)

    analysis_status: dict = {}
    scoring = bool(training_cfg.get("enabled", True)) and not upload_to_space
    if collection_enabled is not None:
        collection_cfg["enabled"] = collection_enabled
    if bool(collection_cfg.get("enabled", True)):
        print("Stage: collect evaluations (Inspect engine)")
        if spec.get("extension"):
            from inspect_pipeline.extend import extend_run
            extend_run(spec_ref)
        elif evaluation_mode == "pairwise_btd":
            from inspect_pipeline.pairwise_collect import collect_pairwise_inspect
            collect_pairwise_inspect(spec_ref, scoring=scoring, analysis_status=analysis_status)
        else:
            from inspect_pipeline.collect import collect_direct_ratings_inspect
            collect_direct_ratings_inspect(spec_ref, scoring=scoring, analysis_status=analysis_status)
    else:
        print("Stage: collect evaluations (skipped; collection.enabled=False)")

    if bool(training_cfg.get("enabled", True)) or upload_to_space:
        from inspect_pipeline.extend import check_spec_covers
        from pipeline.utils import load_records

        records = load_records(collection_cfg["evaluations_path"])
        if records:
            check_spec_covers(spec.get("models", {}), records)

    if upload_to_space:
        # Skip local training — Space handles it
        print("Stage: train + eigentrust (skipped; upload.enabled=True, Space will train)")
    elif scoring and analysis_status.get("artifacts_written") is True:
        print("Stage: analysis complete in Inspect; using the artifacts from its panel score")
    elif scoring:
        if evaluation_mode == "direct_rating":
            print("Stage: direct aggregation + eigentrust")
        else:
            print("Stage: train + eigentrust")
        from scripts.run_train import main as run_train_main

        run_train_main(spec_ref)
    else:
        print("Stage: train + eigentrust (skipped; training.enabled=False)")

    if upload_to_space:
        print("Stage: submitting to ValueArena Space")
        import subprocess

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
    elif upload_to_dataset:
        print("Stage: uploading analyzed run to Hugging Face dataset")
        from scripts.upload_results import upload_run

        dataset_repo = str(upload_cfg.get("repo", "invi-bhagyesh/ValueArena"))
        upload_name = str(upload_cfg.get("name") or spec.get("name") or run_dir.name)
        upload_run(
            upload_name,
            Path(run_dir),
            dataset_repo,
            upload_cfg.get("token") or os.environ.get("HF_TOKEN"),
            group=upload_cfg.get("group"),
            note=upload_cfg.get("note"),
        )


def cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Collect with Inspect, then analyze and upload")
    parser.add_argument("spec", help="Path to run spec")
    parser.add_argument("--collection-enabled", type=str.lower, choices=["true", "false"], default=None,
                        help="Override collection.enabled (true/false)")
    parser.add_argument(
        "--estimate-calls",
        action="store_true",
        help="Print the planned logical generation/API counts without running collection",
    )
    args = parser.parse_args()
    collection_override = None
    if args.collection_enabled is not None:
        collection_override = args.collection_enabled.lower() == "true"
    if args.estimate_calls:
        print(pprint.pformat(estimate_calls(args.spec), sort_dicts=False))
    else:
        main(args.spec, collection_enabled=collection_override)


if __name__ == "__main__":
    cli()
