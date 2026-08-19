"""Run collection then training from a Python run spec.

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

from pipeline.config import (
    load_dataset_scenarios_from_spec,
    load_run_spec,
    select_scenarios,
)
from pipeline.model_refs import is_hf_local_model


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
    spec, run_dir = load_run_spec(spec_ref)
    models = spec.get("models", {})
    dataset_cfg = spec.get("dataset", {})
    scenarios = load_dataset_scenarios_from_spec(dataset_cfg, run_dir=run_dir)
    count_value = dataset_cfg.get("count")
    selected = select_scenarios(
        scenarios,
        start=int(dataset_cfg.get("start", 0)),
        count=None if count_value is None else int(count_value),
        shuffle=bool(dataset_cfg.get("shuffle", False)),
        shuffle_seed=(
            None
            if dataset_cfg.get("shuffle_seed") is None
            else int(dataset_cfg.get("shuffle_seed"))
        ),
    )
    evaluation_cfg = spec.get("evaluation", {})
    mode = evaluation_cfg.get("mode", "pairwise_btd")
    collection_cfg = spec.get("collection", {})
    num_models = len(models)
    if mode == "direct_rating":
        from pipeline.eval.direct_rating import count_cached_responses, estimate_direct_calls

        openrouter_nicks = {
            nick for nick, value in models.items() if not is_hf_local_model(value)
        }
        cached_total, cached_remote = count_cached_responses(
            collection_cfg.get("cached_responses_path"),
            scenario_indices={int(item[0]) for item in selected},
            model_nicks=set(models),
            openrouter_nicks=openrouter_nicks,
        )

        return {
            "mode": mode,
            **estimate_direct_calls(
                num_scenarios=len(selected),
                num_models=num_models,
                num_openrouter_models=len(openrouter_nicks),
                include_self=bool(
                    evaluation_cfg.get("direct_rating", {}).get("include_self", True)
                ),
                cached_responses=cached_total,
                cached_openrouter_responses=cached_remote,
            ),
        }

    sampler_mode = str(collection_cfg.get("sampler_mode", "random_judge_group")).strip().lower()
    cached = bool(collection_cfg.get("cached_responses_path"))
    if sampler_mode == "all_to_all":
        responses = 0 if cached else len(selected) * num_models
        reflections = len(selected) * num_models * num_models
        comparisons = len(selected) * num_models * num_models * max(0, num_models - 1)
    else:
        group_size = max(1, min(int(collection_cfg.get("group_size", 4)), num_models))
        groups = max(1, int(collection_cfg.get("groups", 1)))
        responses = 0 if cached else len(selected) * groups * group_size
        reflections = len(selected) * groups * group_size
        comparisons = len(selected) * groups * group_size * max(0, group_size - 1)
    result = {
        "mode": mode,
        "sampler_mode": sampler_mode,
        "num_scenarios": len(selected),
        "num_models": num_models,
        "response_tasks": responses,
        "reflection_tasks": reflections,
        "comparison_tasks": comparisons,
        "total_logical_generations": responses + reflections + comparisons,
    }
    if sampler_mode == "all_to_all":
        result["ordered_comparisons_per_scenario"] = (
            num_models * num_models * max(0, num_models - 1)
        )
    else:
        result["group_size"] = group_size
        result["groups_per_scenario"] = groups
        result["count_note"] = (
            "Sampled mixed-provider runs may deduplicate overlapping response/reflection "
            "tasks, so those two fields are conservative before assignments are materialized."
        )
    return result


def main(spec_ref: str, collection_enabled: bool | None = None):
    spec, _ = load_run_spec(spec_ref)
    collection_cfg = spec.get("collection", {})
    training_cfg = spec.get("training", {})
    upload_cfg = spec.get("upload", {})
    upload_to_space = bool(upload_cfg.get("enabled", False))
    evaluation_mode = spec.get("evaluation", {}).get("mode", "pairwise_btd")
    if upload_to_space and evaluation_mode == "direct_rating":
        raise SystemExit(
            "upload.enabled=True is not supported for evaluation.mode='direct_rating': "
            "the current ValueArena Space expects pairwise BTD evaluations."
        )
    space_secret = upload_cfg.get("secret") or os.environ.get("SPACE_SECRET", "")
    space_spec_path = None
    if upload_to_space:
        if not space_secret:
            raise SystemExit("Set upload.secret in spec or SPACE_SECRET env var")
        # Validate and materialize this before any expensive collection calls.
        space_spec_path = _write_space_safe_spec(spec)

    if collection_enabled is not None:
        collection_cfg["enabled"] = collection_enabled
    cached_responses_path = collection_cfg.get("cached_responses_path")

    if cached_responses_path and evaluation_mode != "direct_rating":
        print("Stage: collect responses cache")
        from run_collect_responses import main as run_collect_responses_main

        run_collect_responses_main(spec_ref)
    elif evaluation_mode == "direct_rating" and cached_responses_path:
        print(
            "Stage: collect responses cache (integrated into direct-rating collection; "
            "existing cached responses will be reused)"
        )
    else:
        print("Stage: collect responses cache (skipped; collection.cached_responses_path is not set)")

    if bool(collection_cfg.get("enabled", True)):
        print("Stage: collect evaluations")
        from run_collect import main as run_collect_main

        run_collect_main(spec_ref)
    else:
        print("Stage: collect evaluations (skipped; collection.enabled=False)")

    if upload_to_space:
        # Skip local training — Space handles it
        print("Stage: train + eigentrust (skipped; upload.enabled=True, Space will train)")
    elif bool(training_cfg.get("enabled", True)):
        if evaluation_mode == "direct_rating":
            print("Stage: direct aggregation + eigentrust")
        else:
            print("Stage: train + eigentrust")
        from run_train import main as run_train_main

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

        # Write a standalone script and run it detached via nohup
        script_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="va_submit_")
        script_file.write(f"""
import os
# Remove SOCKS proxies (cause socksio import error) but keep HTTP/HTTPS proxy for DNS
for k in list(os.environ):
    kl = k.lower()
    if kl in ("all_proxy", "ftp_proxy", "grpc_proxy", "rsync_proxy"):
        os.environ.pop(k, None)
from gradio_client import Client, handle_file
c = Client("https://invi-bhagyesh-valuearena.hf.space/")
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
        print(f"Submitted! Job running on Space in background.")
        print(f"  Log: {log_file}")
        print(f"  Track: https://huggingface.co/spaces/invi-bhagyesh/ValueArena")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="Path to run spec")
    parser.add_argument("--collection-enabled", type=str, default=None,
                        help="Override collection.enabled (True/False)")
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
