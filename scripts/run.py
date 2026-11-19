"""Run collection with Inspect, then score or publish the judgments."""

from __future__ import annotations

import os
import pprint
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eigenbench.config import load_run_spec
from scripts.upload_results import resolve_upload_backend, submit_to_space, write_space_safe_spec


def estimate_calls(spec_ref: str) -> dict:
    """Use the collection planner without executing models or writing output."""

    spec, run_dir = load_run_spec(spec_ref)
    if spec.get("extension"):
        from inspect_pipeline.collection.extension import extend_run
        return extend_run(spec_ref, dry_run=True)
    if spec["evaluation"]["mode"] == "pairwise_btd":
        from inspect_pipeline.collection.pairwise import estimate_pairwise_calls
        return estimate_pairwise_calls(spec_ref)
    from inspect_pipeline.collection.direct_rating import estimate_direct_calls_for_spec
    return estimate_direct_calls_for_spec(spec, run_dir)


def main(spec_ref: str, collection_enabled: bool | None = None, *, resume_log: str | None = None):
    spec, run_dir = load_run_spec(spec_ref)
    collection_cfg = spec.get("collection", {})
    should_collect = bool(collection_cfg.get("enabled", True) if collection_enabled is None else collection_enabled)
    training_cfg = spec.get("training", {})
    upload_cfg = spec.get("upload", {})
    upload_enabled = bool(upload_cfg.get("enabled", False))
    upload_backend = resolve_upload_backend(upload_cfg)
    upload_to_space = upload_enabled and upload_backend == "valuearena_space"
    upload_to_dataset = upload_enabled and upload_backend == "huggingface_dataset"
    evaluation_mode = spec.get("evaluation", {}).get("mode", "pairwise_btd")
    if resume_log is not None and (
        evaluation_mode != "direct_rating" or spec.get("extension") or not should_collect
    ):
        raise ValueError("--resume-log requires an enabled direct-rating collection, not an extension")
    space_secret = upload_cfg.get("secret") or os.environ.get("SPACE_SECRET", "")
    space_spec_path = None
    if upload_to_space:
        if not space_secret:
            raise SystemExit("Set upload.secret in spec or SPACE_SECRET env var")
        # Validate and materialize this before any expensive collection calls.
        space_spec_path = write_space_safe_spec(spec)

    analysis_status: dict = {}
    scoring = bool(training_cfg.get("enabled", True)) and not upload_to_space
    if should_collect:
        print("Stage: collect evaluations (Inspect engine)")
        if spec.get("extension"):
            from inspect_pipeline.collection.extension import extend_run
            extend_run(spec_ref)
        elif evaluation_mode == "pairwise_btd":
            from inspect_pipeline.collection.pairwise import collect_pairwise_inspect
            collect_pairwise_inspect(spec_ref, scoring=scoring, analysis_status=analysis_status)
        else:
            from inspect_pipeline.collection.direct_rating import collect_direct_ratings_inspect
            collect_direct_ratings_inspect(
                spec_ref, scoring=scoring, analysis_status=analysis_status, resume_log=resume_log,
            )
    else:
        print("Stage: collect evaluations (skipped; collection.enabled=False)")

    if bool(training_cfg.get("enabled", True)) or upload_to_space:
        from inspect_pipeline.collection.extension import check_spec_covers
        from eigenbench.records import load_records

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
        submit_to_space(spec, space_spec_path, space_secret)
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
    parser.add_argument("--resume-log", metavar="EVAL_LOG",
                        help="Retry only unfinished hosted direct judgments from this Inspect log")
    parser.add_argument(
        "--estimate-calls",
        action="store_true",
        help="Print the planned logical generation/API counts without running collection",
    )
    args = parser.parse_args()
    if args.estimate_calls and args.resume_log:
        parser.error("--estimate-calls and --resume-log cannot be combined")
    collection_override = None
    if args.collection_enabled is not None:
        collection_override = args.collection_enabled.lower() == "true"
    if args.estimate_calls:
        print(pprint.pformat(estimate_calls(args.spec), sort_dicts=False))
    else:
        main(args.spec, collection_enabled=collection_override, resume_log=args.resume_log)


if __name__ == "__main__":
    cli()
