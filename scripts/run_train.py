"""Internal training stage for pipeline orchestration.

This module is intended to be invoked by ``scripts/run.py``.
"""

from __future__ import annotations

import os

from pipeline.config import load_run_spec
from pipeline.utils import load_records


def _resolve_output_root(evaluations_path: str, train_cfg: dict) -> str:
    configured = train_cfg.get("output_dir")
    if configured:
        return configured
    evaluations_dir = os.path.dirname(evaluations_path) or "."
    return os.path.join(evaluations_dir, "train")


def main(spec_ref: str):
    _spec, run_dir = load_run_spec(spec_ref)
    verbose = bool(_spec.get("verbose", False))

    train_cfg = _spec["training"]
    if not train_cfg.get("enabled", True):
        if verbose:
            print("Training disabled in run spec.")
        return

    collection_cfg = _spec["collection"]
    evaluations_path = collection_cfg.get("evaluations_path")
    if not evaluations_path:
        raise SystemExit("Set collection.evaluations_path in your run spec.")

    if verbose:
        print(f"Run: {_spec['name']}")
        print(f"Run folder: {run_dir}")
        print(f"Evaluations file: {evaluations_path}")
    constitution_cfg = _spec.get("constitution", {})
    if "num_criteria" not in constitution_cfg:
        raise SystemExit(
            "Set constitution.num_criteria in your run spec. "
            "This controls criterion extraction/truncation."
        )
    num_criteria = int(constitution_cfg["num_criteria"])
    if num_criteria <= 0:
        raise SystemExit("constitution.num_criteria must be a positive integer.")

    data = load_records(evaluations_path)

    evaluation_cfg = _spec.get("evaluation", {})
    if evaluation_cfg.get("mode", "pairwise_btd") == "direct_rating":
        from pipeline.train.direct_analysis import run_direct_analysis

        print("Direct-rating mode: skipping BT/BTD fitting and aggregating ratings directly.")
        out_root = _resolve_output_root(evaluations_path, train_cfg)
        run_direct_analysis(
            records=data,
            models=_spec.get("models", {}),
            num_criteria=num_criteria,
            evaluation_cfg=evaluation_cfg,
            collection_cfg=collection_cfg,
            training_cfg=train_cfg,
            output_root=out_root,
            verbose=verbose,
        )
        return

    from pipeline.train.pairwise_analysis import run_pairwise_analysis

    result = run_pairwise_analysis(
        data=data, models=_spec.get("models", {}), num_criteria=num_criteria,
        train_cfg=train_cfg, output_root=_resolve_output_root(evaluations_path, train_cfg),
        verbose=verbose,
    )
    if result["unavailable_reason"]:
        raise RuntimeError(result["unavailable_reason"])


if __name__ == "__main__":
    raise SystemExit(
        "run_train.py is an internal stage. "
        "Use: python scripts/run.py <spec_module_or_path>"
    )
