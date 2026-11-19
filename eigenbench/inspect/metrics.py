"""Build and score panels from exported Inspect judgment records."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def analysis_config(ctx: dict, *, planned: int | None, write_artifacts: bool = False) -> dict:
    spec = ctx["run_spec"]
    training = spec.get("training", {})
    mode = spec.get("evaluation", {}).get("mode", "pairwise_btd")
    return {
        "model_order": list(ctx["models"]),
        "criteria": ctx["criteria"],
        "evaluation": spec.get("evaluation", {}),
        "training": training,
        "write_artifacts": write_artifacts,
        "collection": {key: ctx["collection_cfg"][key] for key in (
            "sampler_mode", "failure_policy", "evaluations_path", "group_size",
            "response_redundancy", "sampler_seed",
        ) if key in ctx["collection_cfg"]},
        "planned": planned,
        "scoring_samples": planned,
        "target_role": "panel_member_as_judge_and_evaluee",
        "bootstrap_unit": "scenario" if mode == "direct_rating" else "comparison_row",
    }


def _ranking_metrics(trust, elo, labels, *, prefix="", bootstrap=None) -> dict:
    values = {}
    trust = np.asarray(trust)
    if not np.isfinite(trust).all() or not np.isfinite(elo).all():
        raise ValueError("EigenBench analysis produced non-finite ranking values (#41).")
    for index, label in enumerate(labels):
        key = f"{prefix}{label}"
        values.update({f"{key}/eigentrust": float(trust[index]),
                       f"{key}/elo": float(elo[index]),
                       f"{key}/rank": float(1 + np.sum(trust > trust[index]))})
    if bootstrap is not None:
        for row in bootstrap["summary"]:
            key = f"{prefix}{row['model_name']}"
            for field in ("elo_mean", "elo_std", "elo_ci_lower", "elo_ci_upper"):
                values[f"{key}/{field}"] = row[field]
    return values


def _coverage_metrics(count: int, config: dict) -> dict:
    planned = config.get("planned")
    return {"coverage/scored": count,
            "coverage/planned": planned if planned is not None else float("nan"),
            "coverage/scored_fraction": count / planned if planned else float("nan"),
            "ranking_available": 0.0}


def compute_panel_metrics(records: list[dict], config: dict) -> dict:
    """Compute native metrics and optionally write the same analysis artifacts.

    Coverage counts judgments, never criterion rows. Pairwise cleaning diagnostics
    are separate because legacy cleaning may discard repeated observations (#41).
    """
    labels = config["model_order"]
    # Names identify panel members; exported indices belong to this effective order.
    index = {name: i for i, name in enumerate(labels)}
    direct = config["evaluation"].get("mode") == "direct_rating"
    ordered = []
    for original in records:
        row = dict(original)
        if direct:
            for role in ("judge", "evaluee"):
                row[role] = {**row[role], "index": index[row[role]["name"]]}
        else:
            for role in ("judge", "eval1", "eval2"):
                row[role] = index[row[f"{role}_name"]]
        ordered.append(row)
    records = ordered
    planned = config.get("planned")
    values = _coverage_metrics(len(records), config)
    if config.get("fatal_sample_errors", 0):
        values["ranking_refused_errors"] = float(config["fatal_sample_errors"])
    if config.get("requested_epochs", 1) != 1:
        values["ranking_unsupported_epochs"] = 1.0
        warnings.warn("Panel rankings are unavailable for multiple Inspect epochs; collection "
                      "and requested-epoch coverage are preserved.", RuntimeWarning, stacklevel=2)
        return values
    if not records or config.get("fatal_sample_errors", 0):
        return values
    if planned is not None and len(records) < planned and config["collection"].get("failure_policy", "strict") == "strict":
        # Failed or unstarted judgments remain visible in the collection log. Do
        # not hide their error behind an incomplete-matrix error during metrics.
        values["ranking_incomplete"] = 1.0
        return values
    if planned is not None and len(records) < planned and not config.get("omissions_validated", False):
        values["ranking_unvalidated_omissions"] = 1.0
        warnings.warn("Panel rankings are unavailable because omitted judgments were not "
                      "validated against source sample errors.", RuntimeWarning, stacklevel=2)
        return values
    evaluation = config["evaluation"]
    training = config["training"]
    if direct:
        from eigenbench.analysis.direct_trust import build_direct_trust
        from eigenbench.analysis.direct_rating import compute_direct_bootstrap, direct_trust_options
        from eigenbench.analysis.fit import eigentrust_to_elo
        cfg = evaluation.get("direct_rating", {})
        collection = config["collection"]
        allow_sparse = (str(collection.get("sampler_mode", "all_to_all")).strip().lower() != "all_to_all"
                        or collection.get("failure_policy") == "omit_invalid_judgments")
        write_artifacts = config.get("write_artifacts", False) and bool(training.get("enabled", True))
        bootstrap_cfg = training.get("bootstrap") or {}
        if write_artifacts:
            output_root = training.get("output_dir") or Path(collection["evaluations_path"]).parent / "train"
            from eigenbench.analysis.direct_rating import run_direct_analysis
            analysis = run_direct_analysis(
                records=records, models=dict.fromkeys(labels), num_criteria=len(config["criteria"]),
                evaluation_cfg=evaluation, collection_cfg=collection, training_cfg=training,
                output_root=output_root,
            )
            result, bootstrap = analysis["result"], analysis["bootstrap"]
            values["artifacts_written"] = 1.0
        else:
            result = build_direct_trust(
                records, num_models=len(labels), num_criteria=len(config["criteria"]),
                allow_sparse=allow_sparse, **direct_trust_options(cfg),
            )
            bootstrap = compute_direct_bootstrap(
                records=records, model_labels=labels, num_criteria=len(config["criteria"]),
                direct_cfg=cfg, bootstrap_cfg=bootstrap_cfg, allow_sparse=allow_sparse,
            ) if bootstrap_cfg.get("enabled", False) else None
        if bootstrap_cfg.get("enabled", False) and bootstrap is None:
            values["bootstrap_unavailable_nonunique_eigentrust"] = 1.0
        values.update(_ranking_metrics(
            result.eigentrust_scores, eigentrust_to_elo(result.eigentrust_scores, len(labels), direct_rating=True),
            labels, bootstrap=bootstrap,
        ))
    else:
        from eigenbench.analysis.pairwise import run_pairwise_analysis
        output_root = None
        if config.get("write_artifacts", False) and training.get("enabled", True):
            output_root = training.get("output_dir") or Path(config["collection"]["evaluations_path"]).parent / "train"
        result = run_pairwise_analysis(data=records, models=dict.fromkeys(labels),
                                       num_criteria=len(config["criteria"]), train_cfg=training,
                                       output_root=output_root, model_order=labels)
        values.update(result["diagnostics"])
        if result["unavailable_reason"]:
            unavailable = ("model_coverage" if result["diagnostics"].get("panel_models_unobserved")
                           else "cleaning")
            values[f"ranking_unavailable_{unavailable}"] = 1.0
            warnings.warn(result["unavailable_reason"], RuntimeWarning, stacklevel=2)
            return values
        if result["diagnostics"]["comparisons_dropped"]:
            warnings.warn("Legacy pairwise cleaning dropped comparison rows (#41); see native "
                          "comparisons_parsed/retained/dropped metrics.", RuntimeWarning, stacklevel=2)
        if output_root is not None:
            values["artifacts_written"] = 1.0
        for dim, fitted in result["dimensions"].items():
            values.update(_ranking_metrics(fitted["eigentrust"], fitted["elo"], labels,
                                          prefix=f"btd_d{dim}/", bootstrap=fitted.get("bootstrap")))
    values["ranking_available"] = 1.0
    return values


def finalize_panel_metrics(ctx: dict, *, logs, records: list[dict], planned, eval_kwargs: dict):
    """Score a phased run's full panel in a native Inspect log, with no model calls."""
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample

    from inspect_pipeline.metrics import _records_only, panel_scorer

    from .coverage import collection_part
    coverage = collection_part("collection", planned=planned, logs=logs, records=records)
    config = analysis_config(ctx, planned=coverage["planned"],
                             write_artifacts=bool(ctx["run_spec"].get("training", {}).get("enabled", True)))
    config["scoring_samples"] = len(records)
    config["omissions_validated"] = True
    task = Task(
        name="eigenbench_panel", display_name="EigenBench full panel",
        dataset=MemoryDataset([Sample(input="Panel judgment", id=i + 1,
                                      metadata={"panel_record": record})
                               for i, record in enumerate(records)]),
        solver=_records_only(), scorer=panel_scorer(config),
        metadata={"eigenbench_phase": "panel_metrics", "eigenbench": {
            **ctx["meta"], "analysis": config,
            "source_logs": [Path(str(log.location)).name for log in logs],
            "coverage": {key: coverage[key] for key in
                         ("planned", "logged", "exported", "failed", "omitted", "unstarted")},
        }},
    )
    try:
        result = inspect_eval(tasks=task, model=None, **eval_kwargs)
    except Exception as exc:
        result, error = [], exc
    else:
        error = result[0].error.message if len(result) == 1 and result[0].error else None
    if len(result) != 1 or result[0].status != "success":
        # The export is already complete; analysis then runs separately after collection.
        warnings.warn(f"Full-panel Inspect scoring failed; collection is unaffected: {error}",
                      RuntimeWarning, stacklevel=2)
        return None
    return result[0]


def record_analysis_status(log, status: dict | None) -> None:
    """Explicit fresh-run handoff; never infer artifact completion from files."""
    if status is None or log is None or log.status != "success" or log.results is None:
        return
    for result in log.results.scores:
        if result.name == "panel_scorer" and result.metrics.get("artifacts_written") is not None:
            status["artifacts_written"] = result.metrics["artifacts_written"].value == 1.0
