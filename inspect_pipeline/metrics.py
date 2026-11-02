"""Panel-level EigenBench metrics over the same records used by artifact analysis.

A score carries one exported judgment. The metric fits once over the complete
panel. Phased collection writes a scoring-only Inspect task over all judge logs.
"""
from __future__ import annotations

import warnings
from contextvars import ContextVar
from pathlib import Path

import numpy as np
from inspect_ai.hooks import Hooks, SampleEnd, TaskStart, hooks
from inspect_ai.scorer import Score, SampleScore, Target, metric, scorer
from inspect_ai.solver import TaskState, solver


_active_analysis: ContextVar[dict | None] = ContextVar("eigenbench_analysis", default=None)
_active_panel: ContextVar[dict | None] = ContextVar("eigenbench_panel", default=None)


@hooks("eigenbench_panel_config", "Capture resolved epoch coverage for EigenBench panel metrics")
class PanelConfig(Hooks):
    async def on_task_start(self, data: TaskStart) -> None:
        config = ((data.spec.metadata or {}).get("eigenbench") or {}).get("analysis")
        if config is not None:
            epochs = data.spec.config.epochs or 1
            config = {**config, "requested_epochs": epochs,
                      "planned": config["planned"] * epochs if config.get("planned") is not None else None,
                      "scoring_samples": config["scoring_samples"] * epochs}
        _active_analysis.set(config)
        _active_panel.set({"failed": set(), "fatal": set(), "expected": config.get("scoring_samples"),
                           "fingerprint": None, "metrics": None} if config else None)

    async def on_sample_end(self, data: SampleEnd) -> None:
        panel = _active_panel.get()
        if panel is not None and data.sample.error is not None:
            from .export import is_invalid_judgment
            key = (str(data.sample.id), data.sample.epoch)
            panel["failed"].add(key)
            config = _active_analysis.get()
            if (config["collection"].get("failure_policy", "strict") != "omit_invalid_judgments"
                    or not is_invalid_judgment(data.sample.error)):
                panel["fatal"].add(key)


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
        from pipeline.trust.direct_rating import build_direct_trust
        from pipeline.train.direct_analysis import compute_direct_bootstrap
        from pipeline.train.train import eigentrust_to_elo
        cfg = evaluation.get("direct_rating", {})
        collection = config["collection"]
        allow_sparse = (str(collection.get("sampler_mode", "all_to_all")).strip().lower() != "all_to_all"
                        or collection.get("failure_policy") == "omit_invalid_judgments")
        write_artifacts = config.get("write_artifacts", False) and bool(training.get("enabled", True))
        if write_artifacts:
            output_root = training.get("output_dir") or Path(collection["evaluations_path"]).parent / "train"
            from pipeline.train.direct_analysis import run_direct_analysis
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
                include_self=bool(cfg.get("include_self", True)), allow_sparse=allow_sparse,
                normalization=cfg.get("normalization", "zscore_softmax"),
                softmax_temperature=float(cfg.get("softmax_temperature", 1.0)),
                scale_min=float(cfg.get("scale_min", 1)), scale_max=float(cfg.get("scale_max", 10)),
                eigentrust_alpha=float(cfg.get("eigentrust_alpha", 0.0)),
            )
            bootstrap_cfg = training.get("bootstrap") or {}
            bootstrap = compute_direct_bootstrap(
                records=records, model_labels=labels, num_criteria=len(config["criteria"]),
                direct_cfg=cfg, bootstrap_cfg=bootstrap_cfg, allow_sparse=allow_sparse,
            ) if bootstrap_cfg.get("enabled", False) else None
        values.update(_ranking_metrics(
            result.eigentrust_scores, eigentrust_to_elo(result.eigentrust_scores, len(labels)),
            labels, bootstrap=bootstrap,
        ))
    else:
        from pipeline.train.pairwise_analysis import run_pairwise_analysis
        output_root = None
        if config.get("write_artifacts", False) and training.get("enabled", True):
            output_root = training.get("output_dir") or Path(config["collection"]["evaluations_path"]).parent / "train"
        result = run_pairwise_analysis(data=records, models=dict.fromkeys(labels),
                                       num_criteria=len(config["criteria"]), train_cfg=training,
                                       output_root=output_root, model_order=labels)
        values.update(result["diagnostics"])
        if result["unavailable_reason"]:
            values["ranking_unavailable_cleaning"] = 1.0
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


@metric(scores="unreduced")
def eigenbench_rankings():
    """EigenTrust, Elo, rank, optional Elo bootstrap intervals, and judgment coverage."""
    def calculate(scores: list[SampleScore]) -> dict:
        config = scores[0].score.metadata["analysis"] if scores else _active_analysis.get()
        if config is None:
            return {"ranking_available": 0.0, "coverage/scored": 0.0,
                    "coverage/planned": float("nan"), "coverage/scored_fraction": float("nan")}
        records = [score.score.metadata["record"] for score in scores]
        if config.get("requested_epochs", 1) == 1 and any(
            score.score.metadata.get("epoch", 1) != 1 for score in scores
        ):
            config = {**config, "requested_epochs": 0, "planned": None}
        panel = _active_panel.get()
        expected = panel["expected"] if panel is not None else None
        if panel is not None and expected is not None:
            scored = {(str(score.sample_id), score.score.metadata.get("epoch", 1)) for score in scores}
            # Retry can reuse successful scores without SampleEnd events. Count
            # those scores plus terminal unscored errors, each sample only once.
            accounted = len(records) + len(panel["failed"] - scored)
            if accounted < expected:
                values = _coverage_metrics(len(records), config)
                values["ranking_incomplete"] = 1.0
                return values
        if panel is not None:
            config = {**config, "omissions_validated": True,
                      "fatal_sample_errors": len(panel["fatal"])}
        # Inspect invokes metrics during progress and again at completion. Keep
        # one completed-panel fit and bootstrap draw stream for this task only.
        from .export import records_fingerprint
        fingerprint = records_fingerprint([{"analysis": config}, *records])
        if panel is not None and panel["fingerprint"] == fingerprint:
            return dict(panel["metrics"])
        values = compute_panel_metrics(records, config)
        if panel is not None:
            panel.update(fingerprint=fingerprint, metrics=values)
        return values
    return calculate


@scorer(metrics=[eigenbench_rankings()])
def panel_scorer(config: dict):
    """Carry the shared export record into panel-level native metrics."""
    async def score(state: TaskState, target: Target) -> Score:
        resolved = _active_analysis.get()
        if resolved is None:
            # inspect score rebuilds TaskState without TaskStart and also visits
            # errored samples. Persist resolved coverage, never rewrite artifacts.
            context = state.metadata.get("eigenbench_panel_context", {})
            resolved = {**config, **context, "write_artifacts": False}
            if not context and state.epoch != 1:
                resolved = {**resolved, "requested_epochs": 0, "planned": None}
        state.metadata["eigenbench_panel_context"] = {
            "requested_epochs": resolved.get("requested_epochs", 1),
            "planned": resolved.get("planned"),
            "omissions_validated": resolved.get("omissions_validated", False),
        }
        metadata = {"analysis": resolved, "epoch": state.epoch}
        try:
            if "panel_record" in state.metadata:
                record = state.metadata["panel_record"]
            elif config["evaluation"].get("mode") == "direct_rating":
                from .export import direct_record
                record = direct_record(state.metadata, state.store, config["criteria"])
            else:
                from .pairwise import pairwise_record
                record = pairwise_record(state.metadata, state.store, config["criteria"])
        except ValueError as exc:
            return Score(value=float("nan"), explanation=str(exc), metadata=metadata)
        return Score(value=1.0, metadata={**metadata, "record": record})
    return score


@solver
def _records_only():
    async def solve(state, generate):
        return state
    return solve


def finalize_panel_metrics(ctx: dict, *, logs, records: list[dict], planned, eval_kwargs: dict):
    """Score a phased run's full panel in a native Inspect log, with no model calls."""
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample
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
    result = inspect_eval(tasks=task, model=None, **eval_kwargs)
    if len(result) != 1 or result[0].status != "success":
        raise RuntimeError("Full-panel Inspect scoring failed; collection logs remain available")
    return result[0]


def record_analysis_status(log, status: dict | None) -> None:
    """Explicit fresh-run handoff; never infer artifact completion from files."""
    if status is None or log.status != "success" or log.results is None:
        return
    for result in log.results.scores:
        if result.name == "panel_scorer" and result.metrics.get("artifacts_written") is not None:
            status["artifacts_written"] = result.metrics["artifacts_written"].value == 1.0
