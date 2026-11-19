"""Inspect registrations for panel scoring and coverage.

Inspect derives installed registry names from this module's package. Keep
registrations here so existing evaluation logs can use those names.
"""

from __future__ import annotations

import warnings
from contextvars import ContextVar

from inspect_ai.hooks import Hooks, SampleEnd, TaskStart, hooks
from inspect_ai.scorer import Score, SampleScore, Target, metric, scorer
from inspect_ai.solver import TaskState, solver

from eigenbench.inspect.metrics import (
    _coverage_metrics,
    analysis_config as analysis_config,
    compute_panel_metrics as compute_panel_metrics,
    finalize_panel_metrics as finalize_panel_metrics,
    record_analysis_status as record_analysis_status,
)


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
            from eigenbench.inspect.export import is_invalid_judgment
            key = (str(data.sample.id), data.sample.epoch)
            panel["failed"].add(key)
            config = _active_analysis.get()
            if (config["collection"].get("failure_policy", "strict") != "omit_invalid_judgments"
                    or not is_invalid_judgment(data.sample.error)):
                panel["fatal"].add(key)


@metric(scores="unreduced")
def eigenbench_rankings():
    """EigenTrust, Elo, rank, optional Elo bootstrap intervals, and judgment coverage."""
    def calculate(scores: list[SampleScore]) -> dict:
        config = scores[0].score.metadata["analysis"] if scores else _active_analysis.get()
        if config is None:
            return {"ranking_available": 0.0, "coverage/scored": 0.0,
                    "coverage/planned": float("nan"), "coverage/scored_fraction": float("nan")}
        # A score without a record could not be rebuilt; it counts as unscored.
        records = [score.score.metadata["record"] for score in scores if "record" in score.score.metadata]
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
        from eigenbench.inspect.export import records_fingerprint
        from eigenbench.analysis.eigentrust import NonUniqueStationaryError

        fingerprint = records_fingerprint([{"analysis": config}, *records])
        if panel is not None and panel["fingerprint"] == fingerprint:
            return dict(panel["metrics"])
        try:
            values = compute_panel_metrics(records, config)
        except NonUniqueStationaryError as exc:
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
            values = {**_coverage_metrics(len(records), config), "ranking_unavailable_nonunique_eigentrust": 1.0}
        except Exception as exc:  # analysis must never fail the collection it scores
            warnings.warn(f"Panel analysis failed; collection is unaffected: {exc}",
                          RuntimeWarning, stacklevel=2)
            values = {**_coverage_metrics(len(records), config), "ranking_error": 1.0}
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
                from eigenbench.inspect.export import direct_record
                record = direct_record(state.metadata, state.store, config["criteria"])
            else:
                from eigenbench.inspect.pairwise import pairwise_record
                record = {**pairwise_record(state.metadata, state.store, config["criteria"]),
                          "epoch": int(state.epoch)}
        except ValueError as exc:
            return Score(value=float("nan"), explanation=str(exc), metadata=metadata)
        return Score(value=1.0, metadata={**metadata, "record": record})
    return score


@solver
def _records_only():
    async def solve(state, generate):
        return state
    return solve
