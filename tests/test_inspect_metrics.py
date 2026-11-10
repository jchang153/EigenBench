"""Native panel metrics against frozen pre-migration analysis and real Inspect logs."""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from inspect_pipeline.metrics import compute_panel_metrics

FIXTURES = Path(__file__).parent / "fixtures" / "inspect_migration"


def _fixture(name, tmp_path, *, artifacts=False):
    rows = [json.loads(line) for line in (FIXTURES / f"{name}.jsonl").read_text().splitlines()]
    return rows, {
        "model_order": ["a", "b", "c"], "criteria": ["c1", "c2"],
        "evaluation": {"mode": "direct_rating" if name == "direct" else "pairwise_btd"},
        "collection": {"evaluations_path": str(tmp_path / "evaluations.jsonl")},
        "planned": len(rows), "write_artifacts": artifacts,
        "training": {"enabled": artifacts, "output_dir": str(tmp_path / "analysis"),
                     "separate_criteria": False, "max_epochs": 3, "dims": [2], "batch_size": 32,
                     "bootstrap": {"enabled": True, "n_bootstraps": 3, "random_seed": 23}},
    }


def _seed():
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)


def _metrics(log):
    result = next(result for result in log.results.scores if result.name == "panel_scorer")
    return {name: metric.value for name, metric in result.metrics.items()}


@pytest.mark.parametrize("name", ["direct", "pairwise"])
@pytest.mark.parametrize("artifacts", [False, True])
def test_native_metrics_match_frozen_baseline(name, artifacts, tmp_path):
    """Reference numbers come from baseline a903a21, not the new helper under test.

    CPU float64 direct results use 1e-10 absolute tolerance. BTD uses float32
    weights and frozen text output, so its trust tolerance is 1e-7, Elo 1e-4.
    """
    rows, config = _fixture(name, tmp_path, artifacts=artifacts)
    frozen = json.loads((FIXTURES / "analysis.json").read_text())[name]
    _seed()
    actual = compute_panel_metrics(rows, config)
    assert actual["ranking_available"] == 1
    assert actual.get("artifacts_written", 0) == int(artifacts)
    assert actual["coverage/scored"] == actual["coverage/planned"] == len(rows)
    assert actual["coverage/scored_fraction"] == 1
    prefix = "" if name == "direct" else "btd_d2/"
    folder = "direct_rating" if name == "direct" else "btd_d2"
    trust = np.fromstring(frozen[f"{folder}/eigentrust.txt"].split("[")[1].split("]")[0], sep=",")
    from pipeline.train.train import eigentrust_to_elo
    elo = eigentrust_to_elo(trust, len(trust))
    if name == "direct":
        exact = sorted(frozen["direct_rating/summary.json"], key=lambda row: row["model_index"])
        trust = np.asarray([row["eigentrust"] for row in exact])
        elo = np.asarray([row["eigenbench_elo"] for row in exact])
    for i, label in enumerate(config["model_order"]):
        assert actual[f"{prefix}{label}/eigentrust"] == pytest.approx(trust[i], abs=1e-10 if name == "direct" else 1e-7, rel=0)
        assert actual[f"{prefix}{label}/elo"] == pytest.approx(elo[i], abs=1e-10 if name == "direct" else 1e-4, rel=0)
        assert actual[f"{prefix}{label}/rank"] == 1 + np.sum(trust > trust[i])
    for row in frozen[f"{folder}/bootstrap/summary.json"]:
        for field in ("elo_mean", "elo_std", "elo_ci_lower", "elo_ci_upper"):
            assert actual[f"{prefix}{row['model_name']}/{field}"] == pytest.approx(
                row[field], abs=1e-10 if name == "direct" else 1e-4, rel=0)
    if artifacts:
        # Saved bootstrap draws and native intervals must describe the SAME fit.
        saved = json.loads((tmp_path / "analysis" / folder / "bootstrap" / "samples.json").read_text())
        expected = frozen[f"{folder}/bootstrap/samples.json"]
        for a, e in zip(saved, expected):
            np.testing.assert_allclose(a["trust_vector"], e["trust_vector"], atol=1e-7, rtol=0)
            np.testing.assert_allclose(a["elo_vector"], e["elo_vector"], atol=1e-4, rtol=0)
            np.testing.assert_allclose(a["trust_matrix"], e["trust_matrix"], atol=1e-7, rtol=0)


def test_effective_model_order_and_self_setting(tmp_path):
    rows, config = _fixture("direct", tmp_path)
    config["training"]["bootstrap"]["enabled"] = False
    baseline = compute_panel_metrics(rows, config)
    config["model_order"] = ["c", "a", "b"]
    reordered = compute_panel_metrics(rows, config)
    for label in config["model_order"]:
        assert reordered[f"{label}/eigentrust"] == pytest.approx(baseline[f"{label}/eigentrust"])
    config["evaluation"]["direct_rating"] = {"include_self": False}
    with pytest.raises(ValueError, match="self-rating record"):
        compute_panel_metrics(rows, config)
    rows = [r for r in rows if r["judge"]["name"] != r["evaluee"]["name"]]
    config["planned"] = len(rows)
    result = compute_panel_metrics(rows, config)
    assert result["ranking_available"] == 1
    assert result["coverage/scored"] == 18
    assert result["a/eigentrust"] != baseline["a/eigentrust"]


def test_repeated_comparisons_are_kept_by_cleaning(tmp_path):
    rows, config = _fixture("pairwise", tmp_path)
    config["planned"] *= 2
    actual = compute_panel_metrics(rows + rows, config)
    assert actual["coverage/scored"] == 108
    assert actual["comparisons_parsed"] == 216
    assert actual["comparisons_retained"] == 216
    assert actual["comparisons_dropped"] == 0
    assert actual["ranking_available"] == 1


def test_rectangular_criteria_fails_before_fitting(tmp_path, monkeypatch):
    import pipeline.train.pairwise_analysis as analysis
    rows, config = _fixture("pairwise", tmp_path)
    config["training"]["separate_criteria"] = True
    def forbidden(*args, **kwargs):
        raise AssertionError("rectangular criteria must be rejected before fitting")
    monkeypatch.setattr(analysis, "train_vector_bt", forbidden)
    with pytest.raises(ValueError, match="rectangular trust matrix"):
        compute_panel_metrics(rows, config)


@pytest.mark.parametrize("phased", [False, True])
def test_real_pairwise_inspect_metrics_and_phased_coverage(tmp_path, phased):
    from inspect_ai.log import read_eval_log
    from test_inspect_pairwise import _write_spec, _panel
    from inspect_pipeline.export import export_log
    from inspect_pipeline.pairwise_collect import collect_pairwise_inspect
    from inspect_pipeline.coverage import report_from_run_dir
    calls = []
    path = Path(_write_spec(tmp_path, groups=1))
    path.write_text(path.read_text() + f"\nRUN_SPEC['collection']['inspect']['phased'] = {phased!r}\n" +
                    "RUN_SPEC['training'] = {'enabled': True, 'max_epochs': 3, 'dims': [2]}\n")
    status = {}
    rows = collect_pairwise_inspect(str(path), models=_panel(calls), analysis_status=status)
    assert status == {"artifacts_written": True}
    manifest = json.loads((tmp_path / "inspect_run.json").read_text())
    logs = [read_eval_log(str(tmp_path / "inspect_logs" / name)) for name in manifest["log_files"]]
    log = logs[0]
    values = _metrics(log)
    assert values["coverage/scored"] == values["coverage/planned"] == len(rows) == 12
    assert {f"btd_d2/{nick}/elo" for nick in ("a", "b", "c")} <= values.keys()
    assert len(calls) == 24  # 6 responses + 6 reflections + 12 comparisons; finalization adds zero
    report = json.loads((tmp_path / "collection_report.json").read_text())
    assert report["planned"] == report["logged"] == report["exported"] == 12
    assert report["failed"] == report["omitted"] == report["unstarted"] == 0
    # Reconstructing coverage over the manifest also ignores scoring-only samples.
    reconstructed = report_from_run_dir(tmp_path)
    assert reconstructed["logged"] == reconstructed["exported"] == 12
    if phased:
        assert log.eval.metadata["eigenbench_phase"] == "panel_metrics"
        assert set(log.eval.metadata["eigenbench"]["source_logs"]) == set(manifest["log_files"][1:])
        with pytest.raises(ValueError, match="scoring-only full-panel"):
            export_log(log)


def test_real_direct_metrics_match_export(tmp_path):
    from inspect_ai import eval as inspect_eval
    from inspect_pipeline.eigenbench import eigenbench
    from inspect_pipeline.export import records_from_log
    # A minimal callable mock exercises persistence without external providers.
    (tmp_path / "scenarios.json").write_text('["scenario zero", "scenario one"]')
    (tmp_path / "constitution.json").write_text('["criterion"]')
    path = tmp_path / "spec.py"
    # Build a minimal model fixture without invoking any provider.
    path.write_text('''from inspect_ai.model import ModelOutput, get_model

def mock():
    def out(input, tools, tool_choice, config):
        return ModelOutput.from_content(model="mockllm/model", content="<criterion_1_rating>7</criterion_1_rating>")
    return get_model("mockllm/model", custom_outputs=out, memoize=False)
RUN_SPEC = {"models": {"a": mock(), "b": mock()}, "evaluation": {"mode": "direct_rating"},
"dataset": {"path": "scenarios.json"}, "constitution": {"path": "constitution.json", "num_criteria": 1},
"collection": {"inspect": {"cache": False}, "sampler_mode": "all_to_all"}, "training": {"enabled": False}}
''')
    # Native task construction, execution and persisted log (no mocked metric machinery).
    logs = inspect_eval(tasks=eigenbench(str(path)), model=None, display="none",
                        log_dir=str(tmp_path / "logs"))
    assert logs[0].status == "success"
    rows = records_from_log(logs[0])
    values = _metrics(logs[0])
    assert values["coverage/scored"] == 8
    assert values["a/eigentrust"] == values["b/eigentrust"] == 0.5
    assert values["a/elo"] == values["b/elo"] == 1500
    assert values["a/rank"] == values["b/rank"] == 1
    assert len(rows) == 8


def test_historical_analysis_preserves_record_names_with_reordered_spec(tmp_path):
    from scripts.run_train import main
    rows, config = _fixture("pairwise", tmp_path, artifacts=True)
    path = tmp_path / "spec.py"
    evaluations = tmp_path / "evaluations.jsonl"
    evaluations.write_text("".join(json.dumps(row) + "\n" for row in rows))
    spec = {"name": "historical", "models": dict.fromkeys(["c", "a", "b"], "mock"),
            "evaluation": config["evaluation"], "constitution": {"num_criteria": 2},
            "collection": config["collection"], "training": config["training"]}
    path.write_text(f"RUN_SPEC = {spec!r}\n")
    _seed()
    main(str(path))
    saved = json.loads((tmp_path / "analysis" / "btd_d2" / "bootstrap" / "summary.json").read_text())
    frozen = json.loads((FIXTURES / "analysis.json").read_text())["pairwise"]["btd_d2/bootstrap/summary.json"]
    assert [r["model_name"] for r in saved] == [r["model_name"] for r in frozen]
    for row, expected in zip(saved, frozen):
        assert row["elo_mean"] == pytest.approx(expected["elo_mean"], abs=1e-4)


def test_requested_epoch_coverage_survives_an_entire_failed_epoch(tmp_path):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.solver import chain, solver
    from inspect_pipeline.pairwise import eigenbench_pairwise
    from test_pairwise_runner import write_spec, panel

    @solver
    def fail_second_epoch():
        async def solve(state, generate):
            if state.epoch == 2:
                raise RuntimeError("deliberate offline epoch failure")
            return state
        return solve

    calls = []
    task = eigenbench_pairwise(write_spec(tmp_path, groups=1), models=panel(calls))
    task.solver = chain(fail_second_epoch(), task.solver)
    with pytest.warns(RuntimeWarning, match="multiple Inspect epochs"):
        [log] = inspect_eval(task, model=None, epochs=2, fail_on_error=False, display="none",
                             log_dir=str(tmp_path / "epochs"))
    assert log.status == "success"
    assert sum(sample.error is not None for sample in log.samples) == 12
    values = _metrics(log)
    assert values["coverage/scored"] == 12
    assert values["coverage/planned"] == 24
    assert values["coverage/scored_fraction"] == 0.5
    assert values["ranking_available"] == 0
    assert values["ranking_unsupported_epochs"] == 1
    assert not any(key.endswith("/elo") for key in values)


@pytest.mark.parametrize("mode", ["direct_rating", "pairwise_btd"])
def test_progress_and_completion_share_one_fit_and_bootstrap(tmp_path, monkeypatch, mode):
    """Slow actual generation forces Inspect's progress metric callback to run."""
    import time
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model
    import inspect_pipeline.metrics as metrics
    from inspect_pipeline.eigenbench import eigenbench
    from inspect_pipeline.pairwise import eigenbench_pairwise

    (tmp_path / "scenarios.json").write_text('["scenario"]')
    (tmp_path / "constitution.json").write_text('["criterion"]')
    spec = {"models": dict.fromkeys(["a", "b"], "inspect:mockllm/model"),
            "evaluation": {"mode": mode}, "dataset": {"path": "scenarios.json"},
            "constitution": {"path": "constitution.json", "num_criteria": 1},
            "collection": {"sampler_mode": "all_to_all" if mode == "direct_rating" else "random_judge_group",
                           "failure_policy": "omit_invalid_judgments", "group_size": 2,
                           "inspect": {"cache": False}, "openrouter": {"max_attempts": 1}},
            "training": {"enabled": True, "max_epochs": 1, "dims": [2],
                         "bootstrap": {"enabled": True, "n_bootstraps": 2, "random_seed": 23}}}
    path = tmp_path / "spec.py"
    path.write_text(f"RUN_SPEC = {spec!r}\n")

    def mock(nick):
        def output(input, tools, tool_choice, config):
            system = input[0].text
            if system.startswith("Without making"):
                text = "answer " + nick
            elif system.startswith("You are acting as a judge. Carefully consider the following response"):
                text = "reflection"
            else:
                time.sleep(1.1)
                if mode == "direct_rating":
                    text = ("invalid" if nick == "a" and "answer b" in input[1].text else
                            "<criterion_1_rating>7</criterion_1_rating>")
                else:
                    text = "<criterion_1_choice>0</criterion_1_choice>"
            return ModelOutput.from_content(model="mockllm/model", content=text)
        return get_model("mockllm/model", custom_outputs=output, memoize=False)

    invocations = []
    original = metrics.compute_panel_metrics

    def counted(records, config):
        invocations.append(len(records))
        return original(records, config)

    monkeypatch.setattr(metrics, "compute_panel_metrics", counted)
    build = eigenbench if mode == "direct_rating" else eigenbench_pairwise
    task = build(str(path), models={nick: mock(nick) for nick in ("a", "b")}, write_artifacts=True)
    [log] = inspect_eval(task, model=None, display="none", max_samples=1, fail_on_error=False,
                         log_dir=str(tmp_path / "logs"))
    assert log.status == "success"
    expected = 3 if mode == "direct_rating" else 2
    assert invocations == [expected]  # No partial-panel fit or second full-panel fit.
    result = _metrics(log)
    assert result["ranking_available"] == result["artifacts_written"] == 1
    assert result["coverage/scored"] == expected
    assert result["coverage/planned"] == (4 if mode == "direct_rating" else 2)
    folder = "direct_rating" if mode == "direct_rating" else "btd_d2"
    assert (tmp_path / folder / "bootstrap" / "samples.json").is_file()


def test_public_rescore_preserves_requested_epochs_and_coverage(tmp_path):
    from inspect_ai import eval as inspect_eval, score
    from inspect_ai.log import read_eval_log
    from inspect_pipeline.metrics import panel_scorer
    from inspect_pipeline.pairwise import eigenbench_pairwise
    from test_pairwise_runner import write_spec, panel
    task = eigenbench_pairwise(write_spec(tmp_path, groups=1), models=panel([]))
    with pytest.warns(RuntimeWarning, match="multiple Inspect epochs"):
        [log] = inspect_eval(task, model=None, epochs=2, display="none",
                             log_dir=str(tmp_path / "epochs"))
    # Read persisted sample metadata, rather than relying on in-memory state.
    log = read_eval_log(log.location)
    with pytest.warns(RuntimeWarning, match="multiple Inspect epochs"):
        rescored = score(log, panel_scorer(log.eval.metadata["eigenbench"]["analysis"]),
                         action="overwrite")
    assert _metrics(rescored) == _metrics(log)
    assert _metrics(rescored)["coverage/planned"] == 24
    assert _metrics(rescored)["ranking_unsupported_epochs"] == 1


@pytest.mark.parametrize("phased", [False, True])
def test_pairwise_collection_only_bypasses_fitting_and_invalid_training(tmp_path, monkeypatch, phased):
    from inspect_ai.log import read_eval_log
    import inspect_pipeline.metrics as metrics
    from inspect_pipeline.pairwise_collect import collect_pairwise_inspect
    from test_inspect_pairwise import _write_spec, _panel
    path = Path(_write_spec(tmp_path, groups=1))
    path.write_text(path.read_text() + f"\nRUN_SPEC['collection']['inspect']['phased'] = {phased!r}\n" +
                    "RUN_SPEC['training'] = {'enabled': False, 'separate_criteria': True}\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("collection-only must not calculate panel metrics")
    monkeypatch.setattr(metrics, "compute_panel_metrics", forbidden)
    rows = collect_pairwise_inspect(str(path), models=_panel([]), scoring=False)
    assert len(rows) == 12
    manifest = json.loads((tmp_path / "inspect_run.json").read_text())
    for name in manifest["log_files"]:
        log = read_eval_log(str(tmp_path / "inspect_logs" / name))
        assert (log.eval.metadata or {}).get("eigenbench_phase") != "panel_metrics"
        if log.results:
            assert all(result.name != "panel_scorer" for result in log.results.scores)


def test_native_no_score_bypasses_unsupported_training(tmp_path, monkeypatch):
    from inspect_ai import eval as inspect_eval
    import inspect_pipeline.metrics as metrics
    from inspect_pipeline.pairwise import eigenbench_pairwise
    from test_inspect_pairwise import _write_spec, _panel
    path = Path(_write_spec(tmp_path, groups=1))
    path.write_text(path.read_text() + "\nRUN_SPEC['training'] = {'separate_criteria': True}\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("Inspect --no-score must not calculate panel metrics")
    monkeypatch.setattr(metrics, "compute_panel_metrics", forbidden)
    [log] = inspect_eval(eigenbench_pairwise(str(path), models=_panel([])), model=None,
                         score=False, display="none", log_dir=str(tmp_path / "logs"))
    assert log.status == "success"
    assert not log.results or not log.results.scores
