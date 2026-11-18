import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.upload_results import build_index_entry, stage_run


def _meta(dataset: dict) -> dict:
    return {
        "timestamp": "2026-08-26T00:00:00+00:00",
        "dataset": dataset,
        "models": {},
    }


def test_build_index_entry_uses_inclusive_counted_range() -> None:
    entry = build_index_entry(
        "run",
        _meta({"id": "airisk", "start": 0, "count": 200}),
        [],
    )

    assert entry["scenario"] == "airisk [0-199]"


def test_build_index_entry_omits_range_when_count_is_unspecified() -> None:
    entry = build_index_entry(
        "run",
        _meta({"id": "airisk", "start": 50}),
        [],
    )

    assert entry["scenario"] == "airisk"


def _write_spec(run, mode="direct_rating", **settings):
    spec = {"evaluation": {"mode": mode}, "models": {"a": "provider/model"}, **settings}
    (run / "spec.py").write_text(f"RUN_SPEC = {spec!r}\n", encoding="utf-8")


@pytest.mark.parametrize("mode, output", [("direct_rating", "direct_rating"), ("pairwise_btd", "btd_d2")])
@pytest.mark.parametrize("bootstrap_enabled", [True, False, None])
def test_stage_uses_bootstrap_artifacts_only_when_enabled(tmp_path, mode, output, bootstrap_enabled):
    run = tmp_path / "run"
    analysis = run / output
    bootstrap = analysis / "bootstrap"
    bootstrap.mkdir(parents=True)
    training = {} if bootstrap_enabled is None else {"bootstrap": {"enabled": bootstrap_enabled}}
    _write_spec(run, mode, training=training)
    (analysis / "eigentrust.txt").write_text("EigenTrust scores:\n[1.0]\n", encoding="utf-8")
    (analysis / "summary.json").write_text(
        json.dumps([{"model_index": 0, "model_name": "a", "eigenbench_elo": 1575.0}]),
        encoding="utf-8",
    )
    (bootstrap / "summary.json").write_text(json.dumps([{"elo_mean": 900.0}]), encoding="utf-8")
    (bootstrap / "bootstrap_elo.png").write_bytes(b"bootstrap plot")
    (bootstrap / "samples.json").write_text("[]", encoding="utf-8")

    meta, summary = stage_run("demo", run, tmp_path / "staged")

    dest = tmp_path / "staged/runs/demo"
    expected_point = 1575.0 if mode == "direct_rating" else 1500.0
    assert summary[0]["elo_mean"] == (900.0 if bootstrap_enabled else expected_point)
    assert json.loads((dest / "summary.json").read_text()) == summary
    assert meta["bootstrap"]["enabled"] is bool(bootstrap_enabled)
    assert ("bootstrap_elo.png" in meta["artifacts"]["images"]) is bool(bootstrap_enabled)
    assert (dest / "images/bootstrap_elo.png").exists() is bool(bootstrap_enabled)
    has_samples = bool(bootstrap_enabled) and mode == "direct_rating"
    assert ("data/bootstrap_samples.json" in meta["artifacts"]["data"]) is has_samples
    assert (dest / "data/bootstrap_samples.json").exists() is has_samples


def test_stage_uses_configured_analysis_dir_exclusively(tmp_path):
    run = tmp_path / "run"
    (run / "direct_rating").mkdir(parents=True)
    _write_spec(run, training={"output_dir": "latest"})
    with pytest.raises(FileNotFoundError, match="No direct_rating directory"):
        stage_run("demo", run, tmp_path / "staged")

    analysis = run / "latest/direct_rating"
    analysis.mkdir(parents=True)
    (analysis / "eigentrust.txt").write_text("EigenTrust scores:\n[1.0]\n", encoding="utf-8")

    _, summary = stage_run("demo", run, tmp_path / "staged")

    assert summary[0]["elo_mean"] == 1500.0


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("suffix", [".jsonl", ".json"])
def test_stage_uses_configured_evaluations_and_their_inspect_artifacts(tmp_path, absolute, suffix):
    from inspect_pipeline.coverage import evaluations_sha256

    run = tmp_path / "run"
    (run / "direct_rating").mkdir(parents=True)
    export_dir = tmp_path / "collection" if absolute else run / "collection"
    log_dir = export_dir / "logs"
    log_dir.mkdir(parents=True)
    export = export_dir / f"judgments{suffix}"
    path_value = export if absolute else export.relative_to(run)
    _write_spec(run, collection={"evaluations_path": str(path_value), "inspect": {"log_dir": "logs"}})
    (run / "evaluations.jsonl").write_text('{"judge": "old"}\n', encoding="utf-8")
    (run / "inspect_run.json").write_text('{"log_file": "old.eval"}', encoding="utf-8")
    (run / "inspect_logs").mkdir()
    (run / "inspect_logs/old.eval").write_bytes(b"old log")
    record = {"judge": "current"}
    export.write_text(json.dumps([record] if suffix == ".json" else record) + "\n", encoding="utf-8")
    (log_dir / "current.eval").write_bytes(b"current log")
    (export_dir / "inspect_run.json").write_text('{"log_file": "current.eval"}', encoding="utf-8")
    report = {
        "schema_version": 2,
        "planned": 1,
        "collections": [{"logs": [{"file": "current.eval"}]}],
        "evaluations_sha256": evaluations_sha256(export),
    }
    (export_dir / "collection_report.json").write_text(json.dumps(report), encoding="utf-8")

    meta, _ = stage_run("demo", run, tmp_path / "staged")

    dest = tmp_path / "staged/runs/demo"
    staged = dest / "evaluations.jsonl"
    assert [json.loads(line) for line in staged.read_text().splitlines()] == [record]
    assert (dest / "current.eval").read_bytes() == b"current log"
    assert not (dest / "old.eval").exists()
    assert meta["inspect"]["log_file"] == "current.eval"
    assert json.loads((dest / "collection_report.json").read_text()) == {
        **report, "evaluations_sha256": evaluations_sha256(staged),
    }


def test_stage_does_not_replace_a_missing_configured_evaluations_file_with_default(tmp_path):
    run = tmp_path / "run"
    (run / "direct_rating").mkdir(parents=True)
    _write_spec(run, collection={"evaluations_path": "collection/judgments.jsonl"})
    (run / "evaluations.jsonl").write_text('{"judge": "old"}\n', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Configured evaluations file not found"):
        stage_run("demo", run, tmp_path / "staged")

    assert not (tmp_path / "staged/runs/demo").exists()


def test_stage_rebuilds_report_from_configured_logs_and_published_records(tmp_path, monkeypatch):
    import inspect_ai.log
    from inspect_pipeline.coverage import evaluations_sha256

    run = tmp_path / "run"
    (run / "direct_rating").mkdir(parents=True)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_spec(run, collection={"evaluations_path": "judgments.json", "inspect": {"log_dir": str(log_dir)}})
    record = {"record_type": "direct_rating", "scenario_index": 3,
              "judge": {"name": "a"}, "evaluee": {"name": "b"}}
    (run / "judgments.json").write_text(json.dumps([record]), encoding="utf-8")
    (run / "inspect_run.json").write_text('{"log_file": "judgments.eval"}', encoding="utf-8")
    (log_dir / "judgments.eval").touch()

    log_sample = SimpleNamespace(error=None, metadata={"scenario_index": 3, "judge_nick": "a", "eval_nick": "b"})

    def read_log(path):
        assert Path(path) == log_dir / "judgments.eval"
        return SimpleNamespace(location=path, samples=[log_sample], status="success", eval=SimpleNamespace())

    monkeypatch.setattr(inspect_ai.log, "read_eval_log", read_log)
    stage_run("demo", run, tmp_path / "staged")

    dest = tmp_path / "staged/runs/demo"
    report = json.loads((dest / "collection_report.json").read_text())
    assert (report["logged"], report["exported"], report["omitted"]) == (1, 1, 0)
    assert report["evaluations_sha256"] == evaluations_sha256(dest / "evaluations.jsonl")
