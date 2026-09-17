"""Focused regressions for this change."""

def test_omission_report_matches_exports_and_preserves_failure_details():
    from types import SimpleNamespace
    from inspect_pipeline.coverage import report_from_samples

    def sample(i, error=None, **extra):
        return SimpleNamespace(id=str(i), error=SimpleNamespace(message=error) if error else None,
            metadata={"scenario_index": i, "scenario": f"Question {i}", "judge_nick": "judge", "eval_nick": "model", **extra})

    records = [{"record_type": "direct_rating", "scenario_index": 0, "judge": {"name": "judge"}, "evaluee": {"name": "model"}}]
    result = report_from_samples([sample(0), sample(1, "reflection truncated"), sample(2)], records, log_file="run.eval")
    assert (result["logged_samples"], result["exported_samples"], result["omitted_samples"], result["failed_samples"]) == (3, 1, 2, 1)
    assert result["omissions"][0]["error"] == "reflection truncated"
    assert result["omissions"][1]["reason"] == "not_exported"
    assert result["omissions"][0]["judge"] == "judge"


def test_omission_report_rejects_ambiguous_epochs_and_supports_pairwise():
    from types import SimpleNamespace
    import pytest
    from inspect_pipeline.coverage import report_from_samples

    sample = SimpleNamespace(id="p1", error=None, metadata={"scenario_index": 1, "scenario": "q", "judge_nick": "j", "eval1_nick": "a", "eval2_nick": "b"})
    records = [{"scenario_index": 1, "judge_name": "j", "eval1_name": "a", "eval2_name": "b"}]
    assert report_from_samples([sample], records, log_file="run.eval")["omitted_samples"] == 0
    with pytest.raises(ValueError, match="ambiguous"):
        report_from_samples([sample, sample], records, log_file="run.eval")


def test_staging_publishes_omission_report_reference(tmp_path, monkeypatch):
    import json
    import inspect_pipeline.coverage as coverage
    from scripts.upload_results import stage_run

    run = tmp_path / "run"
    (run / "direct_rating").mkdir(parents=True)
    (run / "inspect_logs").mkdir()
    (run / "spec.py").write_text("RUN_SPEC = {'models': {}, 'evaluation': {'mode': 'direct_rating'}}")
    (run / "inspect_logs/run.eval").write_bytes(b"test log")
    (run / "inspect_run.json").write_text(json.dumps({"log_file": "run.eval"}))
    (run / "evaluations.jsonl").write_text("")
    report = {"source": "inspect_log", "omitted_samples": 1}
    monkeypatch.setattr(coverage, "build_collection_report", lambda *args: report)
    meta, _ = stage_run("demo", run, tmp_path / "staged")
    assert meta["inspect"]["collection_report_file"] == "collection_report.json"
    assert json.loads((tmp_path / "staged/runs/demo/collection_report.json").read_text()) == report


def test_uploaded_spec_can_resolve_its_own_file(tmp_path):
    from scripts.upload_results import parse_spec
    path = tmp_path / "spec.py"
    path.write_text('from pathlib import Path\nRUN_SPEC = {"name": Path(__file__).parent.name}\n')
    assert parse_spec(path)["name"] == tmp_path.name
