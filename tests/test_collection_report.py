"""The version 2 collection report and its publication."""

import json
from types import SimpleNamespace as NS

import pytest


def sample(i, error=None, **extra):
    md = {"scenario_index": i, "scenario": f"Question {i}", "judge_nick": "judge", "eval_nick": "model", **extra}
    return NS(id=str(i), error=NS(message=error) if error else None, metadata=md)


def log(name, samples, **task_md):
    return NS(location=f"/logs/{name}", status="success", samples=samples, eval=NS(metadata=task_md))


def direct(i, judge="judge", model="model"):
    return {"record_type": "direct_rating", "scenario_index": i, "judge": {"name": judge},
            "evaluee": {"name": model}, "sampling": {"round": 0}}


def test_report_preserves_failure_details_and_counts_unstarted():
    from inspect_pipeline.coverage import collection_part

    planned = [{"scenario_index": i, "judge_nick": "judge", "eval_nick": "model"} for i in range(4)]
    logs = [log("run.eval", [sample(0), sample(1, "reflection truncated"), sample(2)],
                eigenbench={"judge": "judge"})]
    part = collection_part("collection", planned=planned, logs=logs, records=[direct(0)])

    counts = {name: part[name] for name in ("planned", "logged", "exported", "failed", "omitted", "unstarted")}
    assert counts == {"planned": 4, "logged": 3, "exported": 1, "failed": 1, "omitted": 2, "unstarted": 1}
    first, second = part["omissions"]
    assert (first["error"], first["reason"], first["judge"], first["log_file"]) == (
        "reflection truncated", "sample_error", "judge", "run.eval")
    assert (second["reason"], second["error"]) == ("not_exported", None)
    assert part["unstarted_judgments"] == [{"scenario_index": 3, "judge": "judge", "models": ["model"]}]
    assert part["logs"] == [{"file": "run.eval", "role": "judgments", "model": "judge",
                             "status": "success", "samples": 3, "failed": 1}]


def test_report_supports_pairwise_and_rejects_ambiguous_keys():
    from inspect_pipeline.coverage import collection_part

    pair = NS(id="p1", error=None, metadata={"scenario_index": 1, "scenario": "q", "judge_nick": "j",
                                             "eval1_nick": "a", "eval2_nick": "b"})
    records = [{"scenario_index": 1, "judge_name": "j", "eval1_name": "a", "eval2_name": "b"}]
    part = collection_part("extension", planned=None, logs=[log("pair.eval", [pair])], records=records)
    assert (part["logged"], part["exported"], part["omitted"]) == (1, 1, 0)
    with pytest.raises(ValueError, match="ambiguous"):
        collection_part("extension", planned=None, logs=[log("pair.eval", [pair, pair])], records=records)


def test_extending_a_run_without_a_report_keeps_its_totals_unknown(tmp_path):
    from inspect_pipeline.coverage import collection_part, write_collection_report

    part = collection_part("extension", planned=[{"scenario_index": 0, "judge_nick": "judge", "eval_nick": "model"}],
                           logs=[log("extend.eval", [sample(0)], eigenbench={"judge": "judge"})], records=[direct(0)])
    write_collection_report(tmp_path, part, failure_policy="strict", append=True)

    report = json.loads((tmp_path / "collection_report.json").read_text())
    earlier, extension = report["collections"]
    assert earlier["status"] == "unknown" and earlier["logged"] is None
    assert (extension["planned"], extension["exported"]) == (1, 1)
    # What the run collected first is unknown, so no total can be claimed.
    assert all(report[name] is None for name in ("planned", "logged", "exported", "failed", "omitted", "unstarted"))
    assert json.loads((tmp_path / "inspect_run.json").read_text()) == {"collection_report_file": "collection_report.json"}


def _run_folder(tmp_path, **inspect_run):
    run = tmp_path / "run"
    (run / "direct_rating").mkdir(parents=True)
    (run / "inspect_logs").mkdir()
    (run / "spec.py").write_text("RUN_SPEC = {'models': {}, 'evaluation': {'mode': 'direct_rating'}}")
    for name in ("judge_a.eval", "judge_b.eval", "responses.eval"):
        (run / "inspect_logs" / name).write_bytes(b"test log")
    (run / "inspect_run.json").write_text(json.dumps(inspect_run))
    (run / "evaluations.jsonl").write_text(json.dumps(direct(0, "a")) + "\n" + json.dumps(direct(0, "b")) + "\n")
    return run


def test_upload_publishes_the_run_report_and_every_log(tmp_path):
    from scripts.upload_results import stage_run

    names = ["judge_a.eval", "judge_b.eval", "responses.eval"]
    run = _run_folder(tmp_path, log_file="judge_a.eval", log_files=names)
    report = {"schema_version": 2, "source": "inspect_logs", "planned": 2, "collections": []}
    (run / "collection_report.json").write_text(json.dumps(report))

    meta, _ = stage_run("demo", run, tmp_path / "staged")
    staged = tmp_path / "staged/runs/demo"
    assert meta["inspect"] == {"log_file": "judge_a.eval", "log_files": names,
                               "collection_report_file": "collection_report.json"}
    assert all((staged / name).exists() for name in names)
    assert json.loads((staged / "collection_report.json").read_text()) == report


def test_old_run_gets_a_version_2_report_with_unknown_plan(tmp_path, monkeypatch):
    import inspect_ai.log
    from inspect_pipeline.coverage import report_from_run_dir
    from scripts.upload_results import stage_run

    run = _run_folder(tmp_path, log_file="judge_a.eval", log_files=["judge_a.eval", "judge_b.eval"])
    fakes = {
        "judge_a.eval": log("judge_a.eval", [sample(0, judge_nick="a")], eigenbench={"judge": "a"}),
        "judge_b.eval": log("judge_b.eval", [sample(0, judge_nick="b"), sample(1, "timed out", judge_nick="b")],
                            eigenbench={"judge": "b"}),
    }
    monkeypatch.setattr(inspect_ai.log, "read_eval_log", lambda path: fakes[path.rsplit("/", 1)[-1]])

    report = report_from_run_dir(run)
    assert report["schema_version"] == 2
    assert (report["planned"], report["unstarted"]) == (None, None)
    assert (report["logged"], report["exported"], report["failed"], report["omitted"]) == (3, 2, 1, 1)
    [part] = report["collections"]
    assert [entry["file"] for entry in part["logs"]] == ["judge_a.eval", "judge_b.eval"]
    assert part["unstarted_judgments"] is None

    meta, _ = stage_run("demo", run, tmp_path / "staged")
    assert meta["inspect"]["collection_report_file"] == "collection_report.json"
    assert json.loads((tmp_path / "staged/runs/demo/collection_report.json").read_text()) == report


def test_uploaded_spec_can_resolve_its_own_file(tmp_path):
    from scripts.upload_results import parse_spec
    path = tmp_path / "spec.py"
    path.write_text('from pathlib import Path\nRUN_SPEC = {"name": Path(__file__).parent.name}\n')
    assert parse_spec(path)["name"] == tmp_path.name
