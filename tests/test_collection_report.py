"""The version 2 collection report and its publication."""

import json
from pathlib import Path
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


def pair(sample_id, error=None, **extra):
    md = {"scenario_index": 1, "scenario": "q", "judge_nick": "j", "eval1_nick": "a", "eval2_nick": "b", **extra}
    return NS(id=sample_id, error=NS(message=error) if error else None, metadata=md)


PAIR_RECORD = {"scenario_index": 1, "judge_name": "j", "eval1_name": "a", "eval2_name": "b"}
PAIR_PLAN = {"scenario_index": 1, "judge_nick": "j", "eval1_nick": "a", "eval2_nick": "b"}


def test_recurring_pairs_are_counted_not_rejected():
    from inspect_pipeline.coverage import collection_part

    # The plan asks for the same comparison twice; one of the two was exported.
    part = collection_part("extension", planned=[PAIR_PLAN, PAIR_PLAN],
                           logs=[log("pair.eval", [pair("p1"), pair("p2")])], records=[PAIR_RECORD])
    assert (part["planned"], part["logged"], part["exported"], part["omitted"], part["unstarted"]) == (2, 2, 1, 1, 0)
    # Which of the two identical samples went unexported cannot be told, so none is named.
    [row] = part["omissions"]
    assert (row["sample_id"], row["log_file"], row["models"], row["reason"]) == (None, None, ["a", "b"], "not_exported")

    # A failed repeat is the one omitted, and is named.
    part = collection_part("extension", planned=[PAIR_PLAN, PAIR_PLAN],
                           logs=[log("pair.eval", [pair("p1"), pair("p2", "timed out")])], records=[PAIR_RECORD])
    assert (part["exported"], part["failed"], part["omitted"]) == (1, 1, 1)
    assert [(row["sample_id"], row["error"]) for row in part["omissions"]] == [("p2", "timed out")]

    # Two exported copies of a comparison that only succeeded once cannot be reconciled.
    part = collection_part("extension", planned=None, logs=[log("pair.eval", [pair("p1"), pair("p2", "timed out")])],
                           records=[PAIR_RECORD, PAIR_RECORD])
    assert (part["exported"], part["omitted"], part["failed"]) == (None, None, 1)


def test_epochs_multiply_the_plan():
    from inspect_pipeline.coverage import collection_part

    def epochs_log(name, samples, epochs):
        return NS(location=f"/logs/{name}", status="success", samples=samples,
                  eval=NS(metadata={}, config=NS(epochs=epochs)))

    samples = [NS(**vars(sample(0)), epoch=epoch) for epoch in (1, 2)]
    planned = [{"scenario_index": 0, "judge_nick": "judge", "eval_nick": "model"}]
    part = collection_part("collection", planned=planned, logs=[epochs_log("run.eval", samples, 2)], records=[direct(0)])
    assert (part["planned"], part["logged"], part["exported"], part["omitted"], part["unstarted"]) == (2, 2, 1, 1, 0)
    assert part["omissions"][0]["epoch"] is None  # one of two identical epochs, not named

    # A failed epoch is named, with its epoch.
    failing = [NS(**vars(sample(0)), epoch=1), NS(**vars(sample(0, "timed out")), epoch=2)]
    part = collection_part("collection", planned=planned, logs=[epochs_log("run.eval", failing, 2)], records=[direct(0)])
    assert [(row["epoch"], row["reason"]) for row in part["omissions"]] == [(2, "sample_error")]

    # Logs run with different epochs leave the plan's size unknown.
    logs = [epochs_log("a.eval", samples[:1], 1), epochs_log("b.eval", [NS(**vars(sample(1)), epoch=1)], 2)]
    part = collection_part("collection", planned=planned, logs=logs, records=[direct(0)])
    assert (part["planned"], part["unstarted"]) == (None, None)


def test_rounds_are_told_apart():
    from inspect_pipeline.coverage import collection_part

    samples = [sample(0, sampling_round=0), NS(**{**vars(sample(0, sampling_round=1)), "id": "r1"})]
    records = [{**direct(0), "sampling": {"round": 1}}]
    part = collection_part("collection", planned=None, logs=[log("run.eval", samples)], records=records)
    assert (part["logged"], part["exported"], part["omitted"]) == (2, 1, 1)
    [row] = part["omissions"]
    assert (row["sample_id"], row["round"]) == ("0", 0)


def test_unreconcilable_counts_are_unknown_never_zero():
    from inspect_pipeline.coverage import build_report, collection_part

    logs = [log("run.eval", [sample(0), sample(1, "timed out"), sample(2)])]
    # The export holds a judgment no log accounts for: exported and omitted are unknown.
    part = collection_part("collection", planned=None, logs=logs, records=[direct(0), direct(7)])
    assert (part["exported"], part["omitted"], part["logged"], part["failed"]) == (None, None, 3, 1)
    # Failures are still listed; an unexported judgment is not, since the count is unknown.
    assert [row["reason"] for row in part["omissions"]] == ["sample_error"]
    report = build_report([part], failure_policy="strict")
    assert (report["exported"], report["omitted"], report["failed"]) == (None, None, 1)

    # The logs hold judgments the plan never asked for: unstarted is unknown.
    planned = [{"scenario_index": 0, "judge_nick": "judge", "eval_nick": "model"}]
    part = collection_part("collection", planned=planned, logs=logs, records=[direct(0)])
    assert (part["planned"], part["unstarted"], part["unstarted_judgments"]) == (1, None, None)
    assert (part["exported"], part["omitted"]) == (1, 2)


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


def test_reexported_extension_replaces_its_part(tmp_path):
    from inspect_pipeline.coverage import collection_part, write_collection_report

    first = collection_part("collection", planned=None, logs=[log("run.eval", [sample(0)])], records=[direct(0)])
    write_collection_report(tmp_path, first, failure_policy="strict")
    ext = log("ext.eval", [sample(1), sample(2)])
    for exported in ([], [direct(1), direct(2)]):
        part = collection_part("extension", planned=None, logs=[ext], records=exported)
        write_collection_report(tmp_path, part, failure_policy="strict", append=True)
    report = json.loads((tmp_path / "collection_report.json").read_text())
    assert [part["kind"] for part in report["collections"]] == ["collection", "extension"]
    assert (report["logged"], report["exported"]) == (3, 3)


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
    from inspect_pipeline.coverage import evaluations_sha256

    report = {"schema_version": 2, "source": "inspect_logs", "planned": 2,
              "collections": [{"logs": [{"file": name} for name in names]}],
              "evaluations_sha256": evaluations_sha256(run / "evaluations.jsonl")}
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


def test_upload_rebuilds_report_when_manifest_gains_a_failed_log(tmp_path, monkeypatch):
    import inspect_ai.log
    from inspect_pipeline.coverage import collection_part, write_collection_report
    from scripts.upload_results import stage_run

    run = _run_folder(tmp_path, log_file="judge_a.eval", log_files=["judge_a.eval"])
    records = [direct(0, "a")]
    evaluations = run / "evaluations.jsonl"
    evaluations.write_text(json.dumps(records[0]) + "\n")
    before = evaluations.read_bytes()
    logs = {
        "judge_a.eval": log("judge_a.eval", [sample(0, judge_nick="a")]),
        "judge_b.eval": log("judge_b.eval", [sample(1, "provider failed", judge_nick="b")]),
    }
    part = collection_part("collection", planned=None, logs=[logs["judge_a.eval"]], records=records)
    write_collection_report(run, part, failure_policy="strict")

    manifest = json.loads((run / "inspect_run.json").read_text())
    manifest["log_files"].append("judge_b.eval")
    (run / "inspect_run.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(inspect_ai.log, "read_eval_log", lambda path: logs[Path(path).name])

    stage_run("demo", run, tmp_path / "staged")
    staged = tmp_path / "staged/runs/demo"
    report = json.loads((staged / "collection_report.json").read_text())
    assert evaluations.read_bytes() == before
    assert (report["logged"], report["exported"], report["failed"], report["omitted"]) == (2, 1, 1, 1)
    assert {entry["file"] for part in report["collections"] for entry in part["logs"]} == set(logs)
    assert all((staged / name).is_file() for name in logs)


def test_uploaded_spec_can_resolve_its_own_file(tmp_path):
    from scripts.upload_results import parse_spec
    path = tmp_path / "spec.py"
    path.write_text('from pathlib import Path\nRUN_SPEC = {"name": Path(__file__).parent.name}\n')
    assert parse_spec(path)["name"] == tmp_path.name
