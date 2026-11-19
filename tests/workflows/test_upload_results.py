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


def test_pairwise_elo_matches_the_paper_and_native_analysis(tmp_path):
    import math
    from pipeline.analysis.fit import eigentrust_to_elo

    run = tmp_path / "run"
    analysis = run / "btd_d2"
    analysis.mkdir(parents=True)
    _write_spec(run, "pairwise_btd", models={"a": "a", "b": "b", "c": "c"})
    scores = [1e-15, 0.25, 0.75 - 1e-15]
    (analysis / "eigentrust.txt").write_text(f"EigenTrust scores:\n{scores}\n")
    expected = [1500 + 400 * math.log10(3 * t) for t in scores]
    _, summary = stage_run("demo", run, tmp_path / "staged")
    published = sorted(summary, key=lambda row: row["model_index"])
    assert [row["elo_mean"] for row in published] == eigentrust_to_elo(scores, 3).tolist()
    assert [row["elo_mean"] for row in published] == pytest.approx(expected)


@pytest.mark.parametrize(("scores", "message"), [
    ("[0, 1]", "positive trust"),
    ("[-0.1, 1.1]", "nonnegative"),
    ("[nan, 1]", "finite"),
    ("[1]", "per model"),
    ("[0.75, 0.75]", "sum to one"),
])
@pytest.mark.parametrize("has_bootstrap", [False, True])
def test_pairwise_publication_refuses_invalid_trust(tmp_path, scores, message, has_bootstrap):
    run = tmp_path / "run"
    analysis = run / "btd_d2"
    analysis.mkdir(parents=True)
    _write_spec(run, "pairwise_btd", models={"a": "a", "b": "b"},
                training={"bootstrap": {"enabled": has_bootstrap}})
    if has_bootstrap:
        (analysis / "bootstrap").mkdir()
        (analysis / "bootstrap/summary.json").write_text('[{"elo_mean": 1500}]')
    (analysis / "eigentrust.txt").write_text(f"EigenTrust scores:\n{scores}\n")
    with pytest.raises(ValueError, match=message):
        stage_run("demo", run, tmp_path / "staged")


def test_direct_point_summary_preserves_native_zero_trust_elo(tmp_path):
    import math
    run = tmp_path / "run"
    analysis = run / "direct_rating"
    analysis.mkdir(parents=True)
    _write_spec(run, models={"a": "a", "b": "b"})
    (analysis / "eigentrust.txt").write_text("EigenTrust scores:\n[0. 1.]\n")
    _, summary = stage_run("demo", run, tmp_path / "staged")
    assert summary[-1]["elo_mean"] == pytest.approx(1500 + 400 * math.log10(2e-12))


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


def test_stage_rebuilds_report_only_for_matching_judgment_content(tmp_path):
    from inspect_pipeline.coverage import evaluations_sha256
    from inspect_pipeline.collection.pairwise import collect_pairwise_inspect
    from tests.inspect_integration.test_inspect_pairwise import _panel, _write_spec as write_pairwise_spec

    run = tmp_path / "run"
    logs = tmp_path / "logs"
    spec = Path(write_pairwise_spec(run, groups=1, inspect={
        "cache": False, "display": "none", "log_dir": str(logs),
    }))
    rows = collect_pairwise_inspect(str(spec), models=_panel([]), scoring=False)
    log_file = json.loads((run / "inspect_run.json").read_text())["log_file"]
    (run / "btd_d2").mkdir()
    (run / "collection_report.json").unlink()
    with spec.open("a") as handle:
        handle.write('\nRUN_SPEC["collection"]["evaluations_path"] = "judgments.json"\n')

    # A legacy manifest has log names but no digest linking them to an export.
    (run / "inspect_run.json").write_text(json.dumps({"log_file": log_file}))
    (run / "judgments.json").write_text(json.dumps(rows))
    stage_run("matching", run, tmp_path / "staged")
    dest = tmp_path / "staged/runs/matching"
    report = json.loads((dest / "collection_report.json").read_text())
    assert (report["logged"], report["exported"], report["omitted"]) == (len(rows), len(rows), 0)
    assert report["evaluations_sha256"] == evaluations_sha256(dest / "evaluations.jsonl")
    assert (dest / log_file).is_file()

    # Same keys, count, and sibling directory, but a different benchmark judgment.
    rows[0]["judge response"] = "different verdict"
    (run / "judgments.json").write_text(json.dumps(rows))
    meta, _ = stage_run("different", run, tmp_path / "staged")
    dest = tmp_path / "staged/runs/different"
    assert "inspect" not in meta
    assert not (dest / log_file).exists()
    assert not (dest / "collection_report.json").exists()
    assert json.loads((dest / "evaluations.jsonl").read_text().splitlines()[0]) == rows[0]


@pytest.mark.parametrize("batch", [False, True])
def test_catalogue_replaces_runs_against_the_revision_it_read(tmp_path, monkeypatch, batch):
    from types import SimpleNamespace

    import huggingface_hub
    from scripts import upload_results as upload

    revision = "a" * 40
    published = {}
    slug = "group/run" if batch else "run"
    current = {"runs": [{"slug": "keep", "timestamp": "0"}, {"slug": slug, "timestamp": "1"}]}
    source = tmp_path / "old-index.json"
    source.write_text(json.dumps(current))
    def download(*, revision, token, **kwargs):
        assert revision == "a" * 40 and token == "test-token"
        return str(source)

    class Api:
        token = "test-token"

        def upload_folder(self, **kwargs):
            return SimpleNamespace(oid=revision)

        def dataset_info(self, repo_id):
            assert batch  # Single publication uses its own artifact commit.
            return SimpleNamespace(sha=revision)

        def upload_file(self, *, path_or_fileobj, parent_commit, **kwargs):
            assert parent_commit == revision
            published.update(json.loads(path_or_fileobj))

        def create_commit(self, *, operations, parent_commit, **kwargs):
            assert parent_commit == revision
            index = next(op for op in operations if op.path_in_repo == "index.json")
            published.update(json.loads(index.path_or_fileobj))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: Api())
    monkeypatch.setattr(upload, "stage_run", lambda *args: (_meta({}), []))
    base = tmp_path / "group"
    run = base / "run"
    run.mkdir(parents=True)
    (run / "spec.py").touch()
    if batch:
        upload.upload_batch(base, "group", "org/runs")
    else:
        upload.upload_run(slug, run, "org/runs")
    assert [item["slug"] for item in published["runs"]] == [slug, "keep"]


def test_catalogue_update_never_writes_a_stale_cached_index(tmp_path, monkeypatch):
    """F23: after a timed-out Hub read the client can return a stale cached index.json."""
    import huggingface_hub
    from scripts import upload_results

    stale = tmp_path / "index.json"
    stale.write_text(json.dumps({"last_updated": None, "runs": [{"slug": "old-a", "timestamp": "1"}]}))

    def download(repo_id, filename, repo_type=None, force_download=False, **kwargs):
        if force_download:
            raise TimeoutError("read timed out")
        return str(stale)  # what a cached read returns when the fresh check fails

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    written = []
    api = SimpleNamespace(token=None, upload_file=lambda path_or_fileobj, **kwargs: written.append(path_or_fileobj))
    with pytest.raises(TimeoutError):
        upload_results.update_index("new", "new", {}, [], "org/dataset", api, revision="snapshot")
    assert written == []


def test_catalogue_starts_only_when_the_index_is_missing(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError
    from scripts import upload_results

    def missing(repo_id, filename, **kwargs):
        raise EntryNotFoundError("index.json not found")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    assert upload_results._load_index("org/dataset", revision="snapshot", token=None) == {"last_updated": None, "runs": []}


def test_staging_reads_only_the_configured_output_dir(tmp_path):
    """F22: an older run/btd_d8 must not stand in for training.output_dir results."""
    from scripts.upload_results import find_btd_dir

    (tmp_path / "btd_d8").mkdir()
    configured = {"training": {"dims": [8], "output_dir": "fresh"}}
    assert find_btd_dir(tmp_path, configured) is None
    (tmp_path / "fresh" / "btd_d8").mkdir(parents=True)
    assert find_btd_dir(tmp_path, configured) == tmp_path / "fresh" / "btd_d8"
    assert find_btd_dir(tmp_path, {"training": {"dims": [8]}}) == tmp_path / "btd_d8"
