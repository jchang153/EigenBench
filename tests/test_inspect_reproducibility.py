"""Reproduction metadata and public-copy checks using real offline Inspect logs."""
import hashlib
import json
from pathlib import Path
import zipfile

from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ModelCall, ModelOutput, get_model
import pytest

from inspect_pipeline.eigenbench import eigenbench
from inspect_pipeline.export import records_from_log
from scripts.prepare_inspect_log import prepare_log


def test_default_task_records_pinned_assets_and_resolved_settings():
    from pipeline.config.airisk import DATASET_ID, DATASET_REVISION
    task = eigenbench(cache=False)
    metadata = task.metadata["eigenbench"]
    assert metadata["dataset"]["source"] == DATASET_ID
    assert metadata["dataset"]["revision"] == DATASET_REVISION
    assert len(task.dataset) == 1600
    assert len(metadata["panel"]) == 4
    assert all(member["name"].startswith("openrouter/") for member in metadata["panel"])
    assert metadata["generation"]["response"]["max_tokens"] == 4096
    assert metadata["sampling"]["sampler_mode"] == "all_to_all"
    assert metadata["evaluation"]["direct_rating"]["include_self"] is True
    assert metadata["criteria_sha256"] == hashlib.sha256(json.dumps(metadata["criteria"]).encode()).hexdigest()


@pytest.fixture
def completed_log(tmp_path):
    (tmp_path / "scenarios.json").write_text('["Which response is kinder?"]')
    (tmp_path / "constitution.json").write_text('["Be kind."]')
    spec = tmp_path / "spec.py"
    spec.write_text('''RUN_SPEC = {
"models": {"a": "mock", "b": "mock"}, "evaluation": {"mode": "direct_rating"},
"dataset": {"path": "scenarios.json"}, "constitution": {"path": "constitution.json", "num_criteria": 1},
"collection": {"inspect": {"cache": False}, "sampler_mode": "all_to_all"},
"training": {"enabled": False}}
''')
    def output(input, tools, tool_choice, config):
        return ModelOutput.from_content(model="mockllm/model", content="<criterion_1_rating>7</criterion_1_rating>")
    panel = {nick: get_model("mockllm/model", custom_outputs=output, memoize=False) for nick in ("a", "b")}
    [log] = inspect_eval(eigenbench(str(spec), models=panel), model=None, display="none",
                         metadata={"source_commit": "f21e599302b42ae476e32ed6539a34bf49f5c604"},
                         log_dir=str(tmp_path / "logs"))
    assert log.status == "success"
    return log


def test_public_copy_preserves_records_and_metrics_and_scrubs_all_archive_members(tmp_path, completed_log, monkeypatch):
    secret = "TEST-PRIVATE-KEY-246813579"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    completed_log.eval.model_args = {"headers": {"Authorization": f"Bearer {secret}"},
                                     "refresh_token": "SYNTHETIC-REFRESH-9147"}
    completed_log.eval.model_base_url = "https://user:pass@example.com/v1?token=abc"
    completed_log.eval.metadata["private_note"] = "machine-owner"
    completed_log.eval.metadata["local_file"] = str(Path.home() / "private-project" / "run.json")
    for sample in completed_log.samples:
        for event in sample.events:
            if event.event == "model":
                event.call = ModelCall(request={"private": secret * 200})
    source = tmp_path / "private.eval"
    write_eval_log(completed_log, source)
    original_bytes = source.read_bytes()
    target = prepare_log(source, tmp_path / "public.eval", redact=("machine-owner",))
    public = read_eval_log(target)
    assert records_from_log(public) == records_from_log(completed_log)
    assert [(s.id, s.epoch) for s in public.samples] == [(s.id, s.epoch) for s in completed_log.samples]
    assert public.results.scores[1].metrics == completed_log.results.scores[1].metrics
    assert public.eval.metadata["source_commit"] == completed_log.eval.metadata["source_commit"]
    assert public.eval.metadata["local_file"] == "[local-path]"
    assert public.eval.metadata["eigenbench_public_copy"]["benchmark_content_preserved"] is True
    assert public.eval.model_base_url == "https://example.com/v1?token=[redacted]"
    with zipfile.ZipFile(target) as archive:
        for member in archive.namelist():
            raw = archive.read(member)
            for private in (str(tmp_path), "machine-owner", secret, "user:pass", "SYNTHETIC-REFRESH-9147"):
                assert private.encode() not in raw
    assert source.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="new destination"):
        prepare_log(source, target)


def test_public_copy_refuses_benchmark_edits_and_partial_evidence(tmp_path, completed_log):
    source = tmp_path / "source.eval"
    write_eval_log(completed_log, source)
    with pytest.raises(ValueError, match="change benchmark evidence"):
        prepare_log(source, tmp_path / "public.eval", redact=("kinder",))
    completed_log.results.scores[1].metrics["coverage/scored"].value = 3
    write_eval_log(completed_log, source)
    with pytest.raises(ValueError, match="complete planned coverage"):
        prepare_log(source, tmp_path / "partial.eval")
    assert not (tmp_path / "public.eval").exists()
    assert not (tmp_path / "partial.eval").exists()


@pytest.mark.parametrize("surface", ["panel", "eval", "event"])
def test_public_copy_refuses_model_identity_edits(tmp_path, completed_log, surface):
    name = "mockllm/private-target"
    if surface == "panel":
        completed_log.eval.metadata["eigenbench"]["panel"][0]["name"] = name
    elif surface == "eval":
        completed_log.eval.model = name
    else:
        event = next(e for e in completed_log.samples[0].events if e.event == "model")
        event.model = name
    source = tmp_path / "source.eval"
    write_eval_log(completed_log, source)
    with pytest.raises(ValueError, match="change benchmark evidence"):
        prepare_log(source, tmp_path / "public.eval", redact=("private-target",))


def test_public_copy_refuses_private_paths_in_dictionary_keys(tmp_path, completed_log):
    completed_log.eval.metadata["diagnostics"] = {"/home/another-owner/run": "ok"}
    source = tmp_path / "source.eval"
    write_eval_log(completed_log, source)
    with pytest.raises(ValueError, match="dictionary key"):
        prepare_log(source, tmp_path / "public.eval")
