"""Focused regressions for this change."""
import json
import sys
import pytest

def test_airisk_id_downloads_once_and_selects_unique_questions(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    from pipeline.config import datasets as loaders
    from pipeline.config.airisk import DATASET_ID, DATASET_REVISION
    from scripts import prepare_airiskdilemmas

    source = tmp_path / "model_eval.jsonl"
    # One question appears in two separate action pairs as well.
    rows = [{"dilemma": q, "action": action} for q in ("A", "B", "A", "C") for action in ("yes", "no")]
    source.write_text("\n".join(json.dumps(row) for row in rows))
    cached = False
    calls = []

    def download(**kwargs):
        nonlocal cached
        calls.append(kwargs)
        assert kwargs["repo_id"] == DATASET_ID
        assert kwargs["revision"] == DATASET_REVISION
        assert kwargs["filename"] == "model_eval.jsonl"
        assert kwargs["repo_type"] == "dataset"
        if kwargs.get("local_files_only") and not cached:
            raise LocalEntryNotFoundError("not cached")
        cached = True
        return str(source)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(loaders, "_REPO_ROOT", tmp_path)
    legacy = tmp_path / "data/scenarios/airiskdilemmas.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps(rows))
    scenarios = loaders.load_dataset_scenarios_from_spec({"id": "airisk", "start": 1, "count": 2})
    assert scenarios == ["A", "B", "C"]
    assert loaders.select_scenarios(scenarios, start=1, count=2) == [(1, "B"), (2, "C")]
    assert loaders.load_dataset_scenarios_from_spec("airisk") == scenarios
    assert [call.get("local_files_only", False) for call in calls] == [True, False, True]
    assert json.loads(legacy.read_text()) == rows

    # The optional CLI uses exactly the same preparation as the automatic ID.
    output = tmp_path / "prepared.json"
    monkeypatch.setattr(sys, "argv", ["prepare_airiskdilemmas.py", "--output", str(output)])
    prepare_airiskdilemmas.main()
    assert json.loads(output.read_text()) == scenarios
    assert calls[-1]["local_files_only"] is True


@pytest.mark.parametrize("rows, message", [
    ([{"dilemma": "A"}], "unpaired"),
    ([{"dilemma": "A"}, {"dilemma": "B"}], "different dilemmas"),
    ([{"dilemma": ""}, {"dilemma": ""}], "empty"),
])
def test_airisk_rejects_malformed_action_pairs(rows, message):
    from pipeline.config.airisk import paired_dilemmas

    with pytest.raises(ValueError, match=message):
        paired_dilemmas(rows)


def test_airisk_does_not_redownload_malformed_cached_data(tmp_path, monkeypatch):
    import huggingface_hub
    from pipeline.config.airisk import load_airisk_scenarios

    source = tmp_path / "model_eval.jsonl"
    source.write_text('{"dilemma": "unpaired"}\n')
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(source)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(ValueError, match="unpaired"):
        load_airisk_scenarios()
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
