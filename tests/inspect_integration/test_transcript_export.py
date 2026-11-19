import pytest
from inspect_ai.log import EvalLog, EvalSpec

from inspect_pipeline.export import export_log, write_evaluations_atomic
from eigenbench.records import append_records, load_records, save_records


@pytest.mark.parametrize("suffix", [".JSONL", ".json"])
def test_export_uses_transcript_format_and_keeps_original_on_error(tmp_path, suffix):
    records = [{"scenario": "café"}, {"answer": "a\nb"}]
    transcript = tmp_path / ("source" + suffix)
    output = tmp_path / ("output" + suffix)
    save_records(transcript, iter(records[:1]))
    append_records(transcript, iter(records[1:]))
    write_evaluations_atomic(output, records)
    assert output.read_bytes() == transcript.read_bytes()
    assert load_records(output) == records
    with pytest.raises(TypeError):
        write_evaluations_atomic(output, [{"answer": object()}])
    assert output.read_bytes() == transcript.read_bytes()
    assert list(tmp_path.glob(output.name + ".*.tmp")) == []


@pytest.mark.parametrize("recorded_path", [None, ""])
def test_export_without_destination_refuses_before_writing(tmp_path, monkeypatch, recorded_path):
    monkeypatch.chdir(tmp_path)
    spec = EvalSpec(created="2026-11-18", task="eigenbench", task_id="test", model="mockllm/model", dataset={}, config={},
                    metadata={"eigenbench": {"evaluations_path": recorded_path}})
    with pytest.raises(ValueError, match="no evaluations path"):
        export_log(EvalLog(eval=spec))
    assert list(tmp_path.iterdir()) == []
