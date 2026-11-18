"""Reusing a finished collection only when it is this plan's collection (issue #31).

A rerun used to accept evaluations.jsonl by its row count alone, so a same-size change
to the constitution, a scenario's text, the panel or the settings returned the old
judgments as if they were new. Each case here collects once, changes one thing and
expects the rerun to refuse with that thing named.
"""

from __future__ import annotations

import json

import pytest
from inspect_ai.model import GenerateConfig, ModelOutput, get_model

from test_inspect_collect import run_dir  # noqa: F401  (pytest fixture)

from inspect_pipeline.collect import collect_direct_ratings_inspect
from inspect_pipeline.provenance import FINGERPRINT_FILE, check_reusable, collection_identity


def _collect(spec_path):
    return collect_direct_ratings_inspect(str(spec_path))


def _rewrite(path, change):
    path.write_text(change(path.read_text(encoding="utf-8")), encoding="utf-8")


def test_unchanged_rerun_reuses_and_records_a_fingerprint(run_dir):
    tmp_path, write_spec = run_dir
    spec = write_spec()
    first = _collect(spec)
    fingerprint = json.loads((tmp_path / FINGERPRINT_FILE).read_text())
    assert fingerprint["identity"]["models"][0]["nick"] == "alpha"
    assert _collect(spec) == first


@pytest.mark.parametrize("phased", [False, True])
def test_unseeded_sampled_run_reuses_its_realized_plan(run_dir, phased, monkeypatch):
    from inspect_ai.log import read_eval_log
    import inspect_pipeline.collect as collect
    import inspect_pipeline.eigenbench as eigenbench

    calls = []
    build_assignments = eigenbench.build_direct_assignments

    def tracked_assignments(*args, **kwargs):
        result = build_assignments(*args, **kwargs)
        calls.append((kwargs["sampler_seed"], result))
        return result

    monkeypatch.setattr(eigenbench, "build_direct_assignments", tracked_assignments)
    monkeypatch.setattr(collect, "build_direct_assignments", tracked_assignments)
    tmp_path, write_spec = run_dir
    spec = write_spec(sampler_mode="balanced_unique_judge")
    with spec.open("a") as handle:
        handle.write('\nRUN_SPEC["collection"]["sampler_seed"] = None\n')
        handle.write(f'RUN_SPEC["collection"]["inspect"]["phased"] = {phased!r}\n')

    records = _collect(spec)
    assert len(records) == 9
    fingerprint_path = tmp_path / FINGERPRINT_FILE
    fingerprint = json.loads(fingerprint_path.read_text())
    seed = fingerprint.pop("resolved_sampler_seed")
    assert isinstance(seed, int)
    [(plan_seed, assignments)] = calls
    assert plan_seed == seed
    assert {(row["scenario_index"], row["judge"]["name"], row["evaluee"]["name"])
            for row in records} == {
        (assignment["scenario_index"], assignment["judge_nick"], nick)
        for assignment in assignments for nick in assignment["eval_nicks"]
    }
    assert fingerprint["identity"]["sampling"]["sampler_seed"] is None
    estimate = json.loads((tmp_path / "direct_call_estimate.json").read_text())
    assert estimate["sampler_seed"] == seed
    logs = sorted((tmp_path / "inspect_logs").glob("*.eval"))
    assert logs
    headers = [read_eval_log(log, header_only=True).eval for log in logs]
    judgments = [log for log in headers if log.task.startswith("eigenbench_judge_") or not phased]
    assert len(judgments) == (3 if phased else 1)
    for log in judgments:
        args = log.task_args_passed
        assert args["spec"] == str(spec)
        assert "context" not in args
        assert log.metadata["eigenbench"]["sampling"]["sampler_seed"] is None
        if not phased:
            assert args["resolved_sampler_seed"] == seed
    assert _collect(spec) == records
    assert [call_seed for call_seed, _ in calls] == [seed, seed]
    assert sorted((tmp_path / "inspect_logs").glob("*.eval")) == logs

    fingerprint_path.write_text(json.dumps(fingerprint))
    with pytest.raises(RuntimeError, match="resolved sampler seed was not recorded"):
        _collect(spec)


def test_same_count_constitution_change_is_refused(run_dir):
    tmp_path, write_spec = run_dir
    spec = write_spec()
    _collect(spec)
    constitution = tmp_path / "constitution.json"
    criteria = json.loads(constitution.read_text())
    constitution.write_text(json.dumps([c.replace("good", "kind") for c in criteria]))
    with pytest.raises(RuntimeError, match="constitution's criteria differ"):
        _collect(spec)


def test_changed_scenario_text_at_the_same_index_is_refused(run_dir):
    tmp_path, write_spec = run_dir
    spec = write_spec()
    _collect(spec)
    scenarios = tmp_path / "scenarios.json"
    texts = json.loads(scenarios.read_text())
    texts[1] = "A different question entirely."
    scenarios.write_text(json.dumps(texts))
    with pytest.raises(RuntimeError, match="scenario text changed at index 1"):
        _collect(spec)


def test_changed_generation_settings_are_refused(run_dir):
    _, write_spec = run_dir
    spec = write_spec()
    _collect(spec)
    _rewrite(spec, lambda text: text.replace(
        '"inspect": {', '"generation": {"response": {"temperature": 0.2}}, "inspect": {'))
    with pytest.raises(RuntimeError, match="generation settings changed: response"):
        _collect(spec)


def test_a_different_plan_of_the_same_size_is_refused(run_dir):
    _, write_spec = run_dir
    spec = write_spec(sampler_mode="balanced_unique_judge")
    _collect(spec)
    # Another seed picks other judges for the same number of edges.
    _rewrite(spec, lambda text: text.replace(
        '"sampler_mode":', '"sampler_seed": 7, "sampler_mode":'))
    with pytest.raises(RuntimeError, match="not in this plan|sampling settings changed"):
        _collect(spec)


def test_partial_output_is_refused_naming_the_missing_judgments(run_dir):
    tmp_path, write_spec = run_dir
    spec = write_spec()
    _collect(spec)
    evaluations = tmp_path / "evaluations.jsonl"
    lines = evaluations.read_text().splitlines()
    evaluations.write_text("\n".join(lines[:-2]) + "\n")
    with pytest.raises(RuntimeError, match="2 planned judgment\\(s\\) are missing"):
        _collect(spec)


def test_a_run_without_a_fingerprint_is_reused_with_a_warning(run_dir, capsys):
    tmp_path, write_spec = run_dir
    spec = write_spec()
    first = _collect(spec)
    (tmp_path / FINGERPRINT_FILE).unlink()
    capsys.readouterr()
    assert _collect(spec) == first
    assert "could not be verified" in capsys.readouterr().out


def test_omitted_judgments_count_toward_the_plan(run_dir):
    tmp_path, write_spec = run_dir
    spec = write_spec()
    _collect(spec)
    _rewrite(spec, lambda text: text.replace(
        '"inspect": {', '"failure_policy": "omit_invalid_judgments", "inspect": {'))
    evaluations = tmp_path / "evaluations.jsonl"
    rows = [json.loads(line) for line in evaluations.read_text().splitlines()]
    dropped = rows.pop()
    evaluations.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "omitted_samples.json").write_text(json.dumps({
        "planned": len(rows) + 1, "completed": len(rows), "omitted": 1,
        "samples": [{"scenario_index": dropped["scenario_index"], "judge": dropped["judge"]["name"],
                     "evaluee": dropped["evaluee"]["name"]}],
    }))
    assert len(_collect(spec)) == len(rows)


def _ctx(models):
    return {
        "models": models, "sampling": {"sampler_mode": "all_to_all"}, "include_self": True,
        "generation": {"response": {"max_tokens": 10, "temperature": 0.7}},
        "scale_min": 1, "scale_max": 10, "assignments": [], "selected": [], "criteria": [],
        "seed": {},
    }


def test_a_changed_model_reference_is_named(tmp_path):
    """Same nick, another model: only the fingerprint can tell."""

    evaluations = tmp_path / "evaluations.jsonl"
    before = collection_identity(_ctx({"a": "openai/gpt-4o", "b": "anthropic/claude-3.5-sonnet"}))
    (tmp_path / FINGERPRINT_FILE).write_text(json.dumps({"identity": json.loads(json.dumps(before, default=str))}))
    with pytest.raises(RuntimeError, match="model references changed: b"):
        check_reusable(evaluations, [], _ctx({"a": "openai/gpt-4o", "b": "anthropic/claude-sonnet-4"}), [])


@pytest.mark.parametrize("instance", [False, True], ids=["spec-string", "native-model"])
def test_changed_vllm_endpoint_refuses_saved_export(tmp_path, monkeypatch, instance):
    from inspect_pipeline.model_mapping import model_identity

    value = get_model("vllm/org/endpoint-test", memoize=False) if instance else "inspect:vllm/org/endpoint-test"
    ctx = _ctx({"a": value})
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    assert "vllm_endpoint_sha256" not in model_identity(value)
    monkeypatch.setenv("VLLM_BASE_URL", "http://user:secret@vm-a:8000/v1")
    previous = collection_identity(ctx)
    assert "secret" not in json.dumps(previous)
    (tmp_path / FINGERPRINT_FILE).write_text(json.dumps({"identity": previous}))
    monkeypatch.setenv("VLLM_BASE_URL", "http://vm-b:8000/v1")
    with pytest.raises(RuntimeError, match="model references changed: a"):
        check_reusable(tmp_path / "evaluations.jsonl", [], ctx, [])


@pytest.mark.parametrize("before, after", [
    ({"revision": "revision-1"}, {"revision": "revision-2"}),
    ({"base_url": "http://backend-a.invalid/v1"}, {"base_url": "http://backend-b.invalid/v1"}),
    ({"config": GenerateConfig(seed=1)}, {"config": GenerateConfig(seed=2)}),
], ids=["revision", "backend", "config"])
def test_changed_model_instance_settings_refuse_completed_export(run_dir, before, after):
    from pipeline.config import load_run_spec

    _, write_spec = run_dir
    spec = write_spec()
    original = load_run_spec(str(spec))[0]["models"]

    def panel(settings):
        return {
            nick: get_model(str(model), memoize=False, **{**model.model_args, **settings})
            for nick, model in original.items()
        }

    first = collect_direct_ratings_inspect(str(spec), models=panel(before))
    assert collect_direct_ratings_inspect(str(spec), models=panel(before)) == first
    with pytest.raises(RuntimeError, match="model references changed"):
        collect_direct_ratings_inspect(str(spec), models=panel(after))


def test_new_model_revision_does_not_reuse_old_generation_cache(run_dir, monkeypatch):
    tmp_path, write_spec = run_dir
    monkeypatch.setenv("INSPECT_CACHE_DIR", str(tmp_path / "inspect-cache"))
    spec = write_spec()
    _rewrite(spec, lambda text: text.replace('"cache": False', '"cache": True'))
    calls = []

    def panel(revision):
        def output(input, tools, tool_choice, config):
            calls.append(revision)
            system = input[0].text
            if system.startswith("Without making"):
                text = f"{revision}: {input[1].text}"
            elif system.startswith("You are acting as a judge. Carefully consider the following response"):
                text = "reflection"
            else:
                text = "".join(f"<criterion_{i}_rating>7</criterion_{i}_rating>" for i in range(1, 4))
            return ModelOutput.from_content(model="mockllm/model", content=text)

        return {"alpha": get_model("mockllm/model", revision=revision,
                                   custom_outputs=output, memoize=False)}

    first = collect_direct_ratings_inspect(str(spec), models=panel("revision-1"))
    evaluations = tmp_path / "evaluations.jsonl"
    evaluations.rename(tmp_path / "previous-evaluations.jsonl")
    second = collect_direct_ratings_inspect(str(spec), models=panel("revision-2"))

    assert len(first) == len(second) == 3
    assert calls.count("revision-1") == calls.count("revision-2") == 9
    assert all(row["response"].startswith("revision-2:") for row in second)


def test_model_identity_does_not_consume_scripted_output_generators():
    from inspect_pipeline.eigenbench import _model_resolver
    from inspect_pipeline.model_mapping import model_identity

    outputs = (ModelOutput.from_content(model="mockllm/model", content=str(i)) for i in range(2))
    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)

    model_identity(model)
    assert _model_resolver({"alpha": model})("alpha") is model

    assert [output.completion for output in outputs] == ["0", "1"]
