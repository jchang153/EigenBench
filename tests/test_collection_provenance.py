"""Reusing a finished collection only when it is this plan's collection (issue #31).

A rerun used to accept evaluations.jsonl by its row count alone, so a same-size change
to the constitution, a scenario's text, the panel or the settings returned the old
judgments as if they were new. Each case here collects once, changes one thing and
expects the rerun to refuse with that thing named.
"""

from __future__ import annotations

import json

import pytest

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
    }


def test_a_changed_model_reference_is_named(tmp_path):
    """Same nick, another model: only the fingerprint can tell."""

    evaluations = tmp_path / "evaluations.jsonl"
    before = collection_identity(_ctx({"a": "openai/gpt-4o", "b": "anthropic/claude-3.5-sonnet"}))
    (tmp_path / FINGERPRINT_FILE).write_text(json.dumps({"identity": json.loads(json.dumps(before, default=str))}))
    with pytest.raises(RuntimeError, match="model references changed: b"):
        check_reusable(evaluations, [], _ctx({"a": "openai/gpt-4o", "b": "anthropic/claude-sonnet-4"}), [])
