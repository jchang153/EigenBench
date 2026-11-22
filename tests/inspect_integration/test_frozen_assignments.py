"""A reviewed manifest drives the existing Inspect collector, not a sampler."""

import json

import pytest

from eigenbench.config import load_run_spec
from eigenbench.protocol.direct_rating import direct_input_sha256
from eigenbench.protocol.response_cache import load_cached_responses
from inspect_pipeline.collection.direct_rating import collect_direct_ratings_inspect
from inspect_pipeline.context import load_selection
from inspect_pipeline.eigenbench import eigenbench
from inspect_pipeline.provenance import FINGERPRINT_FILE
from scripts.run import estimate_calls
from tests.inspect_integration.test_inspect_collect import run_dir  # noqa: F401


@pytest.fixture
def frozen(run_dir):
    tmp, write_spec = run_dir
    path = write_spec(sampler_mode="explicit")
    with path.open("a") as spec:
        spec.write(
            '\nRUN_SPEC["collection"]["assignments_path"] = "assignments.json"\n'
            'RUN_SPEC["collection"]["cached_responses_path"] = '
            f'{str(tmp / "responses.jsonl")!r}\n'
            'RUN_SPEC["dataset"]["start"] = 1\n'
        )
    spec, run = load_run_spec(str(path))
    selected, criteria = load_selection(spec, run)
    plan = {
        "dataset_sha256": direct_input_sha256(selected),
        "criteria_sha256": direct_input_sha256(criteria),
        "judgments": [
            {"scenario_index": 3, "judge": "gamma", "evaluee": "alpha", "sampling_round": 0},
            {"scenario_index": 3, "judge": "beta", "evaluee": "alpha", "sampling_round": 1},
            {"scenario_index": 1, "judge": "alpha", "evaluee": "alpha", "sampling_round": 0},
        ],
    }
    manifest = tmp / "assignments.json"
    manifest.write_text(json.dumps(plan))
    return str(path), manifest, plan


@pytest.mark.parametrize("phased", [False, True])
def test_sparse_frozen_plan_estimates_collects_and_caches_exact_edges(frozen, phased):
    spec, manifest, plan = frozen
    with open(spec, "a") as handle:
        handle.write(f'\nRUN_SPEC["collection"]["inspect"]["phased"] = {phased!r}\n')
    cache = manifest.with_name("responses.jsonl")
    cache.write_text(json.dumps({"scenario_index": 3, "responses": {"gamma": "already cached"}}) + "\n")
    rows = plan["judgments"]
    expected = {(r["scenario_index"], r["judge"], r["evaluee"], r["sampling_round"]) for r in rows}
    task = eigenbench(spec, panel_metrics=False)
    assert {tuple(s.metadata[k] for k in ("scenario_index", "judge_nick", "eval_nick", "sampling_round"))
            for s in task.dataset} == expected
    estimate = estimate_calls(spec)
    assert estimate["num_scenarios"] == 2
    assert (estimate["response_tasks"], estimate["rating_tasks"], estimate["total_logical_generations"]) == (
        2, 3, 8
    )
    output = collect_direct_ratings_inspect(spec, scoring=False)
    assert {(r["scenario_index"], r["judge"]["name"], r["evaluee"]["name"], r["sampling"]["round"])
            for r in output} == expected
    assert len({r["response"] for r in output if r["scenario_index"] == 3}) == 1
    cached = load_cached_responses(str(cache), merge_rows=False)
    assert {index: set(responses) for index, responses in cached.items()} == {
        1: {"alpha"}, 3: {"alpha", "gamma"}
    }
    assert estimate_calls(spec)["response_tasks"] == 0
    assert collect_direct_ratings_inspect(spec, scoring=False) == output
    rows.reverse()
    manifest.write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match="judgment plan changed"):
        collect_direct_ratings_inspect(spec, scoring=False)
    manifest.with_name(FINGERPRINT_FILE).unlink()
    with pytest.raises(RuntimeError, match="judgment plan cannot be verified"):
        collect_direct_ratings_inspect(spec, scoring=False)


@pytest.mark.parametrize(("mutation", "message"), [
    ({"scenario_index": 0}, "selected scenario"),
    ({"scenario_index": True}, "selected scenario"),
    ({"judge": "missing"}, "model panel"),
    ({"sampling_round": -1}, "non-negative integer"),
])
def test_malformed_assignment_fails_preflight(frozen, mutation, message):
    spec, manifest, plan = frozen
    plan["judgments"][0].update(mutation)
    manifest.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=message):
        estimate_calls(spec)


def test_duplicate_edge_fails_preflight(frozen):
    spec, manifest, plan = frozen
    plan["judgments"].append({**plan["judgments"][0], "sampling_round": 2})
    manifest.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="duplicate judgment"):
        estimate_calls(spec)


@pytest.mark.parametrize(("file", "before", "field"), [
    ("scenarios.json", "what do you do?", "dataset_sha256"),
    ("constitution.json", "prefer the response", "criteria_sha256"),
])
def test_changed_input_fails_preflight(frozen, file, before, field):
    spec, manifest, _ = frozen
    source = manifest.with_name(file)
    source.write_text(source.read_text().replace(before, "changed"))
    with pytest.raises(ValueError, match=field):
        estimate_calls(spec)


def test_no_self_plan_rejects_self_judgment(frozen):
    spec, _, _ = frozen
    with open(spec, "a") as handle:
        handle.write('\nRUN_SPEC["evaluation"]["direct_rating"] = {"include_self": False}\n')
    with pytest.raises(ValueError, match="self-judging is disabled"):
        estimate_calls(spec)
