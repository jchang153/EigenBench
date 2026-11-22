"""Contract for primary-only pair-balanced direct judging."""

from collections import Counter
import json

import pytest

from eigenbench.config.run_spec import DIRECT_SAMPLER_PAIR_BALANCED as MODE
from eigenbench.config.run_spec import apply_run_defaults, normalize_direct_sampling_settings
from eigenbench.protocol.direct_rating import (
    build_direct_assignments,
    direct_input_sha256,
    estimate_direct_calls,
    load_direct_assignments,
    validate_direct_assignments,
)


def plan(n=9, prompts=50, *, criteria=("safety", "agency"), seed=31, **settings):
    selected = [(2 * i + 1, f"prompt {i}") for i in range(prompts)]
    models = {f"m{i}": None for i in range(n)}
    return build_direct_assignments(
        selected, models, sampler_mode=MODE, criteria=list(criteria),
        sampler_seed=seed, **settings,
    )


@pytest.mark.parametrize("n,prompts", [(2, 0), (2, 1), (5, 3), (5, 20), (9, 50)])
@pytest.mark.parametrize("include_self", [True, False])
def test_one_matching_per_prompt_with_floor_or_ceil_pair_counts(n, prompts, include_self):
    rows = plan(n, prompts, include_self=include_self)
    assert len(rows) == n * prompts
    counts = Counter()
    for p in range(prompts):
        matching = rows[p * n:(p + 1) * n]
        assert sorted(row["judge_idx"] for row in matching) == list(range(n))
        for a, row in enumerate(matching):
            j = row["judge_idx"]
            assert row == {
                "scenario_index": 2 * p + 1, "scenario": f"prompt {p}",
                "judge_idx": j, "judge_nick": f"m{j}", "eval_idxs": [a], "eval_nicks": [f"m{a}"],
                "sampler_mode": MODE, "sampling_round": 0, "group_index": a,
            }
            assert include_self or a != j
            counts[a, j] += 1
    degree = n if include_self else n - 1
    base, extras = divmod(prompts, degree)
    for a in range(n):
        eligible = [counts[a, j] for j in range(n) if include_self or a != j]
        assert sorted(eligible) == [base] * (degree - extras) + [base + 1] * extras


def test_seeded_draws_are_rubric_specific_and_not_fixed_rotations():
    rows = plan()
    assert rows == plan()
    assert rows != plan(seed=32)
    assert rows != plan(criteria=("agency", "safety"))
    assert rows != plan(criteria=("safety", "honesty"))
    # Even a cyclic schedule on shuffled model labels has at most n distinct matchings.
    matchings = {tuple(r["judge_idx"] for r in rows[i:i + 9]) for i in range(0, len(rows), 9)}
    assert len(matchings) > 9


def test_sampler_defaults_remain_opt_in():
    defaults = {"sampler_mode": "all_to_all", "group_size": 4,
                "response_redundancy": 1, "sampler_seed": 42}
    assert normalize_direct_sampling_settings({}) == defaults
    for mode in ("balanced_unique_judge", "partitioned_random_judge"):
        assert normalize_direct_sampling_settings({"sampler_mode": mode}) == {
            **defaults, "sampler_mode": mode,
        }
    assert normalize_direct_sampling_settings({"sampler_mode": MODE}) == {
        **defaults, "sampler_mode": MODE, "group_size": 1,
    }


@pytest.mark.parametrize("include_self", [True, False])
def test_estimate_equals_materialized_response_and_judge_calls(include_self):
    rows = plan(include_self=include_self)
    providers = ({0, 1, 3}, {5, 6}, {2, 4, 7, 8})
    cached = (20, 7, 4)
    args = dict(
        num_models=9, num_scenarios=50, sampler_mode=MODE, sampler_seed=31,
        include_self=include_self, num_openrouter_models=3, cached_responses=sum(cached),
        cached_openrouter_responses=cached[0], cached_other_responses=cached[1],
        other_model_indices=providers[1],
    )
    estimate = estimate_direct_calls(**args)
    assert estimate == estimate_direct_calls(**args, assignments=rows, openrouter_model_indices=providers[0])
    assert estimate["response_tasks"] == len(rows) - sum(cached)
    assert estimate["reflection_tasks"] == estimate["rating_tasks"] == len(rows) == 450
    assert estimate["total_logical_generations"] == 1350 - sum(cached)
    assert estimate["directed_edges_per_scenario"] == 9
    for indices, hits, key in zip(providers, cached, (
        "openrouter_requests", "other_logical_generations", "local_logical_generations",
    )):
        answers = sum(row["eval_idxs"][0] in indices for row in rows)
        judgments = sum(row["judge_idx"] in indices for row in rows)
        assert estimate[key] == answers - hits + 2 * judgments


@pytest.mark.parametrize("field", ["response_redundancy", "group_size"])
@pytest.mark.parametrize("value", [0, 2, 1.5])
def test_reject_unsupported_settings_in_spec_planner_and_estimate(field, value):
    settings = {field: value}
    spec = {"evaluation": {"mode": "direct"}, "collection": {"sampler_mode": MODE, **settings}}
    with pytest.raises(ValueError, match=field):
        apply_run_defaults("spec", __file__, spec)
    with pytest.raises(ValueError, match=field):
        plan(**settings)
    with pytest.raises(ValueError, match=field):
        estimate_direct_calls(num_models=9, num_scenarios=50, sampler_mode=MODE, **settings)


def test_singleton_panel_and_invalid_plans():
    assert [r["judge_idx"] for r in plan(1, 3)] == [0, 0, 0]
    with pytest.raises(ValueError, match="eligible judges"):
        plan(1, 3, include_self=False)
    with pytest.raises(ValueError, match="criteria"):
        plan(criteria=())
    with pytest.raises(ValueError, match="assignments_path"):
        plan(assignments_path="manifest.json")
    spec = {"evaluation": {"mode": "direct"}, "collection": {"sampler_mode": MODE},
            "extension": {"from_evaluations": "prior.jsonl"}}
    with pytest.raises(ValueError, match="extension"):
        apply_run_defaults("spec", __file__, spec)


def test_in_memory_manifest_validation_never_reopens_path(tmp_path):
    path = tmp_path / "plan.json"
    selected, criteria, models = [(7, "scenario")], ["rubric"], {"a": None, "b": None}
    manifest = {
        "dataset_sha256": direct_input_sha256(selected), "criteria_sha256": direct_input_sha256(criteria),
        "judgments": [{"scenario_index": 7, "judge": "a", "evaluee": "b", "sampling_round": 0}],
    }
    path.write_text(json.dumps(manifest))
    expected = load_direct_assignments(str(path), selected, criteria, models, include_self=False)
    path.unlink()
    assert validate_direct_assignments(
        manifest, selected, criteria, models, include_self=False, path=path,
    ) == expected
    with pytest.raises(ValueError, match="plan.json: criteria_sha256"):
        validate_direct_assignments(manifest, selected, ["other"], models, include_self=False, path=path)
