"""Prompt, sampling and cache compatibility with the original collectors."""

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "inspect_migration"


def test_shared_prompts_and_direct_plans_match_pre_migration_fixture():
    from pipeline.protocol import direct_rating as direct
    from eigenbench.protocol.pairwise import (
        build_comparison_prompt, build_comparison_user,
        build_reflection_prompt, build_reflection_user,
    )

    frozen = json.loads((FIXTURES / "protocol.json").read_text())
    assert {
        "direct_reflection_system": direct.build_direct_reflection_prompt("PREFIX"),
        "direct_reflection_user": direct.build_direct_reflection_user_prompt("C", "S", "R"),
        "direct_rating_system": direct.build_direct_rating_prompt("PREFIX"),
        "direct_rating_user": direct.build_direct_rating_user_prompt("C", "S", "R", "F"),
        "pairwise_reflection_system": build_reflection_prompt("PREFIX"),
        "pairwise_comparison_system_ties": build_comparison_prompt(True, "PREFIX"),
        "pairwise_comparison_system_no_ties": build_comparison_prompt(False, "PREFIX"),
        "pairwise_reflection_user": build_reflection_user("C", "S", "R"),
        "pairwise_comparison_user": build_comparison_user("C", "S", "R1", "F1", "R2", "F2"),
    } == frozen["prompts"]
    for mode, assignments in frozen["assignments"].items():
        assert direct.build_direct_assignments(
            [(i, f"Scenario {i}") for i in range(3)], dict.fromkeys(["a", "b", "c"], "mock"),
            sampler_mode=mode, group_size=2, response_redundancy=2, sampler_seed=17,
        ) == assignments


def test_shared_cache_reader_preserves_both_historical_row_policies(tmp_path):
    from pipeline.protocol.response_cache import load_cached_responses, count_cached_responses

    path = tmp_path / "responses.jsonl"
    path.write_text('\n'.join(json.dumps(row) for row in [
        {"scenario_index": 3, "responses": {"a": "first", "b": "kept"}},
        {"scenario_index": 3, "responses": {"a": "last", "invalid": None}},
    ]) + '\n')
    assert load_cached_responses(str(path)) == {3: {"a": "last", "b": "kept"}}
    assert load_cached_responses(str(path), merge_rows=False) == {3: {"a": "last"}}
    assert count_cached_responses(str(path), scenario_indices={3}, model_nicks={"a", "b"},
                                  openrouter_nicks={"a"}) == (2, 1)
