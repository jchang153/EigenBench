"""Comparison cleaning, EigenTrust damping and the BT route."""

import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def planned_rows(prefer):
    from inspect_pipeline.pairwise_plan import pairwise_comparisons, plan_pairwise_groups

    groups = plan_pairwise_groups([(0, "s0")], ["a", "b"], sampler_mode="uniform",
                                  group_size=2, groups=3, sampler_seed=42)
    plan = pairwise_comparisons(groups)
    rows = [[0, 0, p["judge_idx"], p["eval1_idx"], p["eval2_idx"], prefer(p)] for p in plan]
    return rows, [p["group_id"] for p in plan]


def test_repeated_rounds_of_one_pair_are_all_kept():
    from pipeline.utils.comparisons import handle_inconsistencies_with_ties_criteria

    rows, groups = planned_rows(lambda p: 1 if p["eval1_idx"] == 0 else 2)
    assert len(rows) == 6
    for given in (groups, None):
        cleaned = handle_inconsistencies_with_ties_criteria(rows, given)
        assert Counter(map(tuple, cleaned)) == Counter(map(tuple, rows))


def test_strict_inconsistency_ties_only_its_own_round():
    from pipeline.utils.comparisons import handle_inconsistencies_with_ties_criteria

    rows, groups = planned_rows(lambda p: 1 if p["eval1_idx"] == 0 else 2)
    first = groups[0]
    for row, group in zip(rows, groups):
        if group == first:
            row[-1] = 1
    cleaned = handle_inconsistencies_with_ties_criteria(rows, groups)
    assert [row[-1] for row in cleaned].count(0) == 2
    assert len(cleaned) == 6


def test_same_order_repeats_are_not_turned_into_ties():
    from pipeline.utils.comparisons import handle_inconsistencies_with_ties_criteria

    rows = [[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]]
    assert handle_inconsistencies_with_ties_criteria(rows) == rows


def test_order_rule_matches_the_paper_for_one_pair():
    from pipeline.utils.comparisons import handle_inconsistencies_with_ties_criteria

    def clean(first, second):
        return [row[-1] for row in handle_inconsistencies_with_ties_criteria(
            [[0, 0, 0, 1, 2, first], [0, 0, 0, 2, 1, second]])]

    assert clean(1, 2) == [1, 2]
    assert clean(1, 1) == [0, 0]
    assert clean(2, 2) == [0, 0]
    assert clean(1, 0) == [1, 0]
    assert clean(0, 0) == [0, 0]


def test_gaps_in_model_ids_keep_their_comparisons():
    from pipeline.utils.comparisons import handle_inconsistencies_with_ties_criteria

    rows = [[0, 5, 0, 0, 2, 1], [0, 5, 0, 2, 0, 2]]
    assert handle_inconsistencies_with_ties_criteria(rows) == rows


def test_exported_pairwise_records_carry_their_group():
    from inspect_pipeline.pairwise import (STORE_EVAL1_REFLECTION, STORE_EVAL1_RESPONSE,
                                           STORE_EVAL2_REFLECTION, STORE_EVAL2_RESPONSE,
                                           STORE_JUDGE_RESPONSE, pairwise_record)
    from pipeline.utils.comparisons import extract_comparisons_with_ties_criteria

    md = {"scenario": "s", "scenario_index": 0, "eval1_idx": 0, "eval1_nick": "a",
          "eval2_idx": 1, "eval2_nick": "b", "judge_idx": 0, "judge_nick": "a",
          "group_id": "s0000:r2"}
    store = {key: "text" for key in (STORE_EVAL1_RESPONSE, STORE_EVAL2_RESPONSE,
                                     STORE_EVAL1_REFLECTION, STORE_EVAL2_REFLECTION)}
    store[STORE_JUDGE_RESPONSE] = "<criterion_1_choice>1</criterion_1_choice>"
    record = pairwise_record(md, store, ["c1"])
    assert record["group_id"] == "s0000:r2"
    rows, _, groups = extract_comparisons_with_ties_criteria([record], num_criteria=1,
                                                             return_groups=True)
    assert rows == [[0, 0, 0, 0, 1, 1]] and groups == ["s0000:r2"]


CYCLING = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)


def test_eigentrust_damps_only_after_failing_to_converge():
    from pipeline.trust.eigentrust import damp_matrix, eigentrust

    with pytest.warns(RuntimeWarning, match="continuing with damping alpha=0.05"):
        at_1000 = eigentrust(CYCLING, max_iter=1000, verbose=False)
    with pytest.warns(RuntimeWarning):
        at_1001 = eigentrust(CYCLING, max_iter=1001, verbose=False)
    assert torch.allclose(at_1000, at_1001, atol=1e-5)
    damped = damp_matrix(CYCLING, 0.05)
    assert torch.norm(at_1000 @ damped - at_1000, p=1) < 1e-5


def test_eigentrust_is_unchanged_when_it_converges(recwarn):
    from pipeline.trust.eigentrust import eigentrust

    T = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4]], dtype=torch.float64)
    t = eigentrust(T, verbose=False)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
    assert torch.norm(t @ T - t, p=1) < 1e-6
    with pytest.raises(RuntimeError, match="did not converge"):
        eigentrust(CYCLING, max_iter=1000, verbose=False, fallback_alpha=0)


def test_bootstrap_refuses_plain_bt():
    from pipeline.train.bootstrap import _build_model_and_loader

    with pytest.raises(ValueError, match="btd_ties"):
        _build_model_and_loader([[0, 0, 0, 0, 1, 1]], "bt", 1, 2, 2, 4)
