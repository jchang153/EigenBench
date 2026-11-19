"""Comparison cleaning, EigenTrust convergence and the BT route."""

from collections import Counter

import numpy as np
import pytest
import torch


def planned_rows(prefer):
    from pipeline.protocol.pairwise_plan import pairwise_comparisons, plan_pairwise_groups

    groups = plan_pairwise_groups([(0, "s0")], ["a", "b"], sampler_mode="uniform",
                                  group_size=2, groups=3, sampler_seed=42)
    plan = pairwise_comparisons(groups)
    rows = [[0, 0, p["judge_idx"], p["eval1_idx"], p["eval2_idx"], prefer(p)] for p in plan]
    return rows, [p["group_id"] for p in plan]


def test_repeated_rounds_of_one_pair_are_all_kept():
    from pipeline.analysis.comparisons import handle_inconsistencies_with_ties_criteria

    rows, groups = planned_rows(lambda p: 1 if p["eval1_idx"] == 0 else 2)
    assert len(rows) == 6
    for given in (groups, None):
        cleaned = handle_inconsistencies_with_ties_criteria(rows, given)
        assert Counter(map(tuple, cleaned)) == Counter(map(tuple, rows))


def test_strict_inconsistency_ties_only_its_own_round():
    from pipeline.analysis.comparisons import handle_inconsistencies_with_ties_criteria

    rows, groups = planned_rows(lambda p: 1 if p["eval1_idx"] == 0 else 2)
    first = groups[0]
    for row, group in zip(rows, groups):
        if group == first:
            row[-1] = 1
    cleaned = handle_inconsistencies_with_ties_criteria(rows, groups)
    assert [row[-1] for row in cleaned].count(0) == 2
    assert len(cleaned) == 6


def test_same_order_repeats_are_not_turned_into_ties():
    from pipeline.analysis.comparisons import handle_inconsistencies_with_ties_criteria

    rows = [[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]]
    assert handle_inconsistencies_with_ties_criteria(rows) == rows


def test_order_rule_matches_the_paper_for_one_pair():
    from pipeline.analysis.comparisons import handle_inconsistencies_with_ties_criteria

    def clean(first, second):
        return [row[-1] for row in handle_inconsistencies_with_ties_criteria(
            [[0, 0, 0, 1, 2, first], [0, 0, 0, 2, 1, second]])]

    assert clean(1, 2) == [1, 2]
    assert clean(1, 1) == [0, 0]
    assert clean(2, 2) == [0, 0]
    assert clean(1, 0) == [1, 0]
    assert clean(0, 0) == [0, 0]


def test_gaps_in_model_ids_keep_their_comparisons():
    from pipeline.analysis.comparisons import handle_inconsistencies_with_ties_criteria

    rows = [[0, 5, 0, 0, 2, 1], [0, 5, 0, 2, 0, 2]]
    assert handle_inconsistencies_with_ties_criteria(rows) == rows


def test_exported_pairwise_records_carry_their_group():
    from inspect_pipeline.pairwise import (STORE_EVAL1_REFLECTION, STORE_EVAL1_RESPONSE,
                                           STORE_EVAL2_REFLECTION, STORE_EVAL2_RESPONSE,
                                           STORE_JUDGE_RESPONSE, pairwise_record)
    from pipeline.analysis.comparisons import extract_comparisons_with_ties_criteria

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
    assert rows == [[0, 0, 0, 0, 1, 1]] and groups[0][:2] == ("s0000:r2", None)


def _legacy(e1, e2, choice, draw, *, group=None, epoch=None):
    record = {"scenario_index": 0, "judge": 0, "eval1": e1, "eval2": e2,
              "judge response": f"<criterion_1_choice>{choice}</criterion_1_choice>",
              "eval1 response": f"r{e1}-{draw}", "eval2 response": f"r{e2}-{draw}",
              "eval1 reflection": f"f{e1}-{draw}", "eval2 reflection": f"f{e2}-{draw}"}
    if group is not None:
        record.update(group_id=group, epoch=epoch)
    return record


def _clean(records):
    from pipeline.analysis.comparisons import (
        extract_comparisons_with_ties_criteria, handle_inconsistencies_with_ties_criteria,
    )

    rows, _, groups = extract_comparisons_with_ties_criteria(records, num_criteria=1, return_groups=True)
    diagnostics = {}
    return [row[-1] for row in handle_inconsistencies_with_ties_criteria(rows, groups, diagnostics)], diagnostics


def test_records_without_groups_pair_only_within_one_draw():
    # Round 0 kept only its A-B judgment; round 1 has both orders and is consistent.
    records = [_legacy(0, 1, 1, "round0"), _legacy(0, 1, 2, "round1"), _legacy(1, 0, 1, "round1")]
    assert _clean(records)[0] == [1, 2, 1]


def test_epochs_of_one_group_are_separate_draws():
    records = [_legacy(0, 1, 1, "e1", group="g", epoch=1), _legacy(0, 1, 2, "e2", group="g", epoch=2),
               _legacy(1, 0, 1, "e2", group="g", epoch=2)]
    assert _clean(records)[0] == [1, 2, 1]


def test_draw_payload_disambiguates_reused_group_and_epoch():
    forward = _legacy(0, 1, 1, "first", group="g", epoch=1)
    different_draw = _legacy(1, 0, 1, "second", group="g", epoch=1)
    same_draw = _legacy(1, 0, 1, "first", group="g", epoch=1)
    assert _clean([forward, different_draw])[0] == [1, 1]
    assert _clean([forward, same_draw])[0] == [0, 0]


def test_ambiguous_draws_are_kept_and_counted():
    records = [_legacy(0, 1, 1, "same"), _legacy(0, 1, 1, "same"), _legacy(1, 0, 1, "same")]
    scores, diagnostics = _clean(records)
    assert scores == [1, 1, 1] and diagnostics == {"comparisons_ambiguous": 3}


def test_pairwise_analysis_refuses_gaps_in_model_indices():
    from pipeline.analysis.pairwise import run_pairwise_analysis

    records = [_legacy(0, 2, 1, s) for s in "abc"] + [_legacy(2, 0, 2, s) for s in "abc"]
    for n, record in enumerate(records):
        record["scenario_index"] = n % 3
    with pytest.raises(ValueError, match="without gaps"):
        run_pairwise_analysis(data=records, models=dict.fromkeys("ABC"), num_criteria=1,
                              train_cfg={"dims": [2]})


CYCLING = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)


def _weak_blocks(eps=1e-4):
    """Two blocks joined by ``eps``: undamped iteration needs far more than 1000 steps."""
    T = torch.tensor([[0.5, 0.5 - eps, eps / 2, eps / 2],
                      [0.5 - eps, 0.5, eps / 2, eps / 2],
                      [eps, eps, 0.9, 0.1 - 2 * eps],
                      [eps, eps, 0.9 - 2 * eps, 0.1]], dtype=torch.float64)
    return T


def _stationary(T):
    values, vectors = np.linalg.eig(T.numpy().T)
    t = np.real(vectors[:, np.argmin(np.abs(values - 1))])
    return t / t.sum()


@pytest.mark.parametrize("T", [CYCLING, _weak_blocks()], ids=["periodic", "slow"])
def test_eigentrust_returns_the_stationary_vector_when_iteration_stalls(T, recwarn):
    from pipeline.analysis.eigentrust import eigentrust

    at_1000 = eigentrust(T, max_iter=1000, verbose=False)
    assert torch.allclose(at_1000, eigentrust(T, max_iter=1001, verbose=False), atol=1e-9)
    assert np.allclose(at_1000.numpy(), _stationary(T), atol=1e-8)
    assert torch.norm(at_1000 @ T - at_1000, p=1) < 1e-8
    assert not recwarn.list


def test_eigentrust_is_unchanged_when_it_converges():
    from pipeline.analysis.eigentrust import eigentrust

    T = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4]], dtype=torch.float64)
    assert torch.norm(eigentrust(T, verbose=False) @ T - eigentrust(T, verbose=False), p=1) < 1e-6


def test_eigentrust_refuses_a_matrix_without_a_unique_answer():
    from pipeline.analysis.eigentrust import eigentrust

    # Two closed classes, {0, 1} and {2, 3}, each cycling; model 4 feeds the second.
    split = torch.tensor([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="no unique stationary vector"):
        eigentrust(split, max_iter=10, verbose=False)


def test_direct_minmax_refuses_converged_nonunique_eigentrust():
    from pipeline.analysis.direct_trust import build_direct_trust

    rows = [
        {"record_type": "direct_rating", "scenario_index": 0,
         "judge": {"index": judge}, "evaluee": {"index": evaluee},
         "ratings": [{"criterion_index": 0, "rating": 10 if judge == evaluee else 1}]}
        for judge in range(3)
        for evaluee in range(3)
    ]
    config = {"num_models": 3, "num_criteria": 1, "normalization": "minmax_l1"}
    with pytest.raises(RuntimeError, match="no unique stationary vector"):
        build_direct_trust(rows, **config)

    result = build_direct_trust(rows, **config, eigentrust_alpha=0.1)
    np.testing.assert_allclose(result.eigentrust_scores, np.full(3, 1 / 3))


def test_bootstrap_refuses_plain_bt():
    from pipeline.analysis.pairwise_bootstrap import _build_model_and_loader

    with pytest.raises(ValueError, match="btd_ties"):
        _build_model_and_loader([[0, 0, 0, 0, 1, 1]], "bt", 1, 2, 2, 4)


@pytest.mark.parametrize("module", ["pipeline.analysis.pairwise_bootstrap", "pipeline.analysis.direct_rating"])
def test_bootstrap_plot_accepts_a_mean_outside_its_interval(tmp_path, module):
    import importlib

    _save_bootstrap_plot = importlib.import_module(module)._save_bootstrap_plot
    rows = [{"model_name": "a", "elo_mean": 1591.4, "elo_ci_lower": 1598.4, "elo_ci_upper": 1620.0},
            {"model_name": "b", "elo_mean": 1400.0, "elo_ci_lower": 1390.0, "elo_ci_upper": 1410.0}]
    _save_bootstrap_plot(rows, tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0


def test_empty_grouped_holdout_is_reported_not_scored():
    from pipeline.analysis.fit import group_split_comparisons

    comparisons = [[0, s, 0, 0, 1, 1] for s in range(4)]
    with pytest.warns(RuntimeWarning, match="leaves no test groups"):
        train, test = group_split_comparisons(comparisons, test_size=0.2, random_state=42)
    assert len(train) == 4 and test == []


def test_direct_analysis_refuses_records_from_another_panel():
    from pipeline.analysis.fit import check_model_order

    check_model_order([(0, "a"), (1, "b"), (1, None)], ["a", "b", "c"])
    for observed in ([(0, "b"), (1, "a")], [(0, "c"), (1, "d")]):
        with pytest.raises(ValueError, match="spec lists"):
            check_model_order(observed, ["a", "b"])
