"""Trust-matrix overflow, bootstrap plot, empty holdout, partitioned planning and model order."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _btd(scores, log_lambda):
    """A stand-in model whose u @ v.T equals ``scores``."""
    n = scores.shape[0]
    layer = lambda w: SimpleNamespace(weight=SimpleNamespace(data=w))
    return SimpleNamespace(u=layer(torch.eye(n)), v=layer(scores.t().contiguous()),
                           log_lambda=layer(log_lambda))


def test_trust_matrix_is_finite_and_unchanged_by_a_score_shift():
    from pipeline.trust import compute_trust_matrix, compute_trust_matrix_ties, row_normalize

    scores = torch.tensor([[0.2, 1.0, -0.5], [0.0, 0.3, 0.9], [1.2, -0.4, 0.1]])
    log_lambda = torch.full((3, 1), 0.2)
    base = compute_trust_matrix_ties(_btd(scores, log_lambda))
    shifted = compute_trust_matrix_ties(_btd(scores + 100, log_lambda))
    assert torch.isfinite(shifted).all() and torch.allclose(base, shifted, atol=1e-5)
    plain = row_normalize(compute_trust_matrix(_btd(scores + 100, log_lambda)))
    assert torch.allclose(plain, row_normalize(compute_trust_matrix(_btd(scores, log_lambda))), atol=1e-5)


def test_bootstrap_plot_accepts_a_mean_outside_its_interval(tmp_path):
    from pipeline.train.bootstrap import _save_bootstrap_plot

    rows = [{"model_name": "a", "elo_mean": 1591.4, "elo_ci_lower": 1598.4, "elo_ci_upper": 1620.0},
            {"model_name": "b", "elo_mean": 1400.0, "elo_ci_lower": 1390.0, "elo_ci_upper": 1410.0}]
    _save_bootstrap_plot(rows, tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0


def test_empty_grouped_holdout_is_reported_not_scored():
    from pipeline.train.train import group_split_comparisons

    comparisons = [[0, s, 0, 0, 1, 1] for s in range(4)]
    with pytest.warns(RuntimeWarning, match="leaves no test groups"):
        train, test = group_split_comparisons(comparisons, test_size=0.2, random_state=42)
    assert len(train) == 4 and test == []


def test_partitioned_planning_restarts_instead_of_failing():
    from pipeline.protocol.direct_rating import build_direct_assignments

    models = dict.fromkeys("abcd")
    plan = build_direct_assignments([(0, "s0"), (1, "s1")], models, include_self=False,
                                    sampler_mode="partitioned_random_judge", group_size=3,
                                    response_redundancy=3, sampler_seed=42)
    for scenario in (0, 1):
        judges = {e: [] for e in range(4)}
        for row in plan:
            if row["scenario_index"] == scenario:
                for e in row["eval_idxs"]:
                    judges[e].append(row["judge_idx"])
        for evaluee, seen in judges.items():
            assert len(seen) == 3 == len(set(seen)) and evaluee not in seen


def test_direct_analysis_refuses_a_reordered_spec():
    from pipeline.train.train import check_model_order

    check_model_order([(0, "a"), (1, "b"), (2, "other-name")], ["a", "b", "c"])
    with pytest.raises(ValueError, match="spec lists it at index 1"):
        check_model_order([(0, "b"), (1, "a")], ["a", "b"])


def test_pairwise_analysis_refuses_gaps_in_model_indices():
    from pipeline.train.pairwise_analysis import run_pairwise_analysis

    def record(e1, e2, choice):
        return {"scenario_index": 0, "judge": 0, "eval1": e1, "eval2": e2,
                "judge response": f"<criterion_1_choice>{choice}</criterion_1_choice>",
                "eval1 response": "r", "eval2 response": "r",
                "eval1 reflection": "f", "eval2 reflection": "f"}

    with pytest.raises(ValueError, match="without gaps"):
        run_pairwise_analysis(data=[record(0, 2, 1), record(2, 0, 2)], models={},
                              num_criteria=1, train_cfg={"dims": [2]})
