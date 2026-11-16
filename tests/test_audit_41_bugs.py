"""Trust-matrix overflow, bootstrap plot, empty holdout, partitioned planning and model order."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_inspect_collect import run_dir  # noqa: E402,F401  (fixture)


def _btd(scores, log_lambda):
    """A stand-in model whose u @ v.T equals ``scores``."""
    n = scores.shape[0]
    layer = lambda w: SimpleNamespace(weight=SimpleNamespace(data=w))
    return SimpleNamespace(u=layer(torch.eye(n)), v=layer(scores.t().contiguous()),
                           log_lambda=layer(log_lambda))


def test_trust_matrix_is_unchanged_by_a_score_shift():
    from pipeline.trust import compute_trust_matrix, compute_trust_matrix_ties, row_normalize

    scores = torch.tensor([[0.2, 1.0, -0.5], [0.0, 0.3, 0.9], [1.2, -0.4, 0.1]])
    log_lambda = torch.full((3, 1), 0.2)
    base = compute_trust_matrix_ties(_btd(scores, log_lambda))
    shifted = compute_trust_matrix_ties(_btd(scores + 100, log_lambda))
    assert torch.isfinite(shifted).all() and torch.allclose(base, shifted, atol=1e-5)
    plain = row_normalize(compute_trust_matrix(_btd(scores + 100, log_lambda)))
    assert torch.allclose(plain, row_normalize(compute_trust_matrix(_btd(scores, log_lambda))), atol=1e-5)


def test_trust_matrix_keeps_ties_that_outweigh_an_underflowing_win():
    from pipeline.trust import compute_trust_matrix_ties

    # Row 0: s = [e^120, 1], lambda = e^60, so S = [1.5, 0.5] * e^120.
    scores = torch.tensor([[120.0, 0.0], [0.0, 0.0]])
    T = compute_trust_matrix_ties(_btd(scores, torch.tensor([[60.0], [0.0]])))
    assert torch.allclose(T[0], torch.tensor([0.75, 0.25]), atol=1e-6)
    T = compute_trust_matrix_ties(_btd(torch.zeros(2, 2), torch.full((2, 1), 100.0)))
    assert torch.allclose(T, torch.full((2, 2), 0.5))


@pytest.mark.parametrize("module", ["pipeline.train.bootstrap", "pipeline.train.direct_analysis"])
def test_bootstrap_plot_accepts_a_mean_outside_its_interval(tmp_path, module):
    import importlib

    _save_bootstrap_plot = importlib.import_module(module)._save_bootstrap_plot
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


def _judges_per_evaluee(plan, scenario):
    judges = {}
    for row in plan:
        if row["scenario_index"] == scenario:
            for e in row["eval_idxs"]:
                judges.setdefault(e, []).append(row["judge_idx"])
    return judges


@pytest.mark.parametrize("models,include_self,group_size,redundancy", [
    (4, False, 3, 3),   # the audit's case, solved by restarting
    (9, True, 4, 9),    # needs the rotating fallback
])
def test_partitioned_planning_finds_a_schedule(models, include_self, group_size, redundancy):
    from pipeline.protocol.direct_rating import build_direct_assignments

    plan = build_direct_assignments([(0, "s0"), (1, "s1")], dict.fromkeys(range(models)),
                                    include_self=include_self, sampler_mode="partitioned_random_judge",
                                    group_size=group_size, response_redundancy=redundancy, sampler_seed=42)
    for scenario in (0, 1):
        judges = _judges_per_evaluee(plan, scenario)
        assert len(judges) == models
        for evaluee, seen in judges.items():
            assert len(seen) == redundancy == len(set(seen))
            assert include_self or evaluee not in seen


def test_direct_analysis_refuses_records_from_another_panel():
    from pipeline.train.train import check_model_order

    check_model_order([(0, "a"), (1, "b"), (1, None)], ["a", "b", "c"])
    for observed in ([(0, "b"), (1, "a")], [(0, "c"), (1, "d")]):
        with pytest.raises(ValueError, match="spec lists"):
            check_model_order(observed, ["a", "b"])


def test_cli_refuses_a_saved_file_with_another_panels_scenario(run_dir):
    import json as _json

    from inspect_pipeline.collect import collect_direct_ratings_inspect
    from scripts.run import main

    tmp_path, write_spec = run_dir
    spec = write_spec()
    collect_direct_ratings_inspect(str(spec))
    evaluations = tmp_path / "evaluations.jsonl"
    records = [_json.loads(line) for line in evaluations.read_text().splitlines()]
    foreign = [dict(r, scenario_index=99, judge={**r["judge"], "name": "x" + r["judge"]["name"]},
                    evaluee={**r["evaluee"], "name": "x" + r["evaluee"]["name"]})
               for r in records if r["scenario_index"] == 0]
    evaluations.write_text("".join(_json.dumps(r) + "\n" for r in foreign + records))
    with pytest.raises(ValueError, match="spec lists"):
        main(str(spec), collection_enabled=False)
