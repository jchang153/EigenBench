"""Partitioned direct-rating assignment coverage."""

import pytest


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
    (7, False, 6, 6),   # needs a rotating singleton
    (9, False, 7, 7),   # needs a rotating remainder with two evaluees
    (14, False, 5, 11), # some judges must score both full and remainder groups
])
def test_partitioned_planning_finds_a_schedule(models, include_self, group_size, redundancy):
    from pipeline.protocol.direct_rating import build_direct_assignments

    plan = build_direct_assignments([(0, "s0"), (1, "s1")], dict.fromkeys(range(models)),
                                    include_self=include_self, sampler_mode="partitioned_random_judge",
                                    group_size=group_size, response_redundancy=redundancy, sampler_seed=42)
    for scenario in (0, 1):
        for sampling_round in range(redundancy):
            groups = [row["eval_idxs"] for row in plan if row["scenario_index"] == scenario
                      and row["sampling_round"] == sampling_round]
            assert sorted(e for group in groups for e in group) == list(range(models))
            assert [len(group) for group in groups] == [
                min(group_size, models - i) for i in range(0, models, group_size)
            ]
        judges = _judges_per_evaluee(plan, scenario)
        assert len(judges) == models
        for evaluee, seen in judges.items():
            assert len(seen) == redundancy == len(set(seen))
            assert include_self or evaluee not in seen


@pytest.mark.parametrize("models,group_size,redundancy", [(6, 3, 4), (9, 7, 8), (14, 5, 12)])
def test_partitioned_planning_reports_insufficient_judge_capacity(models, group_size, redundancy):
    from pipeline.protocol.direct_rating import build_direct_assignments

    with pytest.raises(ValueError, match="exceed the capacity of distinct non-self judges"):
        build_direct_assignments([(0, "s0")], dict.fromkeys(range(models)), include_self=False,
                                 sampler_mode="partitioned_random_judge", group_size=group_size,
                                 response_redundancy=redundancy, sampler_seed=42)
