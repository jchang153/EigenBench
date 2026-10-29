"""Offline sampling and generation-profile parity with the original collector."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspect_pipeline.pairwise_plan import (
    PAIRWISE_GENERATION_DEFAULTS,
    plan_pairwise_groups,
    resolve_pairwise_generation_settings,
    resolve_pairwise_sampling_settings,
)

def _with_global_seed(seed, fn):
    state = random.getstate()
    try:
        random.seed(seed)
        result = fn()
        return result, random.random()
    finally:
        random.setstate(state)



@pytest.mark.parametrize("num_models", [2, 3, 4, 5, 7])
def test_samplers_replay_the_original_rng_stream(num_models):
    """Same draws and the same RNG consumption as pipeline/eval/samplers.py,
    including the discarded padding sample inside random_groups."""

    from pipeline.eval import samplers

    for group_size in range(2, num_models + 1):
        for seed in range(25):
            want, next_value = _with_global_seed(
                seed, lambda: samplers.sampler_random_judge_group(num_models, group_size)
            )
            rng = random.Random(seed)
            assert samplers.sampler_random_judge_group(num_models, group_size, rng=rng) == want
            assert rng.random() == next_value

            want, next_value = _with_global_seed(
                seed, lambda: samplers.sampler_uniform(num_models, group_size)
            )
            rng = random.Random(seed)
            assert samplers.sampler_uniform(num_models, group_size, rng=rng) == want
            assert rng.random() == next_value



@pytest.mark.parametrize("mode", ["random_judge_group", "uniform"])
def test_plan_follows_the_original_scenario_then_round_loop(mode):
    from pipeline.eval import samplers

    selected = [(4, "S4"), (9, "S9"), (2, "S2")]
    nicks = ["w", "x", "y", "z", "v"]
    kwargs = dict(sampler_mode=mode, group_size=3, groups=2, sampler_seed=11)
    groups = plan_pairwise_groups(selected, nicks, **kwargs)
    golden = {
        "random_judge_group": [(4, [2, 0, 1]), (1, [2, 4, 0]), (4, [3, 1, 2]),
                               (0, [4, 3, 2]), (2, [3, 0, 2]), (2, [4, 0, 2])],
        "uniform": [(3, [4, 3, 1]), (4, [4, 1, 0]), (4, [3, 1, 0]),
                    (3, [2, 1, 0]), (4, [0, 3, 1]), (4, [1, 0, 2])],
    }
    assert [(group.judge_idx, list(group.eval_idxs)) for group in groups] == golden[mode]
    assert groups == plan_pairwise_groups(selected, nicks, **kwargs)
    assert groups != plan_pairwise_groups(selected, nicks, **{**kwargs, "sampler_seed": 12})

    # One global stream, consumed scenario by scenario and round by round, as
    # run_collect.py drove collect_core_evaluations.
    sampler = samplers.select_sampler(mode)

    def original():
        draws = []
        for _ in selected:
            for _round in range(2):
                draws.append(sampler(num_models=len(nicks), group_size=3,
                                     judge_counts=[0] * 5, eval_counts=[0] * 5, alpha=0.0))
        return draws

    draws, _ = _with_global_seed(11, original)
    assert [(g.judge_idx, list(g.eval_idxs)) for g in groups] == [
        (int(j), list(e)) for j, e in draws
    ]
    assert [(g.scenario_index, g.sampling_round, g.group_id) for g in groups] == [
        (4, 0, "s0004:r0"), (4, 1, "s0004:r1"), (9, 0, "s0009:r0"),
        (9, 1, "s0009:r1"), (2, 0, "s0002:r0"), (2, 1, "s0002:r1"),
    ]
    assert all(g.judge_nick == nicks[g.judge_idx] for g in groups)



def test_sampling_settings_match_collect_core_evaluations():
    assert resolve_pairwise_sampling_settings({}, num_models=6) == {
        "sampler_mode": "random_judge_group", "group_size": 4, "groups": 1, "sampler_seed": 42,
    }
    clamped = resolve_pairwise_sampling_settings(
        {"sampler_mode": " Uniform ", "group_size": 9, "groups": 0}, num_models=3
    )
    assert clamped == {"sampler_mode": "uniform", "group_size": 3, "groups": 1, "sampler_seed": 42}
    with pytest.raises(ValueError, match="cannot be planned before collection"):
        resolve_pairwise_sampling_settings({"sampler_mode": "adaptive_inverse_count"}, num_models=4)
    with pytest.raises(ValueError, match="mixed local collector"):
        resolve_pairwise_sampling_settings({"sampler_mode": "all_to_all"}, num_models=4)
    with pytest.raises(ValueError, match="Unknown sampler mode"):
        resolve_pairwise_sampling_settings({"sampler_mode": "balanced"}, num_models=4)
    with pytest.raises(ValueError, match="at least two models"):
        resolve_pairwise_sampling_settings({}, num_models=1)
    with pytest.raises(ValueError, match="integer collection.sampler_seed"):
        resolve_pairwise_sampling_settings({"sampler_seed": None}, num_models=4)



def test_generation_profile_is_the_original_and_ignores_direct_settings():
    expected = {phase: {"max_tokens": 4096, "temperature": 1.0, "per_model": {}}
                for phase in ("response", "reflection", "comparison")}
    assert resolve_pairwise_generation_settings({}) == expected
    # runs/example/spec.py carries a direct-rating block and a legacy max_tokens;
    # the original pairwise path forwarded neither.
    legacy = {
        "max_tokens": 1024,
        "generation": {
            "response": {"max_tokens": 4096, "temperature": 0.7},
            "reflection": {"max_tokens": 2048, "temperature": 0.2},
            "direct_rating": {"max_tokens": 512, "temperature": 0.0},
        },
    }
    assert resolve_pairwise_generation_settings(legacy) == expected
    assert PAIRWISE_GENERATION_DEFAULTS["comparison"] == {"max_tokens": 4096, "temperature": 1.0}

    custom = resolve_pairwise_generation_settings({"pairwise_generation": {
        "comparison": {"max_tokens": 600, "per_model": {"a": {"temperature": 0.5}}},
    }})
    assert custom["comparison"] == {
        "max_tokens": 600, "temperature": 1.0,
        "per_model": {"a": {"max_tokens": 600, "temperature": 0.5}},
    }
    assert custom["response"] == expected["response"]
    for bad in ({"comparison": {"max_tokens": 0}}, {"verdict": {}},
                {"reflection": {"temperature": -1}}, {"response": {"per_model": []}}):
        with pytest.raises(ValueError):
            resolve_pairwise_generation_settings({"pairwise_generation": bad})
