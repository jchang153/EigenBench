"""Focused regressions for this change."""
import pytest
from test_inspect_collect import run_dir, NICKS, SCENARIO_COUNT

def test_per_model_budgets_are_resolved_and_validated():
    from pipeline.eval.direct_rating import resolve_direct_generation_settings
    from inspect_pipeline.phases import phase_config
    settings = resolve_direct_generation_settings({"generation": {"reflection": {
        "max_tokens": 512, "per_model": {"api": {"max_tokens": 2048}},
    }}})
    assert phase_config(settings["reflection"], "api").max_tokens == 2048
    assert phase_config(settings["reflection"], "local").max_tokens == 512
    with pytest.raises(ValueError, match="invalid generation"):
        resolve_direct_generation_settings({"generation": {"reflection": {"per_model": {"api": {"max_tokens": 0}}}}})


@pytest.mark.parametrize("phased", [False, True])
def test_generation_budgets_reach_provider_in_both_collection_modes(run_dir, phased):
    tmp_path, write_spec = run_dir
    path = write_spec(sampler_mode="balanced_unique_judge")
    text = path.read_text()
    # Check the actual config received by the mock provider in every phase.
    text = text.replace('system = input[0].text if input else ""', 'system = input[0].text if input else ""\n        phase = "response" if system.startswith("Without making") else "reflection" if system.startswith("You are acting as a judge. Carefully consider the following response") else "direct_rating"\n        expected = {"response": 701, "reflection": 1701, "direct_rating": 1702} if nick == "alpha" else {"response": 4096, "reflection": 2048, "direct_rating": 512}\n        assert config.max_tokens == expected[phase], (nick, phase, config.max_tokens)')
    text = text.replace('"enabled": True,', '"enabled": True,\n        "generation": {\n            "response": {"per_model": {"alpha": {"max_tokens": 701}}},\n            "reflection": {"per_model": {"alpha": {"max_tokens": 1701}}},\n            "direct_rating": {"per_model": {"alpha": {"max_tokens": 1702}}},\n        },', 1)
    if phased:
        text = text.replace('"cache": False, "display": "none"', '"cache": False, "display": "none", "phased": True')
    path.write_text(text)
    from inspect_pipeline.collect import collect_direct_ratings_inspect
    records = collect_direct_ratings_inspect(str(path))
    assert len(records) == SCENARIO_COUNT * len(NICKS)
