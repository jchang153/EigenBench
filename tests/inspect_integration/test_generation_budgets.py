"""Collection configuration and provider generation regressions."""

import ast
import importlib.util
from pathlib import Path

import pytest
from tests.inspect_integration.test_inspect_collect import run_dir, NICKS, SCENARIO_COUNT


@pytest.mark.parametrize("generation_key", ["generation", "pairwise_generation"])
@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("per_model", [False, True])
def test_generation_temperatures_must_be_finite(generation_key, temperature, per_model):
    from eigenbench.protocol.pairwise_plan import resolve_pairwise_generation_settings
    from eigenbench.protocol.direct_rating import resolve_direct_generation_settings

    resolve = (
        resolve_direct_generation_settings
        if generation_key == "generation"
        else resolve_pairwise_generation_settings
    )
    phase = {"temperature": temperature}
    if per_model:
        phase = {"per_model": {"api": phase}}
    with pytest.raises(ValueError, match="temperature.*finite non-negative"):
        resolve({generation_key: {"reflection": phase}})


def test_hosted_worker_can_load_config_tools_without_importing_the_package(tmp_path):
    # ValueArena's worker verifies the runner source by executing only these entry points.
    root = Path(__file__).resolve().parents[2]
    direct_source = root / "eigenbench/protocol/direct_rating.py"
    tree = ast.parse(direct_source.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_direct_generation_settings"
    )
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(direct_source), "exec"), scope)
    resolve = scope["resolve_direct_generation_settings"]
    assert resolve({}) == {
        "response": {"max_tokens": 4096, "temperature": 0.7, "per_model": {}},
        "reflection": {"max_tokens": 2048, "temperature": 0.2, "per_model": {}},
        "direct_rating": {"max_tokens": 512, "temperature": 0.0, "per_model": {}},
    }
    assert resolve({"max_tokens": "700", "generation": {"reflection": {
        "per_model": {"api": {"max_tokens": "850"}},
    }}})["reflection"] == {
        "max_tokens": 700, "temperature": 0.2,
        "per_model": {"api": {"max_tokens": 850, "temperature": 0.2}},
    }

    module_spec = importlib.util.spec_from_file_location(
        "upstream_run_spec", root / "eigenbench/config/run_spec.py"
    )
    run_spec = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(run_spec)
    path = tmp_path / "spec.py"
    path.write_text("RUN_SPEC = {'evaluation': {'mode': 'direct_rating', 'direct_rating': "
                    "{'criterion_aggregation': ' MEAN ', 'scenario_aggregation': 'mean'}}}")
    spec, _ = run_spec.load_run_spec(str(path))
    assert spec["evaluation"]["direct_rating"]["criterion_aggregation"] == "mean"
    assert spec["evaluation"]["direct_rating"]["scenario_aggregation"] == "mean"
    assert spec["collection"]["sampler_seed"] == 42


def test_direct_sampler_alias_agrees_across_loading_planning_and_estimation(tmp_path):
    from eigenbench.config import apply_run_defaults
    from eigenbench.protocol.direct_rating import build_direct_assignments, estimate_direct_calls

    models = dict.fromkeys(["a", "b", "c"])
    sampling = {"sampler_mode": " Random_Partition ", "group_size": 2, "sampler_seed": 17}
    spec, _ = apply_run_defaults("spec.py", str(tmp_path / "spec.py"), {
        "evaluation": {"mode": "direct_rating"},
        "models": models,
        "collection": sampling,
    })
    assert spec["collection"]["sampler_mode"] == "partitioned_random_judge"
    plan = build_direct_assignments([(0, "scenario")], models, **sampling)
    assert {assignment["sampler_mode"] for assignment in plan} == {"partitioned_random_judge"}
    estimate = estimate_direct_calls(num_scenarios=1, num_models=3, **sampling)
    assert estimate["sampler_mode"] == "partitioned_random_judge"
    assert estimate["rating_tasks"] == sum(len(assignment["eval_idxs"]) for assignment in plan)


def test_per_model_budgets_are_resolved_and_validated():
    from eigenbench.protocol.direct_rating import resolve_direct_generation_settings
    from inspect_pipeline.generation import phase_config
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
    from inspect_pipeline.collection.direct_rating import collect_direct_ratings_inspect
    records = collect_direct_ratings_inspect(str(path))
    assert len(records) == SCENARIO_COUNT * len(NICKS)
