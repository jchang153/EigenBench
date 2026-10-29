"""Offline parity between the pairwise runner's execution layouts."""

import json
from collections import Counter
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_ai.model import ModelOutput, get_model
from inspect_pipeline.pairwise_collect import collect_pairwise_inspect, estimate_pairwise_calls


def write_spec(folder, *, phased=False, groups=2):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scenarios.json").write_text(json.dumps(["Scenario zero", "Scenario one"]))
    (folder / "constitution.json").write_text(json.dumps(["c1", "c2"]))
    spec = {
        "name": "pairwise_test",
        "models": {nick: "inspect:mockllm/model" for nick in ("a", "b", "c")},
        "evaluation": {"mode": "pairwise_btd"},
        "dataset": {"path": "scenarios.json", "count": 2},
        "constitution": {"path": "constitution.json", "num_criteria": 2},
        "collection": {"group_size": 3, "groups": groups, "sampler_seed": 42,
                       "sampler_mode": "random_judge_group",
                       "inspect": {"phased": phased, "cache": False,
                                   "display": "none", "max_samples": 1}},
        "training": {"enabled": False},
    }
    path = folder / "spec.py"
    path.write_text(f"RUN_SPEC = {spec!r}\n")
    return str(path)


def panel(calls, *, invalid=False):
    def mock(nick):
        def output(input, tools, tool_choice, config):
            system, user = input[0].text, input[1].text
            if system.startswith("Without making"):
                phase, text = "response", f"{nick} answers {user}"
            elif system.startswith("You are acting as a judge. Carefully consider the following response"):
                phase, text = "reflection", f"{nick} reflects {user}"
            else:
                phase = "comparison"
                text = "invalid" if invalid else (
                    "<criterion_1_choice>0</criterion_1_choice>"
                    "<criterion_2_choice>2</criterion_2_choice>"
                )
            calls.append((phase, nick, user, config.max_tokens, config.temperature))
            return ModelOutput.from_content(model="mockllm/model", content=text)
        return get_model("mockllm/model", custom_outputs=output, memoize=False)
    return {nick: mock(nick) for nick in ("a", "b", "c")}


def test_fresh_pairwise_single_and_phased_match(tmp_path):
    results = []
    for phased in (False, True):
        spec = write_spec(tmp_path / str(phased), phased=phased)
        calls = []
        models = panel(calls)
        estimate = estimate_pairwise_calls(spec, models=models)
        assert estimate == {
            "mode": "pairwise_btd", "response_tasks": 12,
            "reflection_tasks": 12, "comparison_tasks": 24,
            "total_logical_generations": 48,
        }
        records = collect_pairwise_inspect(spec, models=models)
        assert len(records) == 24
        assert Counter(row[0] for row in calls) == {
            "response": 12, "reflection": 12, "comparison": 24,
        }
        assert {(row[3], row[4]) for row in calls} == {(4096, 1.0)}
        assert all("<criterion_1_choice>0</criterion_1_choice>" in row["judge response"]
                   for row in records)
        info = json.loads((Path(spec).parent / "inspect_run.json").read_text())
        assert info["log_file"]
        with pytest.raises(RuntimeError, match="already contains"):
            collect_pairwise_inspect(spec, models=models)
        results.append(records)
    assert results[0] == results[1]


def test_group_cache_preserves_draws_and_reuses_across_layouts(tmp_path, monkeypatch):
    monkeypatch.setenv("INSPECT_CACHE_DIR", str(tmp_path / "cache"))
    results = []
    for phased in (False, True):
        spec = write_spec(tmp_path / str(phased), phased=phased)
        with Path(spec).open("a") as handle:
            handle.write('\nRUN_SPEC["collection"]["inspect"]["cache"] = True\n')
        calls = []
        results.append(collect_pairwise_inspect(spec, models=panel(calls)))
        if phased:
            assert not calls, "phased collection should reuse the same generation keys"
        else:
            assert Counter(row[0] for row in calls) == {
                "response": 12, "reflection": 12, "comparison": 24,
            }, "identical prompts in distinct groups must remain separate draws"
    assert results[0] == results[1]


def test_native_epochs_draw_new_responses_and_reflections(tmp_path):
    from inspect_ai import eval as inspect_eval
    from inspect_pipeline.pairwise import eigenbench_pairwise

    spec = write_spec(tmp_path, groups=1)
    calls = []
    task = eigenbench_pairwise(spec, models=panel(calls))
    logs = inspect_eval(task, model=None, epochs=2, display="none",
                        log_dir=str(tmp_path / "epochs"))
    assert logs[0].status == "success"
    assert Counter(row[0] for row in calls) == {
        "response": 12, "reflection": 12, "comparison": 24,
    }


def test_pairwise_cache_uses_last_scenario_row(tmp_path):
    spec = write_spec(tmp_path)
    cache = tmp_path / "cache.jsonl"
    cache.write_text('\n'.join(json.dumps(row) for row in [
        {"scenario_index": 0, "responses": {"a": "obsolete a"}},
        {"scenario_index": 0, "responses": {"b": "cached b"}},
    ]) + '\n')
    with Path(spec).open("a") as handle:
        handle.write(f'\nRUN_SPEC["collection"]["cached_responses_path"] = {str(cache)!r}\n')
    calls = []
    models = panel(calls)
    assert estimate_pairwise_calls(spec, models=models)["response_tasks"] == 10
    records = collect_pairwise_inspect(spec, models=models)
    assert Counter(row[0] for row in calls)["response"] == 10
    for record in records:
        if record["scenario_index"] == 0:
            for side in ("eval1", "eval2"):
                if record[f"{side}_name"] == "b":
                    assert record[f"{side} response"] == "cached b"
                assert record[f"{side} response"] != "obsolete a"


@pytest.mark.parametrize("phased", [False, True])
def test_incomplete_pairwise_never_exports(tmp_path, phased):
    spec = write_spec(tmp_path, phased=phased, groups=1)
    with pytest.raises(RuntimeError):
        collect_pairwise_inspect(spec, models=panel([], invalid=True))
    assert not (tmp_path / "evaluations.jsonl").exists()
