"""collection.failure_policy on the Inspect engine (offline, mockllm).

``omit_invalid_judgments`` leaves out a judgment that stayed invalid after every attempt
and records it in omitted_samples.json, as the native engine does. Failed responses and
other errors stop the run under either policy and name the models involved.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NICKS = ["a", "b"]
SCENARIOS = 2
PLANNED = SCENARIOS * len(NICKS) ** 2  # all_to_all with self-ratings

# Judge "a" fails on scenario 0 for evaluee "b", in the way FAULT names:
#   reflection / rating: an empty or unparseable completion on every attempt
#   error: the provider raises, like an authentication failure
#   response: evaluee "b" never answers scenario 0
#   all_ratings: every judge's rating is unparseable
SPEC_TEMPLATE = '''
import re
from inspect_ai.model import ModelOutput, get_model

FAULT = {fault!r}


def _mock(nick):
    def outputs(input, tools, tool_choice, config):
        system = input[0].text
        user = input[1].text
        scenario = int(re.search(r"Scenario number (\\d)", user).group(1))
        if system.startswith("Without making any mention"):
            if FAULT == "response" and nick == "b" and scenario == 0:
                return ModelOutput.from_content(model="mockllm/model", content="")
            return ModelOutput.from_content(model="mockllm/model", content=f"[{{nick}}] answer {{scenario}}")
        evaluee = re.search(r"\\[(\\w)\\] answer", user).group(1)
        failing = nick == "a" and evaluee == "b" and scenario == 0
        if system.startswith("You are acting as a judge. Carefully consider the following response"):
            if failing and FAULT == "reflection":
                return ModelOutput.from_content(model="mockllm/model", content="")
            return ModelOutput.from_content(model="mockllm/model", content="reflection")
        if failing and FAULT == "error":
            raise PermissionError("401 Unauthorized: invalid API key")
        if (failing and FAULT == "rating") or FAULT == "all_ratings":
            return ModelOutput.from_content(model="mockllm/model", content="I would rather not rate this.")
        return ModelOutput.from_content(model="mockllm/model", content="<criterion_1_rating>7</criterion_1_rating>")

    return get_model("mockllm/model", custom_outputs=outputs, memoize=False)


RUN_SPEC = {{
    "name": "policytest",
    "verbose": False,
    "models": {{nick: _mock(nick) for nick in {nicks!r}}},
    "evaluation": {{"mode": "direct_rating"}},
    "dataset": {{"path": {scenarios_path!r}, "start": 0, "count": {scenarios}}},
    "constitution": {{"path": {constitution_path!r}, "num_criteria": 1}},
    "collection": {{
        "enabled": True,
        "sampler_mode": "all_to_all",
        "failure_policy": {policy!r},
        "openrouter": {{"max_attempts": 2}},
        "inspect": {{"cache": False, "display": "none", "phased": {phased!r}}},
    }},
    "training": {{"enabled": False}},
}}
'''


@pytest.fixture
def write_spec(tmp_path):
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps([f"Scenario number {i}: what now?" for i in range(SCENARIOS)]))
    constitution_path = tmp_path / "constitution.json"
    constitution_path.write_text(json.dumps(["Criterion 1: prefer the kinder response"]))

    def write(*, fault=None, policy="omit_invalid_judgments", phased=False):
        path = tmp_path / "spec.py"
        path.write_text(SPEC_TEMPLATE.format(
            fault=fault, policy=policy, phased=phased, nicks=NICKS, scenarios=SCENARIOS,
            scenarios_path=str(scenarios_path), constitution_path=str(constitution_path),
        ))
        return str(path)

    return write


def collect(spec_path):
    from inspect_pipeline.collect import collect_direct_ratings_inspect

    return collect_direct_ratings_inspect(spec_path)


@pytest.mark.parametrize("phased", [False, True])
@pytest.mark.parametrize("fault,stage", [("rating", "direct_rating"), ("reflection", "reflection")])
def test_invalid_judgment_is_omitted_and_recorded(write_spec, tmp_path, fault, stage, phased):
    records = collect(write_spec(fault=fault, phased=phased))

    assert len(records) == PLANNED - 1
    assert ("a", "b", 0) not in {(r["judge"]["name"], r["evaluee"]["name"], r["scenario_index"]) for r in records}
    report = json.loads((tmp_path / "omitted_samples.json").read_text())
    assert (report["planned"], report["completed"], report["omitted"]) == (PLANNED, PLANNED - 1, 1)
    [row] = report["samples"]
    assert (row["scenario_index"], row["judge"], row["evaluee"]) == (0, "a", "b")
    assert (row["stage"], row["reason"]) == (stage, "invalid_response_after_retries")
    assert len((tmp_path / "evaluations.jsonl").read_text().splitlines()) == PLANNED - 1

    # A finished run with omissions short-circuits instead of looking incomplete.
    assert collect(write_spec(fault=fault, phased=phased)) == records


def test_reexport_follows_the_run_policy(write_spec, tmp_path):
    from inspect_ai.log import list_eval_logs
    from inspect_pipeline.export import export_log

    records = collect(write_spec(fault="rating"))
    (tmp_path / "evaluations.jsonl").unlink()
    (tmp_path / "omitted_samples.json").unlink()

    [log] = list_eval_logs(str(tmp_path / "inspect_logs"))
    exported, _ = export_log(log.name, evaluations_path=tmp_path / "evaluations.jsonl")
    assert exported == records
    assert json.loads((tmp_path / "omitted_samples.json").read_text())["omitted"] == 1

    with pytest.raises(RuntimeError, match="failure_policy=strict"):
        export_log(log.name, evaluations_path=tmp_path / "again.jsonl", failure_policy="strict")


@pytest.mark.parametrize("phased", [False, True])
def test_strict_run_writes_a_coverage_file_over_a_stale_one(write_spec, tmp_path, phased):
    stale = {"planned": 99, "completed": 90, "omitted": 9, "samples": [{"judge": "old"}]}
    (tmp_path / "omitted_samples.json").write_text(json.dumps(stale))

    records = collect(write_spec(policy="strict", phased=phased))

    assert len(records) == PLANNED
    assert json.loads((tmp_path / "omitted_samples.json").read_text()) == {
        "planned": PLANNED, "completed": PLANNED, "omitted": 0, "samples": []}


@pytest.mark.parametrize("phased", [False, True])
def test_strict_refuses_an_invalid_judgment(write_spec, tmp_path, phased):
    with pytest.raises(RuntimeError) as error:
        collect(write_spec(fault="rating", policy="strict", phased=phased))

    message = str(error.value)
    assert "nothing exported" in message and "Models involved: a, b" in message
    assert "judge=a evaluee=b" in message
    assert not (tmp_path / "evaluations.jsonl").exists()
    assert not (tmp_path / "omitted_samples.json").exists()


@pytest.mark.parametrize("phased", [False, True])
def test_provider_errors_stop_the_run_when_omitting(write_spec, tmp_path, phased):
    with pytest.raises(RuntimeError) as error:
        collect(write_spec(fault="error", phased=phased))

    message = str(error.value)
    assert "for reasons other than an invalid judgment" in message
    assert "401 Unauthorized" in message and "judge=a evaluee=b" in message
    assert not (tmp_path / "evaluations.jsonl").exists()


@pytest.mark.parametrize("phased", [False, True])
@pytest.mark.parametrize("policy", ["strict", "omit_invalid_judgments"])
def test_failed_responses_stop_the_run(write_spec, tmp_path, policy, phased):
    with pytest.raises(RuntimeError) as error:
        collect(write_spec(fault="response", policy=policy, phased=phased))

    message = str(error.value)
    assert "nothing exported" in message and "evaluee=b" in message
    assert "empty completion" in message
    if phased:
        assert "response(s) failed; no judgments run" in message and "Models involved: b" in message
    assert not (tmp_path / "evaluations.jsonl").exists()
    assert not (tmp_path / "omitted_samples.json").exists()


@pytest.mark.parametrize("policy", ["strict", "omit_invalid_judgments"])
def test_phased_run_missing_a_judge_log_is_refused(write_spec, tmp_path, monkeypatch, policy):
    import inspect_pipeline.collect as collect_module

    run_phased = collect_module._run_phased
    monkeypatch.setattr(collect_module, "_run_phased", lambda *args: run_phased(*args)[:-1])
    with pytest.raises(RuntimeError, match=f"incomplete run: {PLANNED} judgment"):
        collect(write_spec(policy=policy, phased=True))
    assert not (tmp_path / "evaluations.jsonl").exists()


@pytest.mark.parametrize("policy", ["strict", "omit_invalid_judgments"])
def test_single_task_log_short_of_the_plan_is_refused(write_spec, tmp_path, monkeypatch, policy):
    import inspect_pipeline.collect as collect_module

    estimate = collect_module.write_call_estimate
    monkeypatch.setattr(collect_module, "write_call_estimate",
                        lambda *args: {**estimate(*args), "rating_tasks": PLANNED + 1})
    with pytest.raises(RuntimeError, match=f"incomplete run: {PLANNED + 1} judgment"):
        collect(write_spec(policy=policy))
    assert not (tmp_path / "evaluations.jsonl").exists()


def test_every_judgment_invalid_exports_nothing(write_spec, tmp_path):
    with pytest.raises(RuntimeError, match="every judgment in the log was invalid"):
        collect(write_spec(fault="all_ratings"))
    assert not (tmp_path / "evaluations.jsonl").exists()


def test_unknown_policy_is_rejected(write_spec):
    with pytest.raises(ValueError, match="failure_policy"):
        collect(write_spec(policy="omit_everything"))
