"""Cached responses and model lifecycle on the Inspect engine (issue #31, offline, mockllm).

Response tasks generate only the cells the cache lacks, keep cached text exactly, refuse a
cache made for other scenario text and write new responses back. Between phases only local
servers are closed; an API model's client stays usable for the judgments that follow.
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
PLANNED = SCENARIOS * len(NICKS) ** 2

# Every response call is logged to CALLS. Closing a model logs it to CLOSED; a closed API
# model then fails like a closed HTTP client, while a LOCAL one restarts, as vLLM does.
SPEC_TEMPLATE = '''
import re
from inspect_ai.model import ModelOutput, get_model

CALLS, CLOSED, LOCAL = {calls!r}, {closed!r}, {local!r}


def _closed():
    try:
        return open(CLOSED).read().split()
    except FileNotFoundError:
        return []


def _mock(nick):
    def outputs(input, tools, tool_choice, config):
        system, user = input[0].text, input[1].text
        if system.startswith("Without making any mention"):
            scenario = re.search(r"Scenario number (\\d)", user).group(1)
            open(CALLS, "a").write(f"{{nick}} {{scenario}}\\n")
            return ModelOutput.from_content(model="mockllm/model", content=f"[{{nick}}] answer {{scenario}}")
        if system.startswith("You are acting as a judge. Carefully consider the following response"):
            return ModelOutput.from_content(model="mockllm/model", content="reflection")
        return ModelOutput.from_content(model="mockllm/model", content="<criterion_1_rating>7</criterion_1_rating>")

    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    model._eb_local = nick in LOCAL
    generate = model.api.generate

    async def aclose():
        open(CLOSED, "a").write(nick + "\\n")

    async def guarded(*args, **kwargs):
        if nick in _closed() and nick not in LOCAL:
            raise RuntimeError("Cannot send a request, as the client has been closed")
        return await generate(*args, **kwargs)

    model.api.aclose, model.api.generate = aclose, guarded
    return model


RUN_SPEC = {{
    "name": "cachetest",
    "verbose": False,
    "models": {{nick: _mock(nick) for nick in {nicks!r}}},
    "evaluation": {{"mode": "direct_rating"}},
    "dataset": {{"path": {scenarios_path!r}, "start": 0, "count": {scenarios}}},
    "constitution": {{"path": {constitution_path!r}, "num_criteria": 1}},
    "collection": {{
        "enabled": True,
        "sampler_mode": "all_to_all",
        "cached_responses_path": {cache_path!r},
        "inspect": {{"cache": False, "display": "none", "phased": {phased!r}}},
    }},
    "training": {{"enabled": False}},
}}
'''


def scenario(i):
    return f"Scenario number {i}: what now?"


@pytest.fixture
def run(tmp_path):
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps([scenario(i) for i in range(SCENARIOS)]))
    constitution_path = tmp_path / "constitution.json"
    constitution_path.write_text(json.dumps(["Criterion 1: prefer the kinder response"]))

    def write(*, phased, local=(), cache=None):
        if cache is not None:
            (tmp_path / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in cache))
        path = tmp_path / "spec.py"
        path.write_text(SPEC_TEMPLATE.format(
            calls=str(tmp_path / "calls.txt"), closed=str(tmp_path / "closed.txt"), local=list(local),
            nicks=NICKS, scenarios=SCENARIOS, scenarios_path=str(scenarios_path),
            constitution_path=str(constitution_path), cache_path=str(tmp_path / "responses.jsonl"),
            phased=phased,
        ))
        return str(path)

    return tmp_path, write


def lines(path):
    return sorted(path.read_text().split("\n")[:-1]) if path.exists() else []


def collect(spec):
    from inspect_pipeline.collect import collect_direct_ratings_inspect

    return collect_direct_ratings_inspect(spec)


# Scenario 0 is cached for both models, scenario 1 only for "a".
PARTIAL = [
    {"scenario": scenario(0), "scenario_index": 0, "responses": {"a": "CACHED a 0", "b": "CACHED b 0"}},
    {"scenario": scenario(1), "scenario_index": 1, "responses": {"a": "CACHED a 1"}},
]


@pytest.mark.parametrize("phased", [False, True])
def test_a_partial_cache_generates_only_the_missing_responses(run, phased):
    tmp_path, write = run
    records = collect(write(phased=phased, cache=PARTIAL))

    assert lines(tmp_path / "calls.txt") == ["b 1"]
    estimate = json.loads((tmp_path / "direct_call_estimate.json").read_text())
    assert estimate["response_tasks"] == 1
    assert len(records) == PLANNED
    responses = {(r["scenario_index"], r["evaluee"]["name"]): r["response"] for r in records}
    assert responses == {(0, "a"): "CACHED a 0", (0, "b"): "CACHED b 0",
                         (1, "a"): "CACHED a 1", (1, "b"): "[b] answer 1"}

    # The cache keeps its rows and gains scenario 1, now complete.
    rows = [json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()]
    assert rows[:2] == PARTIAL
    assert [row["scenario_index"] for row in rows[2:]] == [1]
    assert rows[-1]["responses"] == {"a": "CACHED a 1", "b": "[b] answer 1"}


@pytest.mark.parametrize("phased", [False, True])
def test_a_cache_for_other_scenario_text_is_refused_before_any_call(run, phased):
    tmp_path, write = run
    cache = [{**PARTIAL[0], "scenario": "Scenario number 0: something else?"}]
    with pytest.raises(ValueError, match="other scenario text at index 0"):
        collect(write(phased=phased, cache=cache))
    assert not (tmp_path / "calls.txt").exists()


@pytest.mark.parametrize("local", [(), ("a",)], ids=["api", "mixed"])
def test_phased_run_closes_only_local_models(run, local, monkeypatch):
    import inspect_pipeline.collect as collect_module

    # Mock models are Model objects, which never count as local; mark them as the spec says.
    monkeypatch.setattr(collect_module, "_is_local", lambda value: getattr(value, "_eb_local", False))
    tmp_path, write = run
    records = collect(write(phased=True, local=local))

    assert len(records) == PLANNED
    # A local model is released after each of its phases; an API model never is.
    assert lines(tmp_path / "closed.txt") == sorted(local * 2)
