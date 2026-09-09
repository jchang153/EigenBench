"""End-to-end tests for the Inspect AI direct-rating engine (offline, mockllm).

Covers the native CLI path (`inspect eval` + export) and the programmatic
driver, then feeds the exported evaluations.jsonl through the legacy analysis.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspect_pipeline.model_mapping import to_inspect_model  # noqa: E402

NICKS = ["alpha", "beta", "gamma"]
NUM_CRITERIA = 3
SCENARIO_COUNT = 3

# A spec whose models are scripted mockllm instances. Responses embed a
# per-call counter so a shared response can be told apart from a re-generated
# one; ratings are a deterministic function of (judge, evaluee, criterion).
SPEC_TEMPLATE = '''
import sys
sys.path.insert(0, {repo_root!r})

import re
from inspect_ai.model import ModelOutput, get_model

NICKS = {nicks!r}
NUM_CRITERIA = {num_criteria}
_calls = {{"n": 0}}


def rating_value(judge, evaluee, criterion_one_based):
    return (ord(judge[0]) * 3 + ord(evaluee[0]) * 5 + criterion_one_based * 7) % 10 + 1


def _mock(nick):
    def outputs(input, tools, tool_choice, config):
        system = input[0].text if input else ""
        user = input[1].text if len(input) > 1 else ""
        if system.startswith("Without making any mention"):
            _calls["n"] += 1
            content = "[{{}}#{{}}] answers".format(nick, _calls["n"])
        elif system.startswith(
            "You are acting as a judge. Carefully consider the following response"
        ):
            content = "[{{}}] reflection".format(nick)
        elif system.startswith("You are acting as a judge. Carefully consider how well"):
            match = re.search(r"<response>\\n\\[(\\w+)#", user)
            assert match, "no evaluee in rating prompt"
            evaluee = match.group(1)
            content = "\\n".join(
                "<criterion_{{0}}_rating>{{1}}</criterion_{{0}}_rating>".format(
                    i, rating_value(nick, evaluee, i)
                )
                for i in range(1, NUM_CRITERIA + 1)
            )
        else:
            raise AssertionError("unexpected system prompt: " + system[:80])
        return ModelOutput.from_content(model="mockllm/model", content=content)

    return get_model("mockllm/model", custom_outputs=outputs, memoize=False)


RUN_SPEC = {{
    "name": "mocktest",
    "verbose": False,
    "models": {{nick: _mock(nick) for nick in NICKS}},
    "evaluation": {{"mode": "direct_rating"}},
    "dataset": {{"path": {scenarios_path!r}, "start": 0, "count": {scenario_count}}},
    "constitution": {{"path": {constitution_path!r}, "num_criteria": {num_criteria}}},
    "collection": {{
        "enabled": True,
        "sampler_mode": {sampler_mode!r},
        "inspect": {{"cache": False, "display": "none"}},
    }},
    "training": {{"enabled": True, "bootstrap": {{"enabled": False}}}},
}}
'''


def rating_value(judge: str, evaluee: str, criterion_one_based: int) -> int:
    return (ord(judge[0]) * 3 + ord(evaluee[0]) * 5 + criterion_one_based * 7) % 10 + 1


@pytest.fixture
def run_dir(tmp_path):
    """A run folder with scenarios, a constitution, and a mock-model spec."""

    scenarios = [f"Scenario number {i}: what do you do?" for i in range(6)]
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")

    constitution = [
        f"Criterion {i} for Testing: prefer the response that is good in way {i}"
        for i in range(1, NUM_CRITERIA + 1)
    ]
    constitution_path = tmp_path / "constitution.json"
    constitution_path.write_text(json.dumps(constitution), encoding="utf-8")

    def write_spec(sampler_mode="all_to_all"):
        spec_path = tmp_path / "spec.py"
        spec_path.write_text(
            SPEC_TEMPLATE.format(
                repo_root=str(REPO_ROOT),
                nicks=NICKS,
                num_criteria=NUM_CRITERIA,
                scenarios_path=str(scenarios_path),
                constitution_path=str(constitution_path),
                scenario_count=SCENARIO_COUNT,
                sampler_mode=sampler_mode,
            ),
            encoding="utf-8",
        )
        return spec_path

    return tmp_path, write_spec


def _check_records(records, *, expected_edges, sampler_mode):
    assert len(records) == expected_edges
    seen_edges = set()
    responses_by_key = {}
    for record in records:
        assert list(record.keys()) == [
            "schema_version",
            "record_type",
            "constitution",
            "scenario",
            "scenario_index",
            "judge",
            "evaluee",
            "sampling",
            "response",
            "reflection",
            "judgment_raw",
            "ratings",
        ]
        assert record["schema_version"] == 2
        assert record["record_type"] == "direct_rating"
        assert record["sampling"]["mode"] == sampler_mode
        judge, evaluee = record["judge"]["name"], record["evaluee"]["name"]
        assert record["response"].startswith(f"[{evaluee}#")
        assert record["reflection"] == f"[{judge}] reflection"
        assert [entry["criterion_index"] for entry in record["ratings"]] == list(
            range(NUM_CRITERIA)
        )
        for entry in record["ratings"]:
            assert entry["rating"] == rating_value(
                judge, evaluee, entry["criterion_index"] + 1
            )
        edge = (record["scenario_index"], record["judge"]["index"], record["evaluee"]["index"])
        assert edge not in seen_edges
        seen_edges.add(edge)
        responses_by_key.setdefault(
            (record["scenario_index"], evaluee), set()
        ).add(record["response"])

    # ResponsePool: every judge of a response saw the identical text, and each
    # (scenario, evaluee) response was generated exactly once.
    assert all(len(texts) == 1 for texts in responses_by_key.values())
    all_responses = {next(iter(texts)) for texts in responses_by_key.values()}
    assert len(all_responses) == SCENARIO_COUNT * len(NICKS)
    return seen_edges


def test_cli_inspect_eval_and_export(run_dir):
    """The native path: inspect eval -> export_evaluations.py."""

    tmp_path, write_spec = run_dir
    spec_path = write_spec(sampler_mode="all_to_all")
    log_dir = tmp_path / "logs"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            "inspect_pipeline/eigenbench.py",
            "-T",
            f"spec={spec_path}",
            "--log-dir",
            str(log_dir),
            "--display",
            "none",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    evaluations_path = tmp_path / "exported.jsonl"
    export = subprocess.run(
        [
            sys.executable,
            "scripts/export_evaluations.py",
            str(log_dir),
            "-o",
            str(evaluations_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert export.returncode == 0, export.stdout + export.stderr

    records = [
        json.loads(line)
        for line in evaluations_path.read_text(encoding="utf-8").splitlines()
    ]
    # all_to_all with include_self: S * M * M edges
    _check_records(
        records,
        expected_edges=SCENARIO_COUNT * len(NICKS) * len(NICKS),
        sampler_mode="all_to_all",
    )


def test_programmatic_run_and_analysis(run_dir):
    """The wrapper path: collect -> export -> legacy trust-matrix analysis."""

    tmp_path, write_spec = run_dir
    spec_path = write_spec(sampler_mode="balanced_unique_judge")

    from inspect_pipeline.collect import collect_direct_ratings_inspect

    records = collect_direct_ratings_inspect(str(spec_path))

    # balanced_unique_judge with redundancy 1: S * M edges
    _check_records(
        records,
        expected_edges=SCENARIO_COUNT * len(NICKS),
        sampler_mode="balanced_unique_judge",
    )

    evaluations_path = tmp_path / "evaluations.jsonl"
    assert evaluations_path.exists()
    assert len(evaluations_path.read_text().splitlines()) == len(records)

    estimate = json.loads((tmp_path / "direct_call_estimate.json").read_text())
    assert estimate["rating_tasks"] == SCENARIO_COUNT * len(NICKS)

    # rerun short-circuits on the completed file
    assert len(collect_direct_ratings_inspect(str(spec_path))) == len(records)

    from pipeline.train.direct_analysis import run_direct_analysis

    run_direct_analysis(
        records=records,
        models={nick: f"mock/{nick}" for nick in NICKS},
        num_criteria=NUM_CRITERIA,
        evaluation_cfg={"mode": "direct_rating", "direct_rating": {"include_self": True}},
        training_cfg={"bootstrap": {"enabled": False}},
        output_root=tmp_path / "train",
        collection_cfg={"sampler_mode": "balanced_unique_judge"},
        verbose=False,
    )
    out_dir = tmp_path / "train" / "direct_rating"
    summary = json.loads((out_dir / "summary.json").read_text())
    assert {entry["model_name"] for entry in summary} == set(NICKS)
    assert (out_dir / "trust_matrix.csv").exists()


def test_model_mapping():
    ref = to_inspect_model("anthropic/claude-sonnet-4")
    assert ref.name == "openrouter/anthropic/claude-sonnet-4"
    assert not ref.is_local

    assert to_inspect_model("inspect:mockllm/model").name == "mockllm/model"

    ref = to_inspect_model("hf_local:Qwen/Qwen2.5-7B-Instruct")
    assert ref.name == "vllm/Qwen/Qwen2.5-7B-Instruct"
    assert ref.is_local

    ref = to_inspect_model(
        {
            "provider": "hf_local",
            "kind": "lora",
            "repo_id": "someorg/some-adapter",
            "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
            "revision": "abc123",
        }
    )
    assert ref.name == "vllm/Qwen/Qwen2.5-7B-Instruct:someorg/some-adapter@abc123"
    assert ref.is_local

    with pytest.raises(ValueError):
        to_inspect_model("inspect:no-slash")
    with pytest.raises(ValueError):
        to_inspect_model("")
