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


def test_phased_matches_single_task(run_dir):
    """A phased run must produce exactly the records an edge-per-sample run does.

    Phasing exists only to keep one model resident at a time; it must not change
    the protocol or the output.
    """

    tmp_path, write_spec = run_dir
    spec_path = write_spec(sampler_mode="all_to_all")

    from inspect_pipeline.collect import collect_direct_ratings_inspect

    single = collect_direct_ratings_inspect(str(spec_path))

    # Re-run the same spec through the phased path.
    (tmp_path / "evaluations.jsonl").unlink()
    spec_text = spec_path.read_text().replace(
        '"inspect": {"cache": False, "display": "none"}',
        '"inspect": {"cache": False, "display": "none", "phased": True}',
    )
    spec_path.write_text(spec_text, encoding="utf-8")
    phased = collect_direct_ratings_inspect(str(spec_path))

    assert len(phased) == len(single)

    def key(r):
        return (r["scenario_index"], r["judge"]["index"], r["evaluee"]["index"])

    def normalize(r):
        # The mock stamps a call counter into each response to prove pooling;
        # the two paths generate in a different order, so only the identity of
        # the responding model is comparable across them.
        out = dict(r)
        out["response"] = r["response"].split("#", 1)[0]
        return out

    for a, b in zip(sorted(single, key=key), sorted(phased, key=key)):
        assert normalize(a) == normalize(b), f"phased record differs at {key(a)}"

    info = json.loads((tmp_path / "inspect_run.json").read_text())
    assert info["log_file"].endswith(".eval")
    # One log per judge.
    assert len(info.get("log_files", [info["log_file"]])) == len(NICKS)


def test_extend_adds_a_model(run_dir, monkeypatch):
    """Adding a model fills a new row and column without redoing the rest."""

    import numpy as np
    from inspect_ai.model import ModelOutput, get_model

    from inspect_pipeline.collect import collect_direct_ratings_inspect
    from inspect_pipeline.extend import (
        count_matrix, detect_mode, extend_run, model_names, plan_addition,
        projected_counts,
    )

    tmp_path, write_spec = run_dir
    spec_path = write_spec(sampler_mode="all_to_all")
    base = collect_direct_ratings_inspect(str(spec_path))
    mode = detect_mode(base)
    before = np.array(count_matrix(base, mode, len(model_names(base, mode))))

    # The plan is decided before any inference, so it can be checked on its own.
    plan = plan_addition(base, "delta", include_self=True, seed=7)
    after_planned = np.array(projected_counts(base, plan))
    assert after_planned.shape == (before.shape[0] + 1,) * 2
    # Existing counts are untouched: no work is repeated.
    assert (after_planned[:-1, :-1] == before).all()
    # Edges where the new model judges reuse a response that already exists.
    assert any(e.response is not None for e in plan.as_judge)

    def outputs(input, tools, tool_choice, config):
        system = input[0].text if input else ""
        if system.startswith("Without making any mention"):
            content = "[delta#1] answers"
        elif system.startswith(
            "You are acting as a judge. Carefully consider the following response"
        ):
            content = "[delta] reflection"
        else:
            content = "\n".join(
                f"<criterion_{i}_rating>5</criterion_{i}_rating>"
                for i in range(1, NUM_CRITERIA + 1)
            )
        return ModelOutput.from_content(model="mockllm/model", content=content)

    delta = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    result = extend_run(str(spec_path), "delta", delta, include_self=True, seed=7)

    merged = [
        json.loads(line)
        for line in (tmp_path / "evaluations.jsonl").read_text().splitlines()
    ]
    assert len(merged) == len(base) + result["collected"]

    names = model_names(merged, mode)
    assert names[-1] == "delta"
    after = np.array(count_matrix(merged, mode, len(names)))
    assert (after[:-1, :-1] == before).all(), "existing cells must not change"
    assert after[:, -1].sum() > 0 and after[-1, :].sum() > 0, "new row and column filled"

    # Every new record is well formed for the analysis layer.
    new = [r for r in merged if "delta" in (r["judge"]["name"], r["evaluee"]["name"])]
    assert len(new) == result["collected"]
    for r in new:
        assert [e["criterion_index"] for e in r["ratings"]] == list(range(NUM_CRITERIA))


def test_pairwise_prompts_match_legacy():
    """New pairwise records must be interchangeable with the legacy ones."""

    import pipeline.eval.criteria_collectors as cc
    from inspect_pipeline.pairwise import build_comparison_user, build_reflection_user

    captured = []

    def fake(model_name, messages, max_tokens=None, **kw):
        captured.append(messages)
        n = len(captured)
        if n <= 2:
            return f"RESP{n}"
        if n <= 4:
            return f"REFL{n - 2}"
        return "<criterion_1_choice>1</criterion_1_choice>"

    original = cc.get_model_response
    cc.get_model_response = fake
    try:
        cc.collect_group_criteria_evaluations(
            criteria=["c1"], scenario="SCENARIO", scenario_index=0,
            models={"a": "x/a", "b": "x/b"}, judge_idx=0, eval_idxs=[0, 1],
        )
    finally:
        cc.get_model_response = original

    assert captured[2][1]["content"] == build_reflection_user("c1", "SCENARIO", "RESP1")
    assert captured[4][1]["content"] == build_comparison_user(
        "c1", "SCENARIO", "RESP1", "REFL1", "RESP2", "REFL2"
    )


def test_pairwise_extend_collects(tmp_path):
    """A pairwise run can be extended on the Inspect engine."""

    import numpy as np
    from inspect_ai.model import ModelOutput, get_model

    from inspect_pipeline.extend import (
        PAIRWISE, count_matrix, extend_run, model_names, plan_addition,
    )

    names = ["a", "b", "c"]
    scenarios = [f"Scenario {i}: what do you do?" for i in range(4)]
    records = []
    for s, text in enumerate(scenarios):
        for ji, j in enumerate(names):
            for x in range(len(names)):
                for y in range(x + 1, len(names)):
                    records.append({
                        "constitution": "c1", "scenario": text, "scenario_index": s,
                        "eval1": x, "eval1_name": names[x],
                        "eval1 response": f"{names[x]} on {s}", "eval1 reflection": "r",
                        "eval2": y, "eval2_name": names[y],
                        "eval2 response": f"{names[y]} on {s}", "eval2 reflection": "r",
                        "judge": ji, "judge_name": j,
                        "judge response": "<criterion_1_choice>1</criterion_1_choice>",
                    })

    constitution = tmp_path / "constitution.json"
    constitution.write_text(json.dumps(["c1"]), encoding="utf-8")
    scen_file = tmp_path / "scenarios.json"
    scen_file.write_text(json.dumps(scenarios), encoding="utf-8")
    evals = tmp_path / "evaluations.jsonl"
    evals.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    spec = tmp_path / "spec.py"
    spec.write_text(f"""
from inspect_ai.model import ModelOutput, get_model


def _mock(nick):
    def outputs(input, tools, tool_choice, config):
        system = input[0].text if input else ""
        if system.startswith("Without making any mention"):
            c = f"[{{nick}}] answers"
        elif system.startswith(
            "You are acting as a judge. Carefully consider the following response"
        ):
            c = f"[{{nick}}] reflection"
        else:
            c = "<criterion_1_choice>1</criterion_1_choice>"
        return ModelOutput.from_content(model="mockllm/model", content=c)

    return get_model("mockllm/model", custom_outputs=outputs, memoize=False)


RUN_SPEC = {{
    "name": "pw",
    "models": {{n: _mock(n) for n in {names!r}}},
    "evaluation": {{"mode": "pairwise_btd"}},
    "dataset": {{"path": {str(scen_file)!r}, "count": {len(scenarios)}}},
    "constitution": {{"path": {str(constitution)!r}, "num_criteria": 1}},
    "collection": {{"evaluations_path": {str(evals)!r},
                    "inspect": {{"cache": False, "display": "none"}}}},
    "training": {{"enabled": False}},
}}
""", encoding="utf-8")

    before = np.array(count_matrix(records, PAIRWISE, len(names)))
    plan = plan_addition(records, "d", seed=3)
    assert plan.mode == PAIRWISE
    assert all(e.opponent for e in plan.edges), "every comparison needs an opponent"

    def outputs(input, tools, tool_choice, config):
        system = input[0].text if input else ""
        if system.startswith("Without making any mention"):
            content = "[d] answers"
        elif system.startswith(
            "You are acting as a judge. Carefully consider the following response"
        ):
            content = "[d] reflection"
        else:
            content = "<criterion_1_choice>1</criterion_1_choice>"
        return ModelOutput.from_content(model="mockllm/model", content=content)

    d = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    result = extend_run(str(spec), "d", d, seed=3)

    merged = [json.loads(line) for line in evals.read_text().splitlines()]
    assert len(merged) == len(records) + result["collected"]

    names_after = model_names(merged, PAIRWISE)
    assert names_after[-1] == "d"
    after = np.array(count_matrix(merged, PAIRWISE, len(names_after)))
    assert after[:, -1].sum() > 0 and after[-1, :].sum() > 0

    new = [r for r in merged if "d" in (r["judge_name"], r["eval1_name"], r["eval2_name"])]
    for r in new:
        assert r["judge response"], "a comparison must record a verdict"
        assert r["eval1 response"] and r["eval2 response"]
