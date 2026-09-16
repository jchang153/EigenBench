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


@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("scenario_count", [3, 200])
@pytest.mark.parametrize("sampler_mode", ["balanced_unique_judge", "all_to_all"])
def test_direct_extension_covers_each_scenario_once(
    include_self, scenario_count, sampler_mode
):
    from collections import Counter

    from inspect_pipeline.extend import plan_addition, projected_counts
    from pipeline.eval.direct_rating import build_direct_assignments

    names = [f"model_{i}" for i in range(8)]
    scenarios = [(i, f"Scenario {i}") for i in range(scenario_count)]
    assignments = build_direct_assignments(
        scenarios, dict.fromkeys(names, "mockllm/model"),
        sampler_mode=sampler_mode, include_self=include_self, sampler_seed=42,
    )
    records = [
        {
            "record_type": "direct_rating",
            "scenario_index": a["scenario_index"],
            "scenario": a["scenario"],
            "judge": {"index": a["judge_idx"], "name": a["judge_nick"]},
            "evaluee": {"index": e, "name": names[e]},
            "response": f"saved response {a['scenario_index']} {names[e]}",
        }
        for a in assignments for e in a["eval_idxs"]
    ]
    plan = plan_addition(records, "new", include_self=include_self)
    assert plan == plan_addition(records, "new", include_self=include_self)
    assert Counter(e.scenario_index for e in plan.as_evaluee) == {
        i: 1 for i in range(scenario_count)
    }
    assert plan.responses_needed() == set(range(scenario_count))
    assert all(e.evaluee == "new" for e in plan.as_evaluee)
    eligible = names + (["new"] if include_self else [])
    loads = Counter(e.judge for e in plan.as_evaluee)
    assert set(loads) <= set(eligible)
    assert max(loads[j] for j in eligible) - min(loads[j] for j in eligible) <= 1
    identities = [(e.scenario_index, e.judge, e.evaluee) for e in plan.edges]
    assert len(identities) == len(set(identities))
    for e in plan.as_judge:
        assert e.judge == "new" and e.evaluee in names
        assert e.response == f"saved response {e.scenario_index} {e.evaluee}"
    grid = projected_counts(records, plan)
    assert sum(row[-1] for row in grid) == scenario_count
    assert sum(grid[-1]) == plan.targets["row_target"]
    assert plan.summary()["new_responses"] == scenario_count


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
    new_responses = [r for r in new if r["evaluee"]["name"] == "delta"]
    assert len(new_responses) == SCENARIO_COUNT
    assert {r["scenario_index"] for r in new_responses} == set(range(SCENARIO_COUNT))
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


@pytest.mark.parametrize("phased", [False, True])
@pytest.mark.parametrize("new_names, extra_count", [(["d"], 0), (["d", "e"], 2)])
def test_pairwise_extend_collects(tmp_path, new_names, extra_count, phased):
    """A pairwise run can be extended on the Inspect engine."""

    import numpy as np
    from inspect_ai.model import ModelOutput, get_model

    from inspect_pipeline.extend import (
        PAIRWISE, count_matrix, extend_run, model_names, plan_addition,
    )

    names = ["a", "b", "c"]
    scenarios = [f"Scenario {i}: what do you do?" for i in range(6)]
    records = []
    for s, text in enumerate(scenarios[:4]):
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

    if phased:
        with spec.open("a") as handle:
            handle.write('\nRUN_SPEC["collection"]["inspect"]["phased"] = True\n')
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
    result = extend_run(str(spec), new_models=dict.fromkeys(new_names, d),
                        additional_scenarios=extra_count, seed=3)

    merged = [json.loads(line) for line in evals.read_text().splitlines()]
    assert len(merged) == len(records) + result["collected"]

    names_after = model_names(merged, PAIRWISE)
    assert names_after[-len(new_names):] == new_names
    after = np.array(count_matrix(merged, PAIRWISE, len(names_after)))
    assert after[:, -1].sum() > 0 and after[-1, :].sum() > 0

    new = [r for r in merged if "d" in (r["judge_name"], r["eval1_name"], r["eval2_name"])]
    for r in new:
        assert r["judge response"], "a comparison must record a verdict"
        assert r["eval1 response"] and r["eval2 response"]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("new_names", [[], ["delta", "epsilon"]])
def test_extension_spec_models_and_scenarios(run_dir, new_names, native):
    """A new spec extends source records without modifying the source run."""
    from collections import Counter

    from inspect_pipeline.collect import collect_direct_ratings_inspect
    from inspect_pipeline.extend import extend_run
    from inspect_pipeline.export import export_log

    tmp_path, write_spec = run_dir
    base_spec = write_spec(sampler_mode="balanced_unique_judge")
    original = collect_direct_ratings_inspect(str(base_spec))
    source = tmp_path / "evaluations.jsonl"
    before = source.read_bytes()
    expanded = tmp_path / "expanded"
    expanded.mkdir()
    output = expanded / "evaluations.jsonl"
    spec = expanded / "spec.py"
    spec.write_text(SPEC_TEMPLATE.format(
        repo_root=str(REPO_ROOT), nicks=NICKS + new_names, num_criteria=NUM_CRITERIA,
        scenarios_path=str(tmp_path / "scenarios.json"),
        constitution_path=str(tmp_path / "constitution.json"), scenario_count=5,
        sampler_mode="balanced_unique_judge",
    ) + '\nRUN_SPEC["extension"] = {"from_evaluations": "../evaluations.jsonl", '
        '"additional_scenarios": 2}\n')
    # Dry-run through the public command line, with no inference or output writes.
    result = subprocess.run(
        [sys.executable, "scripts/extend_run.py", str(spec), "--dry-run"],
        cwd=REPO_ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["additional_scenarios"] == 2
    assert not output.exists()
    if native:
        result = subprocess.run(
            [sys.executable, "-m", "inspect_ai", "eval", "inspect_pipeline/extend.py",
             "-T", f"spec={spec}", "--log-dir", str(expanded / "inspect_logs"),
             "--display", "none"], cwd=REPO_ROOT, text=True, capture_output=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        log = next((expanded / "inspect_logs").glob("*.eval"))
        rows, _ = export_log(log, evaluations_path=output, append=True)
        summary = {"collected": len(rows)}
    else:
        from scripts.run_collect import main as collect_extension
        summary = collect_extension(str(spec))
    combined = [json.loads(line) for line in output.read_text().splitlines()]
    assert combined[:len(original)] == original
    assert source.read_bytes() == before
    added = combined[len(original):]
    assert len(added) == summary["collected"]
    for name in new_names:
        assert Counter(r["scenario_index"] for r in added if r["evaluee"]["name"] == name) == {
            i: 1 for i in range(5)
        }
    for scenario in (3, 4):
        fresh = [r for r in added if r["scenario_index"] == scenario]
        assert Counter(r["evaluee"]["name"] for r in fresh) == dict.fromkeys(NICKS + new_names, 1)
        assert Counter(r["judge"]["name"] for r in fresh) == dict.fromkeys(NICKS + new_names, 1)
    saved = {(r["scenario_index"], r["evaluee"]["name"]): r["response"] for r in original}
    for r in added:
        key = r["scenario_index"], r["evaluee"]["name"]
        if key in saved:
            assert r["response"] == saved[key]
    identities = [(r["scenario_index"], r["judge"]["index"], r["evaluee"]["index"])
                  for r in combined]
    assert len(identities) == len(set(identities))
    expected_order = NICKS + new_names
    for r in combined:
        for role in ("judge", "evaluee"):
            assert expected_order[r[role]["index"]] == r[role]["name"]
    with pytest.raises(ValueError, match="output already exists"):
        extend_run(str(spec))
    # Native Inspect export also includes the baseline when writing a fresh output.
    log = next((expanded / "inspect_logs").glob("*.eval"))
    exported = expanded / "exported.jsonl"
    with pytest.raises(ValueError, match="require --append"):
        export_log(log, evaluations_path=exported)
    export_log(log, evaluations_path=exported, append=True)
    assert [json.loads(line) for line in exported.read_text().splitlines()] == combined

    # A missing or modified baseline must never produce an extension-only output.
    rejected = expanded / "rejected.jsonl"
    source.unlink()
    with pytest.raises(ValueError, match="source is missing"):
        export_log(log, evaluations_path=rejected, append=True)
    assert not rejected.exists()
    changed = [dict(r) for r in original]
    changed[0]["scenario"] = "changed after collection"
    source.write_text("\n".join(json.dumps(r) for r in changed) + "\n")
    with pytest.raises(ValueError, match="source changed"):
        export_log(log, evaluations_path=rejected, append=True)
    assert not rejected.exists()
    source.write_bytes(before)
    # Formatting changes are harmless, but old logs without a fingerprint and
    # successful logs containing only a subset of the plan must be rejected.
    from inspect_pipeline.export import load_log
    source.write_text("\n".join(json.dumps(r, sort_keys=True) for r in original) + "\n")
    export_log(log, evaluations_path=expanded / "reformatted.jsonl", append=True)
    source.write_bytes(before)
    parsed_log = load_log(log)
    fingerprint = parsed_log.eval.metadata["eigenbench"].pop("extension_source_sha256")
    with pytest.raises(ValueError, match="no source fingerprint"):
        export_log(parsed_log, evaluations_path=rejected, append=True)
    parsed_log.eval.metadata["eigenbench"]["extension_source_sha256"] = fingerprint
    parsed_log.samples = parsed_log.samples[:-1]
    with pytest.raises(ValueError, match="complete plan"):
        export_log(parsed_log, evaluations_path=rejected, append=True)
    assert not rejected.exists()


def test_joint_model_planner_and_pairwise_analysis():
    from collections import Counter

    from inspect_pipeline.extend import plan_additions, projected_counts
    from pipeline.utils.comparisons import handle_inconsistencies_with_ties_criteria

    names = ["a", "b", "c"]
    records = [
        {"judge": j, "judge_name": names[j], "eval1": a, "eval1_name": names[a],
         "eval2": b, "eval2_name": names[b], "scenario_index": s, "scenario": f"s{s}",
         "eval1 response": f"{a}-{s}", "eval2 response": f"{b}-{s}"}
        for s in range(4) for j in range(3) for a in range(3) for b in range(3) if a != b
    ]
    plan = plan_additions(records, ["d", "e"], new_scenarios={4: "s4", 5: "s5"})
    assert len(projected_counts(records, plan)) == 5
    index = {n: i for i, n in enumerate(names + ["d", "e"])}
    keys = [(e.scenario_index, e.judge, e.evaluee, e.opponent) for e in plan.edges]
    assert len(keys) == len(set(keys))
    for s, j, a, b in keys:
        assert (s, j, b, a) in keys
    rows = [[0, e.scenario_index, index[e.judge], index[e.evaluee], index[e.opponent],
             1 if index[e.evaluee] < index[e.opponent] else 2] for e in plan.edges]
    cleaned = handle_inconsistencies_with_ties_criteria(rows)
    assert Counter(map(tuple, cleaned)) == Counter(map(tuple, rows))
    for s in (4, 5):
        assert {e.evaluee for e in plan.edges if e.scenario_index == s} == set(index)
    assert all(any(e.judge == n for e in plan.edges) for n in ("d", "e"))


def test_extension_rejects_incompatible_inputs(run_dir):
    from inspect_pipeline.collect import collect_direct_ratings_inspect
    from inspect_pipeline.extend import extend_run, plan_additions

    tmp_path, write_spec = run_dir
    spec = write_spec(sampler_mode="balanced_unique_judge")
    records = collect_direct_ratings_inspect(str(spec))
    with pytest.raises(ValueError, match="unique"):
        plan_additions(records, ["delta", "delta"])
    with pytest.raises(ValueError, match="overlap"):
        plan_additions(records, ["delta"], new_scenarios={0: "different"})
    before = (tmp_path / "evaluations.jsonl").read_bytes()
    with pytest.raises(ValueError, match="unused scenarios"):
        extend_run(str(spec), new_models={"delta": "mockllm/model"}, additional_scenarios=2)
    with pytest.raises(ValueError, match="already appear"):
        extend_run(str(spec), new_models={"alpha": "mockllm/model"})
    assert (tmp_path / "evaluations.jsonl").read_bytes() == before


@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("new_count", [2, 12])
def test_many_new_models_share_one_population(include_self, new_count):
    from collections import Counter
    from inspect_pipeline.extend import plan_additions, projected_counts

    names = ["a", "b", "c"]
    records = [
        {"record_type": "direct_rating", "scenario_index": s, "scenario": str(s),
         "judge": {"index": (i + 1) % 3, "name": names[(i + 1) % 3]},
         "evaluee": {"index": i, "name": n}, "response": f"{n}-{s}"}
        for s in range(60) for i, n in enumerate(names)
    ]
    new = [f"new{i}" for i in range(new_count)]
    plan = plan_additions(records, new, include_self=include_self)
    for name in new:
        edges = [e for e in plan.as_evaluee if e.evaluee == name]
        assert Counter(e.scenario_index for e in edges) == dict.fromkeys(range(60), 1)
        eligible = [n for n in names + new if include_self or n != name]
        loads = Counter(e.judge for e in edges)
        assert set(loads) == set(eligible)
        assert max(loads.values()) - min(loads.values()) <= 1
    if not include_self:
        assert all(e.judge != e.evaluee for e in plan.edges)
    grid = projected_counts(records, plan)
    assert all(sum(row) == 60 for row in grid[len(names):])
    assert plan.summary()["new_responses"] == 60 * new_count


def test_pairwise_strict_export_rejects_cancelled_and_missing_content():
    from types import SimpleNamespace
    from inspect_pipeline.pairwise import records_from_pairwise_log

    sample = SimpleNamespace(id="partial", error=None, metadata={}, store={})
    with pytest.raises(RuntimeError, match="did not complete"):
        records_from_pairwise_log(SimpleNamespace(status="cancelled", samples=[sample]), ["c1"])
    with pytest.raises(RuntimeError, match="missing generated content"):
        records_from_pairwise_log(SimpleNamespace(status="success", samples=[sample]), ["c1"])


def test_phased_extension_matches_single_and_closes_models(run_dir, monkeypatch):
    import re
    import runpy
    from inspect_pipeline.collect import collect_direct_ratings_inspect
    from inspect_pipeline.extend import extend_run, extend_model
    from inspect_pipeline.export import export_log
    import inspect_pipeline.extension_phased as phased

    tmp_path, write_spec = run_dir
    spec = write_spec(sampler_mode="balanced_unique_judge")
    collect_direct_ratings_inspect(str(spec))
    target = tmp_path / "evaluations.jsonl"
    baseline = target.read_bytes()
    with spec.open("a") as handle:
        handle.write('\nRUN_SPEC["dataset"]["count"] = 5\n')
    factory = runpy.run_path(str(spec))["_mock"]
    newcomers = {n: factory(n) for n in ("delta", "epsilon")}
    extend_run(str(spec), new_models=newcomers, additional_scenarios=2)
    single = [json.loads(line) for line in target.read_text().splitlines()]
    target.write_bytes(baseline)
    with spec.open("a") as handle:
        handle.write('\nRUN_SPEC["collection"]["inspect"]["phased"] = True\n')

    # Instrument the actual resolver/close lifecycle: a different model must
    # never resolve while another model's phase is still open.
    active = set()
    closed = []
    contexts = phased._isolated_context
    close = phased._close_loaded

    def isolated(prepared, nick):
        ctx, loaded = contexts(prepared, nick)
        resolve = ctx["resolve_model"]
        def tracked(name):
            assert not active or active == {name}
            active.add(name)
            return resolve(name)
        ctx["resolve_model"] = tracked
        return ctx, loaded

    async def tracked_close(loaded):
        assert len(loaded) == 1
        await close(loaded)
        closed.extend(active)
        active.clear()

    monkeypatch.setattr(phased, "_isolated_context", isolated)
    monkeypatch.setattr(phased, "_close_loaded", tracked_close)
    extend_run(str(spec), new_models=newcomers, additional_scenarios=2)
    actual = [json.loads(line) for line in target.read_text().splitlines()]
    def normalized(rows):
        return [re.sub(r"#\d+", "#counter", json.dumps(r, sort_keys=True)) for r in rows]
    assert normalized(actual) == normalized(single)
    assert not active and len(closed) == 10  # Five response phases, five judges.
    info = json.loads((tmp_path / "inspect_run.json").read_text())
    assert len(info["log_files"]) == 5
    judge_log = tmp_path / "inspect_logs" / info["log_files"][0]
    with pytest.raises(ValueError, match="only one judge"):
        export_log(judge_log, evaluations_path=tmp_path / "partial.jsonl", append=True)
    target.write_bytes(baseline)
    with pytest.raises(ValueError, match="requires phased execution"):
        extend_model(str(spec), new_model="delta", model_id="inspect:mockllm/model")


def test_phased_extension_closes_after_failed_eval(monkeypatch):
    from types import SimpleNamespace
    import inspect_pipeline.extension_phased as phased

    closed = []
    async def close():
        closed.append(True)
    model = SimpleNamespace(api=SimpleNamespace(aclose=close))
    def fail(**kwargs):
        raise RuntimeError("provider failed")
    monkeypatch.setattr(phased, "inspect_eval", fail)
    with pytest.raises(RuntimeError, match="provider failed"):
        phased._run_one(None, [model], {})
    assert closed == [True]


def test_extension_phasing_defaults_to_local_models():
    from inspect_pipeline.extend import _use_phased_extension
    prepared = {"spec": {"collection": {}}, "models": {"local": "hf_local:org/model"}}
    assert _use_phased_extension(prepared)
    prepared["spec"]["collection"]["inspect"] = {"phased": False}
    assert not _use_phased_extension(prepared)
    prepared["models"] = {"api": "inspect:mockllm/model"}
    prepared["spec"]["collection"]["inspect"] = {"phased": True}
    assert _use_phased_extension(prepared)
