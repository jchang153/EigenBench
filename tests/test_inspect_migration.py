"""Offline command migration and frozen protocol/analysis compatibility."""

import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "inspect_migration"


def test_hosted_requirements_remain_independent_of_working_directory():
    """ValueArena installs an absolute requirements path before changing cwd."""
    from packaging.requirements import Requirement
    from scripts.generate_requirements import render_requirements

    rendered = render_requirements()
    assert (ROOT / "requirements.txt").read_bytes() == rendered.encode("utf-8")
    entries = [line for line in rendered.splitlines() if line.strip() and not line.startswith("#")]
    assert all(not line.startswith(("-", ".", "/")) for line in entries)
    requirements = {requirement.name: requirement for requirement in map(Requirement, entries)}
    assert {"openai", "anthropic", "google-genai", "torch", "transformers"} <= requirements.keys()
    vllm_marker = requirements["vllm"].marker
    assert vllm_marker is not None
    assert vllm_marker.evaluate({"sys_platform": "linux"})
    assert not vllm_marker.evaluate({"sys_platform": "darwin"})


def _cli(script, spec, *args):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), str(spec), *args],
        cwd=ROOT, env={**os.environ, "MPLBACKEND": "Agg"},
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _mock_spec(folder, mode, phased):
    (folder / "scenarios.json").write_text(json.dumps(["scenario"]))
    (folder / "constitution.json").write_text(json.dumps(["criterion"]))
    spec = {
        "models": dict.fromkeys(["a", "b"], "inspect:mockllm/model"),
        "evaluation": {"mode": mode},
        "dataset": {"path": "scenarios.json"},
        "constitution": {"path": "constitution.json", "num_criteria": 1},
        "collection": {"sampler_mode": "all_to_all" if mode == "direct_rating" else "random_judge_group",
                       "group_size": 2, "sampler_seed": 7,
                       "cached_responses_path": str(folder / "responses.jsonl"),
                       "inspect": {"phased": phased, "cache": True, "display": "none"}},
        "training": {"enabled": False},
    }
    source = '''
from inspect_ai.model import ModelOutput, get_model
from pathlib import Path
CALLS = Path(__file__).with_name("calls.txt")
def mock(nick):
    def output(input, tools, tool_choice, config):
        with CALLS.open("a") as handle:
            handle.write(nick + "\\n")
        system = input[0].text
        if system.startswith("Without making"):
            text = "answer " + nick
        elif system.startswith("You are acting as a judge. Carefully consider the following response"):
            text = "reflection"
        elif "assign one integer rating" in system:
            text = "<criterion_1_rating>7</criterion_1_rating>"
        else:
            text = "<criterion_1_choice>0</criterion_1_choice>"
        return ModelOutput.from_content(model="mockllm/model", content=text)
    return get_model("mockllm/model", custom_outputs=output, memoize=False)
'''
    path = folder / "spec.py"
    path.write_text(source + f"\nRUN_SPEC = {spec!r}\n" +
                    'RUN_SPEC["models"] = {nick: mock(nick) for nick in RUN_SPEC["models"]}\n')
    return path


@pytest.mark.parametrize("script", ["run.py", "run_inspect.py"])
@pytest.mark.parametrize("mode", ["direct_rating", "pairwise_btd"])
@pytest.mark.parametrize("phased", [False, True])
def test_public_commands_use_inspect_and_never_precollect(tmp_path, monkeypatch, script, mode, phased):
    monkeypatch.setenv("INSPECT_CACHE_DIR", str(tmp_path / "inspect_cache"))
    path = _mock_spec(tmp_path, mode, phased)
    estimate = _cli(script, path, "--estimate-calls")
    assert "total_logical_generations" in estimate.stdout
    assert not (tmp_path / "calls.txt").exists()
    assert not (tmp_path / "direct_call_estimate.json").exists()
    _cli(script, path)
    rows = [json.loads(line) for line in (tmp_path / "evaluations.jsonl").read_text().splitlines()]
    assert len(rows) == (4 if mode == "direct_rating" else 2)
    assert list((tmp_path / "inspect_logs").glob("*.eval"))
    assert (tmp_path / "inspect_run.json").exists()
    calls = (tmp_path / "calls.txt").read_bytes()
    # Analysis-only runs must not touch an absent response cache, including pairwise.
    cache = tmp_path / "responses.jsonl"
    cache.unlink(missing_ok=True)
    _cli(script, path, "--collection-enabled", "false")
    assert (tmp_path / "calls.txt").read_bytes() == calls
    assert not cache.exists()
    if mode == "direct_rating":
        _cli(script, path)  # completed direct export resumes without generating
        assert (tmp_path / "calls.txt").read_bytes() == calls


@pytest.mark.parametrize("script", ["run_collect.py", "run_collect_responses.py"])
def test_retired_commands_fail_before_loading_a_spec(tmp_path, script):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "missing.py"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0
    assert "retired" in result.stderr and "scripts/run.py" in result.stderr
    assert "Traceback" not in result.stderr


def test_retired_commands_share_the_migration_message():
    from scripts import run_collect, run_collect_responses

    assert run_collect_responses.main is run_collect.main


def _assert_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_equal(a, b)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
    else:
        assert actual == expected


@pytest.mark.parametrize("mode,name", [("direct_rating", "direct"), ("pairwise_btd", "pairwise")])
@pytest.mark.parametrize("entrypoint", ["hosted", "runner"])
def test_historical_records_keep_frozen_analysis_and_bootstrap(tmp_path, mode, name, entrypoint):
    """The hosted import remains executable, and analysis-only collection stays disabled."""
    spec = {
        "name": "migration_fixture", "models": dict.fromkeys(["a", "b", "c"], "mock"),
        "evaluation": {"mode": mode}, "constitution": {"num_criteria": 2},
        "collection": {"enabled": False, "evaluations_path": str(FIXTURES / f"{name}.jsonl"),
                       "cached_responses_path": str(tmp_path / "never_generated.jsonl")},
        "training": {"enabled": True, "output_dir": str(tmp_path / "train"),
                     "separate_criteria": False, "max_epochs": 3, "dims": [2], "batch_size": 32,
                     "bootstrap": {"enabled": True, "n_bootstraps": 3, "random_seed": 23}},
    }
    path = tmp_path / "spec.py"
    path.write_text(f"RUN_SPEC = {spec!r}\n")
    if entrypoint == "hosted":
        from scripts.run_train import main
    else:
        from scripts.run import main
    with torch.random.fork_rng():
        torch.manual_seed(73)
        np.random.seed(73)
        random.seed(73)
        main(str(path))
    assert not (tmp_path / "never_generated.jsonl").exists()
    expected = json.loads((FIXTURES / "analysis.json").read_text())[name]
    for relative, frozen in expected.items():
        output = tmp_path / "train" / relative
        if output.suffix == ".json":
            actual = json.loads(output.read_text())
        elif output.suffix == ".csv":
            actual = np.loadtxt(output, delimiter=",").tolist()
        else:
            actual = output.read_text()
        _assert_equal(actual, frozen)


def test_extension_dispatch_uses_the_inspect_extension(tmp_path, monkeypatch):
    from inspect_pipeline import extend
    from scripts.run import main
    path = tmp_path / "spec.py"
    path.write_text("RUN_SPEC = {'extension': {'from_evaluations': 'old.jsonl'}, "
                    "'training': {'enabled': False}}\n")
    calls = []
    monkeypatch.setattr(extend, "extend_run", lambda spec: calls.append(spec))
    main(str(path))
    assert calls == [str(path)]


@pytest.mark.parametrize("mode", ["direct_rating", "pairwise_btd"])
@pytest.mark.parametrize("training_enabled,space_upload,artifacts_written,expect_scoring,expect_training", [
    (True, False, True, True, False),
    (True, False, False, True, True),
    (False, False, False, False, False),
    (True, True, False, False, False),
])
def test_runner_preserves_analysis_ownership(tmp_path, monkeypatch, mode, training_enabled,
                                            space_upload, artifacts_written, expect_scoring,
                                            expect_training):
    from inspect_pipeline import collect, pairwise_collect
    from scripts import run as runner, run_train
    import tempfile

    records = tmp_path / "evaluations.jsonl"
    records.write_text("")
    output = tmp_path / "train"
    output.mkdir()
    # Existing artifacts do not establish that this invocation completed analysis.
    (output / "eigentrust.json").write_text('{"stale": true}')
    spec = {
        "models": {"a": "mockllm/model"}, "evaluation": {"mode": mode},
        "collection": {"enabled": True, "evaluations_path": str(records)},
        "training": {"enabled": training_enabled, "output_dir": str(output)},
        "upload": {"enabled": space_upload, "backend": "valuearena_space",
                   "secret": "offline-test-only", "git_commit": "test"},
    }
    monkeypatch.setattr(runner, "load_run_spec", lambda _: (spec, tmp_path))
    monkeypatch.setattr(runner, "_write_space_safe_spec", lambda _: str(tmp_path / "space_spec.py"))
    named_temporary_file = tempfile.NamedTemporaryFile
    monkeypatch.setattr(tempfile, "NamedTemporaryFile",
                        lambda *args, **kwargs: named_temporary_file(*args, dir=tmp_path, **kwargs))
    uploads = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: uploads.append((args, kwargs)))
    collection_calls, training_calls = [], []

    def collect_panel(spec_ref, *, scoring, analysis_status, resume_log=None):
        assert resume_log is None
        collection_calls.append((spec_ref, scoring))
        if artifacts_written:
            analysis_status["artifacts_written"] = True
        return []

    monkeypatch.setattr(collect, "collect_direct_ratings_inspect", collect_panel)
    monkeypatch.setattr(pairwise_collect, "collect_pairwise_inspect", collect_panel)
    monkeypatch.setattr(run_train, "main", lambda spec_ref: training_calls.append(spec_ref))
    runner.main("offline-spec")
    assert collection_calls == [("offline-spec", expect_scoring)]
    assert training_calls == (["offline-spec"] if expect_training else [])
    assert len(uploads) == int(space_upload)
