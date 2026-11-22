"""One offline draw, then replay the reviewed judgments through both public entry points."""

import json
import re
import sys
from collections import Counter

import pytest
from inspect_ai import eval as inspect_eval, eval_retry
from inspect_ai.model import Model

from eigenbench.config import load_run_spec
from eigenbench.protocol import direct_rating
from inspect_pipeline import context
from inspect_pipeline.context import load_selection
from inspect_pipeline.eigenbench import eigenbench
from inspect_pipeline.export import records_from_log
from inspect_pipeline.provenance import read_resolved_sampler_seed
from scripts import run
from tests.inspect_integration.test_inspect_collect import NICKS, run_dir  # noqa: F401


MODE = "pair_balanced_random_matching"
PREVIEW = "--preview-assignments"


def _configure(path, statement):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{statement}\n")


@pytest.fixture
def spec(run_dir, monkeypatch):
    folder, write_spec = run_dir
    monkeypatch.chdir(folder)
    path = write_spec(sampler_mode=MODE)
    # Distinct, stable provider names across loads; keep the fixture's scripted outputs.
    path.write_text(path.read_text().replace(
        'get_model("mockllm/model",', 'get_model(f"mockllm/{nick}",',
    ))
    _configure(path, 'RUN_SPEC["dataset"]["start"] = 1')
    _configure(path, 'RUN_SPEC["training"]["enabled"] = False')
    return path


@pytest.fixture
def cli(monkeypatch):
    def invoke(spec, *args):
        monkeypatch.setattr(sys, "argv", ["scripts/run.py", str(spec), *args])
        run.cli()

    return invoke


def _no_generation(*args, **kwargs):
    pytest.fail("preview or stale-plan preflight attempted model generation")


def _no_draw(*args, **kwargs):
    pytest.fail("redrew the pair-balanced assignments after preview")


def _edges(rows):
    return Counter((r["scenario_index"], r["judge"], r["evaluee"], r["sampling_round"])
                   for r in rows)


def _record_edges(records):
    return Counter((r["scenario_index"], r["judge"]["name"], r["evaluee"]["name"],
                    r["sampling"]["round"]) for r in records)


@pytest.mark.parametrize("seed", [73, None], ids=["fixed-seed", "unseeded"])
def test_preview_replayed_by_cli_and_native_task(spec, cli, monkeypatch, capsys, seed):
    _configure(spec, f'RUN_SPEC["collection"]["sampler_seed"] = {seed!r}')
    _configure(spec, 'RUN_SPEC["collection"]["evaluations_path"] = "reviewed/ratings.jsonl"')
    evaluations = spec.parent / "reviewed" / "ratings.jsonl"
    manifest = evaluations.with_name("judge_assignments.json")
    expected = Counter()
    draw = direct_rating._pair_balanced_assignments

    def draw_once(*args, **kwargs):
        assert not expected, "first preview drew more than once"
        assignments = draw(*args, **kwargs)
        expected.update(
            (a["scenario_index"], a["judge_nick"], nick, a["sampling_round"])
            for a in assignments for nick in a["eval_nicks"]
        )
        return assignments

    with monkeypatch.context() as guard:
        guard.setattr(Model, "generate", _no_generation)
        guard.setattr(direct_rating, "_pair_balanced_assignments", draw_once)
        cli(spec, PREVIEW)
        stdout = capsys.readouterr().out
        assert str(manifest) in stdout
        assert re.search(r"\b9\b", stdout.replace(str(manifest), ""))

    frozen = manifest.read_bytes()
    plan = json.loads(frozen)
    selected, criteria = load_selection(*load_run_spec(str(spec)))
    assert plan["dataset_sha256"] == direct_rating.direct_input_sha256(selected)
    assert plan["criteria_sha256"] == direct_rating.direct_input_sha256(criteria)
    assert _edges(plan["judgments"]) == expected
    assert Counter((s, e) for s, _, e, _ in expected.elements()) == {
        (s, e): 1 for s in (1, 2, 3) for e in NICKS
    }

    monkeypatch.setattr(direct_rating, "_pair_balanced_assignments", _no_draw)
    with monkeypatch.context() as guard:
        guard.setattr(Model, "generate", _no_generation)
        cli(spec, PREVIEW)
        assert manifest.read_bytes() == frozen
        cli(spec, "--estimate-calls")
        assert not evaluations.exists()
        assert not evaluations.with_name("inspect_logs").exists()
        # Construct the registered task directly, outside the runner's prepared context.
        task = eigenbench(str(spec))
        assert task.metadata["eigenbench"]["sampler_mode"] == MODE
        metadata = [sample.metadata for sample in task.dataset]
        assert {md["sampler_mode"] for md in metadata} == {MODE}
        assert Counter((md["scenario_index"], md["judge_nick"], md["eval_nick"], md["sampling_round"])
                       for md in metadata) == expected

    cli(spec)
    records = [json.loads(line) for line in evaluations.read_text().splitlines()]
    assert {r["sampling"]["mode"] for r in records} == {MODE}
    assert _record_edges(records) == expected
    with monkeypatch.context() as guard:
        guard.setattr(Model, "generate", _no_generation)
        cli(spec)
    assert manifest.read_bytes() == frozen


def test_unseeded_native_log_retries_without_redrawing(spec, cli, monkeypatch):
    _configure(spec, 'RUN_SPEC["collection"]["sampler_seed"] = None')
    # Runner retries require the original answers for any unfinished judgments.
    cache = spec.with_name("responses.jsonl")
    cache.write_text("".join(json.dumps({
        "scenario_index": s, "responses": {nick: f"[{nick}#{s}] answers" for nick in NICKS}
    }) + "\n" for s in (1, 2, 3)))
    _configure(spec, f'RUN_SPEC["collection"]["cached_responses_path"] = {str(cache)!r}')
    cli(spec, PREVIEW)
    plan = json.loads(spec.with_name("judge_assignments.json").read_text())
    seed = plan["resolved_sampler_seed"]
    assert type(seed) is int
    expected = _edges(plan["judgments"])
    monkeypatch.setattr(direct_rating, "_pair_balanced_assignments", _no_draw)

    generate = Model.generate
    failed = False

    async def fail_once(model, *args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("interrupted mock judgment")
        return await generate(model, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Model, "generate", fail_once)
        [initial] = inspect_eval(eigenbench, task_args={"spec": str(spec), "panel_metrics": False},
                                 model=None, fail_on_error=False, display="none",
                                 log_dir=str(spec.with_name("native-logs")))
    assert sum(sample.error is not None for sample in initial.samples) == 1
    assert initial.eval.metadata["eigenbench"]["resolved_sampler_seed"] == seed
    resolved = []
    build_context = context._run_context_from_spec

    def capture_retry_seed(*args, **kwargs):
        ctx = build_context(*args, **kwargs)
        resolved.append(ctx["resolved_sampler_seed"])
        return ctx

    with monkeypatch.context() as guard:
        guard.setattr(context, "_run_context_from_spec", capture_retry_seed)
        [retried] = eval_retry(initial.location, display="none", log_dir=str(spec.with_name("native-retry")))
    assert retried.status == "success"
    assert set(resolved) == {seed}
    assert _record_edges(records_from_log(retried)) == expected

    # Consume the original native log, not a runner-generated log with explicit task args.
    cli(spec, "--resume-log", initial.location)
    evaluations = spec.with_name("evaluations.jsonl")
    assert read_resolved_sampler_seed(evaluations) == seed
    assert _record_edges(map(json.loads, evaluations.read_text().splitlines())) == expected


@pytest.mark.parametrize("drift", [
    "panel-order", "panel-model", "seed", "sampler", "include-self", "rubric", "dataset", "valid-edges",
])
def test_stale_preview_fails_closed(spec, cli, monkeypatch, drift):
    manifest = spec.with_name("judge_assignments.json")
    monkeypatch.setattr(Model, "generate", _no_generation)
    cli(spec, PREVIEW)
    monkeypatch.setattr(direct_rating, "_pair_balanced_assignments", _no_draw)

    changes = {
        "panel-order": 'RUN_SPEC["models"] = dict(reversed(RUN_SPEC["models"].items()))',
        "panel-model": 'RUN_SPEC["models"]["alpha"] = _mock("replacement")',
        "seed": 'RUN_SPEC["collection"]["sampler_seed"] = None',
        "sampler": 'RUN_SPEC["collection"]["sampler_mode"] = "balanced_unique_judge"',
        "include-self": 'RUN_SPEC["evaluation"]["direct_rating"] = {"include_self": False}',
    }
    if drift in changes:
        _configure(spec, changes[drift])
    elif drift in ("dataset", "rubric"):
        source = spec.with_name("scenarios.json" if drift == "dataset" else "constitution.json")
        data = json.loads(source.read_text())
        data[1] += " (changed)"
        source.write_text(json.dumps(data))
    else:
        # Swap two whole matchings: every prompt and every directed pair still has
        # the right counts. Structural validation alone cannot detect this edit.
        plan = json.loads(manifest.read_text())
        rows = plan["judgments"]
        before = _edges(rows)
        for row in rows:
            row["scenario_index"] = {1: 2, 2: 1, 3: 3}[row["scenario_index"]]
        assert _edges(rows) != before
        manifest.write_text(json.dumps(plan))

    reviewed = manifest.read_bytes()
    for attempt in (lambda: cli(spec, PREVIEW), lambda: cli(spec, "--estimate-calls"),
                    lambda: eigenbench(str(spec)), lambda: cli(spec)):
        with pytest.raises((ValueError, RuntimeError), match=r"(?i)assign|manifest|preview|frozen|sha256"):
            attempt()
        assert manifest.read_bytes() == reviewed
        assert not spec.with_name("evaluations.jsonl").exists()


def test_existing_cli_flags_still_dispatch(cli, spec):
    with pytest.raises(ValueError, match="--resume-log requires an enabled direct-rating collection"):
        cli(spec, "--collection-enabled", "false", "--resume-log", "partial.eval")
