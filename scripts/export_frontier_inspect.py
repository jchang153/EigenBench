"""Export the completed frontier direct-rating run as native Inspect logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalResults,
    EvalSample,
    EvalSpec,
    EvalStats,
    read_eval_log,
    write_eval_log,
)
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.scorer import Score
from inspect_ai.viewer import (
    SampleScoreView,
    SampleScoreViewSort,
    TaskSamplesColumn,
    TaskSamplesSort,
    TaskSamplesView,
    ViewerConfig,
)

from pipeline.eval.direct_rating import (
    build_direct_rating_prompt,
    build_direct_rating_user_prompt,
    build_direct_reflection_prompt,
    build_direct_reflection_user_prompt,
)


DEFAULT_RUN_DIR = _REPO_ROOT / "runs/direct_rating_frontier_200"
EXPECTED_RUNS = {
    "kindness": {"records": 1600, "criteria": 8, "missing": set()},
    "conservatism": {
        "records": 1599,
        "criteria": 10,
        "missing": {(4, "Gemini 3.7 Flash")},
    },
    "environmental_ethics": {
        "records": 1599,
        "criteria": 12,
        "missing": {(4, "Gemini 3.7 Flash")},
    },
}
CONSTITUTION_LABELS = {
    "kindness": "Kindness",
    "conservatism": "Conservatism",
    "environmental_ethics": "Environmental ethics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_model_map(run_dir: Path) -> dict[str, str]:
    common_path = run_dir / "common.py"
    values = runpy.run_path(str(common_path))
    models = values.get("MODELS")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"MODELS is missing from {common_path}")
    return {str(name): str(model) for name, model in models.items()}


def load_records(path: Path, expected: dict[str, Any]) -> list[dict[str, Any]]:
    finalized_path = Path(f"{path}.checkpoint") / "finalized.json"
    finalized = json.loads(finalized_path.read_text(encoding="utf-8"))
    source_hash = sha256_file(path)
    if finalized.get("sha256") != source_hash:
        raise ValueError(f"Finalized hash does not match {path}")

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != expected["records"] or finalized.get("records") != len(records):
        raise ValueError(f"Unexpected record count in {path}: {len(records)}")

    seen: set[tuple[int, str]] = set()
    for record in records:
        if record.get("schema_version") != 2 or record.get("record_type") != "direct_rating":
            raise ValueError(f"Unsupported record schema in {path}")
        key = (int(record["scenario_index"]), str(record["evaluee"]["name"]))
        if key in seen:
            raise ValueError(f"Duplicate scenario/evaluee record in {path}: {key}")
        seen.add(key)
        ratings = record.get("ratings")
        if not isinstance(ratings, list) or len(ratings) != expected["criteria"]:
            raise ValueError(f"Unexpected criterion count in {path}: {key}")

    return records


def validate_coverage(
    records: list[dict[str, Any]],
    model_names: list[str],
    expected_missing: set[tuple[int, str]],
) -> None:
    actual = {
        (int(record["scenario_index"]), str(record["evaluee"]["name"]))
        for record in records
    }
    expected = {(scenario, model) for scenario in range(200) for model in model_names}
    if expected - actual != expected_missing or actual - expected:
        raise ValueError("Scenario/evaluee coverage does not match the finalized run")


def criterion_label(record: dict[str, Any]) -> str:
    first = str(record["ratings"][0]["criterion"])
    match = re.match(r"Criterion 1 for (.*?):", first)
    if not match:
        raise ValueError(f"Cannot read constitution name from: {first}")
    return match.group(1)


def display_call(stage: str, system_prompt: str, user_prompt: str) -> str:
    """Lay out one original API call without merging its two prompt roles."""
    return (
        f"## {stage} call (independent API call)\n\n"
        f"### System prompt\n\n{system_prompt}\n\n"
        f"### User prompt\n\n{user_prompt}"
    )


def make_sample(
    record: dict[str, Any],
    *,
    constitution_slug: str,
    judge_model: str,
    model_map: dict[str, str],
) -> EvalSample:
    criteria = str(record["constitution"])
    scenario = str(record["scenario"])
    response = str(record["response"])
    reflection = str(record["reflection"])
    judgment = str(record["judgment_raw"])

    reflection_input = ChatMessageUser(
        content=display_call(
            "Reflection",
            build_direct_reflection_prompt(),
            build_direct_reflection_user_prompt(criteria, scenario, response),
        ),
        source="input",
        metadata={"stage": "reflection", "call": 1},
    )
    reflection_output = ChatMessageAssistant(
        content=reflection,
        source="generate",
        model=judge_model,
        metadata={"stage": "reflection", "call": 1},
    )
    rating_input = ChatMessageUser(
        content=display_call(
            "Rating",
            build_direct_rating_prompt(),
            build_direct_rating_user_prompt(
                criteria,
                scenario,
                response,
                reflection,
            ),
        ),
        source="input",
        metadata={"stage": "rating", "call": 2, "new_api_call": True},
    )
    rating_output = ChatMessageAssistant(
        content=judgment,
        source="generate",
        model=judge_model,
        metadata={"stage": "rating", "call": 2},
    )

    ratings = {
        f"criterion_{int(item['criterion_index']) + 1}": int(item["rating"])
        for item in record["ratings"]
    }
    mean_rating = sum(ratings.values()) / len(ratings)
    criterion_text = {
        f"criterion_{int(item['criterion_index']) + 1}": str(item["criterion"])
        for item in record["ratings"]
    }
    record_hash = hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    evaluee_name = str(record["evaluee"]["name"])
    scenario_index = int(record["scenario_index"])

    return EvalSample(
        id=f"Scenario {scenario_index + 1:03d} | {evaluee_name}",
        epoch=1,
        input=[reflection_input],
        target=evaluee_name,
        messages=[
            reflection_input,
            reflection_output,
            rating_input,
            rating_output,
        ],
        output=ModelOutput.from_content(judge_model, judgment),
        scores={
            "mean_rating": Score(
                value=mean_rating,
                metadata={
                    "description": "Mean of this judgment's criterion ratings",
                    "scale": {"min": 1, "max": 10},
                },
            ),
            "criterion_ratings": Score(
                value=ratings,
                answer=judgment,
                explanation=reflection,
                metadata={
                    "criterion_text": criterion_text,
                    "scale": {"min": 1, "max": 10},
                },
            )
        },
        metadata={
            "dataset": "AIRiskDilemmas",
            "scenario_index": scenario_index,
            "scenario": scenario,
            "constitution": constitution_slug,
            "judge_index": int(record["judge"]["index"]),
            "judge_name": str(record["judge"]["name"]),
            "judge_model": judge_model,
            "evaluee_index": int(record["evaluee"]["index"]),
            "evaluee_name": evaluee_name,
            "evaluee_model": model_map[evaluee_name],
            "sampling": record["sampling"],
            "imported_from_jsonl": True,
            "call_structure": "reflection and rating were separate API calls",
            "record_sha256": record_hash,
        },
    )


def build_log(
    records: list[dict[str, Any]],
    *,
    source_path: Path,
    source_hash: str,
    source_git_commit: str,
    constitution_slug: str,
    judge_name: str,
    judge_model: str,
    model_map: dict[str, str],
    created: str,
) -> EvalLog:
    samples = [
        make_sample(
            record,
            constitution_slug=constitution_slug,
            judge_model=judge_model,
            model_map=model_map,
        )
        for record in sorted(
            records,
            key=lambda item: (int(item["scenario_index"]), int(item["evaluee"]["index"])),
        )
    ]
    sample_ids = [str(sample.id) for sample in samples]
    constitution_name = criterion_label(records[0])
    task_name = f"{CONSTITUTION_LABELS[constitution_slug]} direct ratings"
    criterion_text = [str(item["criterion"]) for item in records[0]["ratings"]]
    relative_source = str(source_path.relative_to(_REPO_ROOT))
    task_slug = f"{constitution_slug}-{slug(judge_name)}"

    return EvalLog(
        status="success",
        eval=EvalSpec(
            eval_set_id="frontier-direct-ratings-200",
            eval_id=f"frontier-direct-ratings-200-{task_slug}",
            run_id="frontier-direct-ratings-200",
            created=created,
            task=task_name,
            task_id=task_slug,
            task_version=1,
            task_file="pipeline/eval/direct_rating.py",
            task_display_name=task_name,
            task_registry_name="direct_criterion_rating_import",
            dataset=EvalDataset(
                name="AIRiskDilemmas 0-199",
                location=relative_source,
                samples=len(samples),
                sample_ids=sample_ids,
                shuffled=False,
            ),
            model=judge_name,
            config=EvalConfig(epochs=1, log_samples=True),
            packages={"inspect_ai": "0.3.240"},
            viewer=ViewerConfig(
                task_samples_view=TaskSamplesView(
                    name="Direct ratings",
                    columns=[
                        TaskSamplesColumn(id="sampleId"),
                        TaskSamplesColumn.score("mean_rating"),
                        TaskSamplesColumn(id="sampleStatus", visible=False),
                        TaskSamplesColumn(id="input", visible=False),
                        TaskSamplesColumn(id="target", visible=False),
                        TaskSamplesColumn(id="answer", visible=False),
                        TaskSamplesColumn(id="tokens", visible=False),
                        TaskSamplesColumn(id="duration", visible=False),
                    ],
                    sort=[TaskSamplesSort(column="sampleId", dir="asc")],
                    multiline=False,
                    score_labels={"mean_rating": "Mean rating"},
                ),
                sample_score_view=SampleScoreView(
                    default="grid",
                    sort=SampleScoreViewSort(column="name", dir="asc"),
                ),
            ),
            metadata={
                "evaluation_mode": "direct_rating",
                "imported_from_jsonl": True,
                "import_schema_version": 1,
                "source_path": relative_source,
                "source_sha256": source_hash,
                "source_git_commit": source_git_commit,
                "constitution_name": constitution_name,
                "criteria": criterion_text,
                "sampling_mode": "balanced_unique_judge",
                "response_redundancy": 1,
                "include_self": False,
                "timing_and_token_usage_recorded": False,
                "call_structure": "reflection and rating were separate API calls",
            },
        ),
        results=EvalResults(
            total_samples=len(samples),
            completed_samples=len(samples),
            scores=[],
            metadata={"criterion_score_count": sum(len(record["ratings"]) for record in records)},
        ),
        stats=EvalStats(started_at=created, completed_at=created),
        samples=samples,
    )


def validate_written_log(path: Path, *, expected_samples: int) -> None:
    log = read_eval_log(path)
    if log.status != "success" or log.samples is None:
        raise ValueError(f"Inspect could not read a successful log from {path}")
    if len(log.samples) != expected_samples:
        raise ValueError(f"Inspect log sample count mismatch in {path}")
    if log.results is None or log.results.completed_samples != expected_samples:
        raise ValueError(f"Inspect log results mismatch in {path}")
    for sample in log.samples:
        if not sample.scores or {
            "criterion_ratings",
            "mean_rating",
        } - sample.scores.keys():
            raise ValueError(f"Missing criterion score in {path}: {sample.id}")
        if len(sample.messages) != 4:
            raise ValueError(f"Incomplete two-call transcript in {path}: {sample.id}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")

    model_map = load_model_map(run_dir)
    model_names = list(model_map)
    commit = source_commit()
    created = datetime.now(UTC).isoformat()
    written: list[dict[str, Any]] = []

    for constitution_slug, expected in EXPECTED_RUNS.items():
        source_path = run_dir / constitution_slug / "evaluations.jsonl"
        records = load_records(source_path, expected)
        validate_coverage(records, model_names, expected["missing"])
        source_hash = sha256_file(source_path)
        by_judge: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            judge_name = str(record["judge"]["name"])
            if judge_name not in model_map:
                raise ValueError(f"Unknown judge in {source_path}: {judge_name}")
            by_judge[judge_name].append(record)

        if set(by_judge) != set(model_names):
            raise ValueError(f"Judge roster mismatch in {source_path}")
        for judge_name in model_names:
            log = build_log(
                by_judge[judge_name],
                source_path=source_path,
                source_hash=source_hash,
                source_git_commit=commit,
                constitution_slug=constitution_slug,
                judge_name=judge_name,
                judge_model=f"openrouter/{model_map[judge_name]}",
                model_map=model_map,
                created=created,
            )
            path = output_dir / f"{constitution_slug}__{slug(judge_name)}.eval"
            write_eval_log(log, location=path, format="eval")
            validate_written_log(path, expected_samples=len(by_judge[judge_name]))
            written.append(
                {
                    "path": path.name,
                    "constitution": constitution_slug,
                    "judge": judge_name,
                    "samples": len(by_judge[judge_name]),
                    "sha256": sha256_file(path),
                }
            )

    manifest = {
        "schema_version": 1,
        "source_git_commit": commit,
        "logs": written,
        "total_logs": len(written),
        "total_samples": sum(item["samples"] for item in written),
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
