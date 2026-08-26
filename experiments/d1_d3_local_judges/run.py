"""Run the balanced D1-versus-D3 matrix with local vLLM judges."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pipeline.eval.checkpoint import CollectionCheckpoint
from pipeline.utils import load_records, save_records

from .analysis import generate_reports

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"
DEFAULT_OUTPUT_DIR = ROOT / "runs/d1_d3_local_judges"
PROMPT_VERSION = "d1-d3-local-v1"
OLMO_2_REPO_ID = "allenai/OLMo-2-1124-7B-Instruct"


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    user: str

    @classmethod
    def load(cls, path: Path) -> PromptTemplate:
        text = path.read_text(encoding="utf-8")
        system_marker = "## System message"
        user_marker = "## User message"
        if system_marker not in text or user_marker not in text:
            raise ValueError(f"prompt must contain system and user sections: {path}")
        _, remainder = text.split(system_marker, 1)
        system, user = remainder.split(user_marker, 1)
        return cls(system=system.strip(), user=user.strip())

    def messages(self, **values: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _render(self.system, values)},
            {"role": "user", "content": _render(self.user, values)},
        ]


@dataclass(frozen=True)
class ResponseCell:
    scenario_index: int
    scenario: str
    scenario_sha256: str
    evaluee_name: str
    evaluee_model_id: str
    response: str
    response_sha256: str


@dataclass(frozen=True)
class GenerationTask:
    identity: dict
    messages: list[dict[str, str]]
    parser: Callable[[str], dict]


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(encoded)


def _render(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    unresolved = [
        token
        for token in (
            "{criteria}",
            "{criterion}",
            "{scenario}",
            "{response}",
            "{reflection}",
            "{criterion_reflection}",
        )
        if token in rendered
    ]
    if unresolved:
        raise ValueError(f"unresolved prompt fields: {unresolved}")
    return rendered


def _load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("experiment config must contain a JSON object")
    judges = payload.get("judges")
    if not isinstance(judges, list) or len(judges) != 4:
        raise ValueError("experiment config must define exactly four judges")
    names = [item.get("name") for item in judges]
    if len(set(names)) != len(names):
        raise ValueError("judge names must be unique")
    for judge in judges:
        if judge.get("prompt_format") not in {"chat_template", "plain"}:
            raise ValueError(
                f"unsupported prompt_format for {judge.get('name')}: "
                f"{judge.get('prompt_format')}"
            )
        model = judge.get("model")
        if not isinstance(model, dict) or model.get("provider") != "hf_local":
            raise ValueError(f"judge must use a structured hf_local model: {judge}")
    return payload


def _load_criteria(path: Path) -> tuple[list[str], list[str]]:
    criteria = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(criteria, list) or not all(
        isinstance(item, str) and item.strip() for item in criteria
    ):
        raise ValueError(f"constitution must be a non-empty JSON string list: {path}")
    criterion_ids = [f"criterion_{index:02d}" for index in range(1, len(criteria) + 1)]
    return criteria, criterion_ids


def _load_response_cells(
    responses_path: Path,
    reference: dict,
) -> list[ResponseCell]:
    records = load_records(responses_path)
    by_index = {}
    for record in records:
        if not isinstance(record, dict) or "scenario_index" not in record:
            continue
        index = int(record["scenario_index"])
        if index in by_index:
            raise ValueError(f"duplicate scenario_index in response source: {index}")
        by_index[index] = record

    evaluee_ids = {
        str(item["name"]): str(item["source_model_id"])
        for item in reference["evaluees"]
    }
    cells = []
    for expected in reference["inputs"]:
        scenario_index = int(expected["scenario_index"])
        evaluee_name = str(expected["evaluee"])
        record = by_index.get(scenario_index)
        if record is None:
            raise ValueError(
                f"response source is missing reference scenario {scenario_index}"
            )
        scenario = record.get("scenario")
        responses = record.get("responses")
        if not isinstance(scenario, str) or not isinstance(responses, dict):
            raise TypeError(
                f"scenario {scenario_index} must contain scenario text and responses"
            )
        response = responses.get(evaluee_name)
        if not isinstance(response, str) or not response.strip():
            raise ValueError(
                f"scenario {scenario_index} has no response for {evaluee_name}"
            )
        scenario_hash = _sha256_text(scenario)
        response_hash = _sha256_text(response)
        if scenario_hash != expected["scenario_sha256"]:
            raise ValueError(
                f"scenario hash mismatch for scenario {scenario_index}: "
                f"expected {expected['scenario_sha256']}, got {scenario_hash}"
            )
        if response_hash != expected["response_sha256"]:
            raise ValueError(
                f"response hash mismatch for scenario {scenario_index}, "
                f"evaluee {evaluee_name}: expected {expected['response_sha256']}, "
                f"got {response_hash}"
            )
        cells.append(
            ResponseCell(
                scenario_index=scenario_index,
                scenario=scenario,
                scenario_sha256=scenario_hash,
                evaluee_name=evaluee_name,
                evaluee_model_id=evaluee_ids[evaluee_name],
                response=response,
                response_sha256=response_hash,
            )
        )

    expected_count = len(reference["scenario_indices"]) * len(reference["evaluees"])
    if len(cells) != expected_count:
        raise ValueError(
            f"reference manifest should define {expected_count} cells, found {len(cells)}"
        )
    return cells


def _criteria_payload(criteria: list[str], criterion_ids: list[str]) -> str:
    return "\n".join(
        f"{criterion_id}: {criterion}"
        for criterion_id, criterion in zip(criterion_ids, criteria)
    )


def _object_schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _schemas(criterion_ids: list[str]) -> dict[str, dict]:
    reflection_item = _object_schema(
        {
            "criterion_id": {"type": "string", "enum": criterion_ids},
            "reflection": {"type": "string", "minLength": 1},
        },
        ["criterion_id", "reflection"],
    )
    rating_item = _object_schema(
        {
            "criterion_id": {"type": "string", "enum": criterion_ids},
            "rating": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        ["criterion_id", "rating"],
    )
    return {
        "d1_reflection": _object_schema(
            {"reflection": {"type": "string", "minLength": 1}},
            ["reflection"],
        ),
        "d1_judgment": _object_schema(
            {
                "ratings": {
                    "type": "array",
                    "minItems": len(criterion_ids),
                    "maxItems": len(criterion_ids),
                    "items": rating_item,
                }
            },
            ["ratings"],
        ),
        "d3_reflection": _object_schema(
            {
                "reflections": {
                    "type": "array",
                    "minItems": len(criterion_ids),
                    "maxItems": len(criterion_ids),
                    "items": reflection_item,
                }
            },
            ["reflections"],
        ),
        "d3_judgment": _object_schema(
            {"rating": {"type": "integer", "minimum": 1, "maximum": 10}},
            ["rating"],
        ),
    }


def _parse_json_object(text: str) -> dict:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise TypeError("structured completion must be a JSON object")
    return payload


def _parse_d1_reflection(text: str) -> dict:
    payload = _parse_json_object(text)
    if set(payload) != {"reflection"}:
        raise ValueError("D1 reflection must contain only 'reflection'")
    reflection = payload["reflection"]
    if not isinstance(reflection, str) or not reflection.strip():
        raise ValueError("D1 reflection is empty")
    return {"reflection": reflection.strip()}


def _parse_entries(
    text: str,
    *,
    container: str,
    value_field: str,
    criterion_ids: list[str],
) -> dict:
    payload = _parse_json_object(text)
    if set(payload) != {container} or not isinstance(payload[container], list):
        raise ValueError(f"completion must contain only a '{container}' list")
    entries = payload[container]
    if len(entries) != len(criterion_ids):
        raise ValueError(
            f"{container} must contain {len(criterion_ids)} entries, got {len(entries)}"
        )
    parsed = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"criterion_id", value_field}:
            raise ValueError(
                f"each {container} entry must contain criterion_id and {value_field}"
            )
        criterion_id = entry["criterion_id"]
        if criterion_id not in criterion_ids or criterion_id in parsed:
            raise ValueError(f"invalid or duplicate criterion ID: {criterion_id}")
        value = entry[value_field]
        if value_field == "reflection":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"empty reflection for {criterion_id}")
            value = value.strip()
        else:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 10
            ):
                raise ValueError(f"invalid rating for {criterion_id}: {value!r}")
        parsed[criterion_id] = value
    missing = set(criterion_ids) - set(parsed)
    if missing:
        raise ValueError(f"missing criterion IDs: {sorted(missing)}")
    return {criterion_id: parsed[criterion_id] for criterion_id in criterion_ids}


def _parse_d3_rating(text: str) -> dict:
    payload = _parse_json_object(text)
    if set(payload) != {"rating"}:
        raise ValueError("isolated judgment must contain only 'rating'")
    value = payload["rating"]
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
        raise ValueError(f"isolated rating must be an integer from 1 to 10: {value!r}")
    return {"rating": value}


def _identity(
    *,
    judge: dict,
    cell: ResponseCell,
    design: str,
    stage: str,
    criterion_id: str | None = None,
) -> dict:
    model = judge["model"]
    identity = {
        "prompt_version": PROMPT_VERSION,
        "design": design,
        "stage": stage,
        "scenario_index": cell.scenario_index,
        "judge": judge["name"],
        "evaluee": cell.evaluee_name,
        "model": model["repo_id"],
        "revision": model.get("revision"),
    }
    if criterion_id is not None:
        identity["criterion_id"] = criterion_id
    return identity


def _plain_prompt(messages: list[dict[str, str]]) -> str:
    sections = [
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    ]
    return "\n\n".join(sections) + "\n\nASSISTANT:\n"


def _format_prompt(
    tokenizer, messages: list[dict[str, str]], prompt_format: str
) -> str:
    if prompt_format == "plain":
        return _plain_prompt(messages)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "The configured chat-template model could not render its messages. "
            "Use prompt_format='plain' only if the model is intentionally a base model."
        ) from exc


def _execute_phase(
    *,
    llm,
    tokenizer,
    tasks: list[GenerationTask],
    checkpoint: CollectionCheckpoint,
    prompt_format: str,
    schema: dict,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
    batch_size: int,
    verbose: bool,
) -> None:
    pending = [
        task
        for task in tasks
        if checkpoint.load_completed(task.identity) is None
        and not _is_terminal_failure(checkpoint, task.identity)
    ]
    if not pending:
        if verbose and tasks:
            print(
                f"  {tasks[0].identity['design']} {tasks[0].identity['stage']}: "
                f"{len(tasks)} already settled"
            )
        return
    try:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as exc:
        raise RuntimeError(
            "This experiment requires a recent vLLM with offline structured outputs. "
            "Upgrade vLLM inside the RunPod environment."
        ) from exc

    structured = StructuredOutputsParams(json=schema)
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        structured_outputs=structured,
    )
    phase = f"{tasks[0].identity['design']} {tasks[0].identity['stage']}"
    total_pending = len(pending)
    for offset in range(0, total_pending, batch_size):
        batch = pending[offset : offset + batch_size]
        retry = batch
        for attempt in range(1, max_attempts + 1):
            prompts = [
                _format_prompt(tokenizer, task.messages, prompt_format)
                for task in retry
            ]
            if verbose:
                print(
                    f"  {phase}: batch {offset // batch_size + 1}, "
                    f"attempt {attempt}, n={len(retry)}"
                )
            started = time.perf_counter()
            try:
                outputs = llm.generate(prompts, params)
            except Exception as exc:
                error = {
                    "error_type": "local_generation_error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "retryable": True,
                    "exhausted": False,
                    "attempt": attempt,
                }
                for task in retry:
                    checkpoint.save_failed(task.identity, error)
                raise RuntimeError(
                    "Local generation stopped. Re-run the identical command to resume "
                    "from completed checkpoint calls."
                ) from exc
            elapsed = time.perf_counter() - started
            if len(outputs) != len(retry):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(retry)} prompts"
                )

            next_retry = []
            failures = []
            for task, output in zip(retry, outputs):
                candidate = output.outputs[0]
                content = candidate.text
                try:
                    parsed = task.parser(content)
                except (TypeError, ValueError) as exc:
                    failure = {
                        "task": task,
                        "content": content,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                    if attempt < max_attempts:
                        next_retry.append(task)
                    else:
                        failures.append(failure)
                        checkpoint.save_failed(
                            task.identity,
                            {
                                "error_type": "invalid_structured_output",
                                "message": failure["message"],
                                "content": content,
                                "retryable": True,
                                "exhausted": True,
                                "attempt": attempt,
                            },
                        )
                    continue
                checkpoint.save_completed(
                    task.identity,
                    {
                        "content": content,
                        "parsed": parsed,
                        "prompt_tokens": len(
                            getattr(output, "prompt_token_ids", None) or []
                        ),
                        "completion_tokens": len(
                            getattr(candidate, "token_ids", None) or []
                        ),
                        "finish_reason": getattr(candidate, "finish_reason", None),
                        "attempt": attempt,
                        "batch_elapsed_seconds": elapsed,
                        "batch_size": len(retry),
                    },
                )
            if failures:
                print(
                    f"  {phase}: skipped {len(failures)} calls after "
                    f"{max_attempts} structured-output validation failures"
                )
            retry = next_retry
            if not retry:
                break


def _is_terminal_failure(
    checkpoint: CollectionCheckpoint,
    identity: dict,
) -> bool:
    failure = checkpoint.load_failed(identity)
    return failure is not None and bool(failure.get("exhausted"))


def _save_dependency_skip(
    checkpoint: CollectionCheckpoint,
    *,
    identity: dict,
    dependency: dict,
) -> None:
    if _is_terminal_failure(checkpoint, identity):
        return
    checkpoint.save_failed(
        identity,
        {
            "error_type": "dependency_skipped",
            "message": "Skipped because the required reflection call failed.",
            "retryable": False,
            "exhausted": True,
            "skipped": True,
            "dependency": dependency,
        },
    )


def _build_d1_reflection_tasks(
    judge: dict,
    cells: list[ResponseCell],
    prompt: PromptTemplate,
    criteria_text: str,
) -> list[GenerationTask]:
    return [
        GenerationTask(
            identity=_identity(
                judge=judge,
                cell=cell,
                design="01",
                stage="reflection",
            ),
            messages=prompt.messages(
                criteria=criteria_text,
                scenario=cell.scenario,
                response=cell.response,
            ),
            parser=_parse_d1_reflection,
        )
        for cell in cells
    ]


def _build_d1_judgment_tasks(
    judge: dict,
    cells: list[ResponseCell],
    prompt: PromptTemplate,
    criteria_text: str,
    criterion_ids: list[str],
    checkpoint: CollectionCheckpoint,
) -> list[GenerationTask]:
    tasks = []
    for cell in cells:
        reflection_identity = _identity(
            judge=judge,
            cell=cell,
            design="01",
            stage="reflection",
        )
        reflection = checkpoint.load_completed(reflection_identity)
        if reflection is None:
            if _is_terminal_failure(checkpoint, reflection_identity):
                _save_dependency_skip(
                    checkpoint,
                    identity=_identity(
                        judge=judge,
                        cell=cell,
                        design="01",
                        stage="judgment",
                    ),
                    dependency=reflection_identity,
                )
                continue
            raise RuntimeError(
                f"missing D1 reflection checkpoint: {reflection_identity}"
            )
        tasks.append(
            GenerationTask(
                identity=_identity(
                    judge=judge,
                    cell=cell,
                    design="01",
                    stage="judgment",
                ),
                messages=prompt.messages(
                    criteria=criteria_text,
                    scenario=cell.scenario,
                    response=cell.response,
                    reflection=reflection["parsed"]["reflection"],
                ),
                parser=lambda text, ids=criterion_ids: _parse_entries(
                    text,
                    container="ratings",
                    value_field="rating",
                    criterion_ids=ids,
                ),
            )
        )
    return tasks


def _build_d3_reflection_tasks(
    judge: dict,
    cells: list[ResponseCell],
    prompt: PromptTemplate,
    criteria_text: str,
    criterion_ids: list[str],
) -> list[GenerationTask]:
    return [
        GenerationTask(
            identity=_identity(
                judge=judge,
                cell=cell,
                design="03",
                stage="reflection",
            ),
            messages=prompt.messages(
                criteria=criteria_text,
                scenario=cell.scenario,
                response=cell.response,
            ),
            parser=lambda text, ids=criterion_ids: _parse_entries(
                text,
                container="reflections",
                value_field="reflection",
                criterion_ids=ids,
            ),
        )
        for cell in cells
    ]


def _build_d3_judgment_tasks(
    judge: dict,
    cells: list[ResponseCell],
    prompt: PromptTemplate,
    criteria: list[str],
    criterion_ids: list[str],
    checkpoint: CollectionCheckpoint,
) -> list[GenerationTask]:
    tasks = []
    for cell in cells:
        reflection_identity = _identity(
            judge=judge,
            cell=cell,
            design="03",
            stage="reflection",
        )
        reflection = checkpoint.load_completed(reflection_identity)
        if reflection is None:
            if _is_terminal_failure(checkpoint, reflection_identity):
                for criterion_id in criterion_ids:
                    _save_dependency_skip(
                        checkpoint,
                        identity=_identity(
                            judge=judge,
                            cell=cell,
                            design="03",
                            stage="judgment",
                            criterion_id=criterion_id,
                        ),
                        dependency=reflection_identity,
                    )
                continue
            raise RuntimeError(
                f"missing D3 reflection checkpoint: {reflection_identity}"
            )
        for criterion_id, criterion in zip(criterion_ids, criteria):
            tasks.append(
                GenerationTask(
                    identity=_identity(
                        judge=judge,
                        cell=cell,
                        design="03",
                        stage="judgment",
                        criterion_id=criterion_id,
                    ),
                    messages=prompt.messages(
                        criterion=f"{criterion_id}: {criterion}",
                        scenario=cell.scenario,
                        response=cell.response,
                        criterion_reflection=reflection["parsed"][criterion_id],
                    ),
                    parser=_parse_d3_rating,
                )
            )
    return tasks


def _expected_identities(
    judges: list[dict],
    cells: list[ResponseCell],
    criterion_ids: list[str],
) -> list[dict]:
    identities = []
    for judge in judges:
        for design in ("01", "03"):
            identities.extend(
                _identity(
                    judge=judge,
                    cell=cell,
                    design=design,
                    stage="reflection",
                )
                for cell in cells
            )
            if design == "01":
                identities.extend(
                    _identity(
                        judge=judge,
                        cell=cell,
                        design=design,
                        stage="judgment",
                    )
                    for cell in cells
                )
            else:
                identities.extend(
                    _identity(
                        judge=judge,
                        cell=cell,
                        design=design,
                        stage="judgment",
                        criterion_id=criterion_id,
                    )
                    for cell in cells
                    for criterion_id in criterion_ids
                )
    return identities


def _judge_complete(
    checkpoint: CollectionCheckpoint,
    judge: dict,
    cells: list[ResponseCell],
    criterion_ids: list[str],
) -> bool:
    return all(
        checkpoint.load_completed(identity) is not None
        or _is_terminal_failure(checkpoint, identity)
        for identity in _expected_identities([judge], cells, criterion_ids)
    )


def _call_outcomes(
    checkpoint: CollectionCheckpoint,
    identities: list[dict],
) -> tuple[int, list[dict], list[dict]]:
    completed = 0
    terminal = []
    unresolved = []
    for identity in identities:
        if checkpoint.load_completed(identity) is not None:
            completed += 1
            continue
        failure = checkpoint.load_failed(identity)
        if failure is not None and bool(failure.get("exhausted")):
            status = "skipped" if failure.get("skipped") else "failed"
            terminal.append({**identity, "status": status, "error": failure})
        else:
            unresolved.append(identity)
    return completed, terminal, unresolved


def _run_judge(
    *,
    judge: dict,
    base_info: dict,
    tokenizer,
    cells: list[ResponseCell],
    criteria: list[str],
    criterion_ids: list[str],
    templates: dict[str, PromptTemplate],
    schemas: dict[str, dict],
    generation: dict,
    checkpoint: CollectionCheckpoint,
    max_attempts: int,
    batch_size: int,
    verbose: bool,
) -> None:
    from pipeline.providers.vllm_local import VLLMEngineManager

    criteria_text = _criteria_payload(criteria, criterion_ids)
    print(f"\nJudge: {judge['name']} ({judge['model']['repo_id']})")
    max_model_len = 4096 if judge["model"]["repo_id"] == OLMO_2_REPO_ID else 8192
    with VLLMEngineManager(
        base_info["base_model_path"], max_model_len=max_model_len
    ) as llm:
        d1_reflections = _build_d1_reflection_tasks(
            judge, cells, templates["d1_reflection"], criteria_text
        )
        _execute_phase(
            llm=llm,
            tokenizer=tokenizer,
            tasks=d1_reflections,
            checkpoint=checkpoint,
            prompt_format=judge["prompt_format"],
            schema=schemas["d1_reflection"],
            max_tokens=int(generation["d1_reflection_max_tokens"]),
            temperature=float(generation["temperature"]),
            max_attempts=max_attempts,
            batch_size=batch_size,
            verbose=verbose,
        )
        d1_judgments = _build_d1_judgment_tasks(
            judge,
            cells,
            templates["d1_judgment"],
            criteria_text,
            criterion_ids,
            checkpoint,
        )
        _execute_phase(
            llm=llm,
            tokenizer=tokenizer,
            tasks=d1_judgments,
            checkpoint=checkpoint,
            prompt_format=judge["prompt_format"],
            schema=schemas["d1_judgment"],
            max_tokens=int(generation["d1_judgment_max_tokens"]),
            temperature=float(generation["temperature"]),
            max_attempts=max_attempts,
            batch_size=batch_size,
            verbose=verbose,
        )
        d3_reflections = _build_d3_reflection_tasks(
            judge,
            cells,
            templates["d3_reflection"],
            criteria_text,
            criterion_ids,
        )
        _execute_phase(
            llm=llm,
            tokenizer=tokenizer,
            tasks=d3_reflections,
            checkpoint=checkpoint,
            prompt_format=judge["prompt_format"],
            schema=schemas["d3_reflection"],
            max_tokens=int(generation["d3_reflection_max_tokens"]),
            temperature=float(generation["temperature"]),
            max_attempts=max_attempts,
            batch_size=batch_size,
            verbose=verbose,
        )
        d3_judgments = _build_d3_judgment_tasks(
            judge,
            cells,
            templates["d3_judgment"],
            criteria,
            criterion_ids,
            checkpoint,
        )
        _execute_phase(
            llm=llm,
            tokenizer=tokenizer,
            tasks=d3_judgments,
            checkpoint=checkpoint,
            prompt_format=judge["prompt_format"],
            schema=schemas["d3_judgment"],
            max_tokens=int(generation["d3_judgment_max_tokens"]),
            temperature=float(generation["temperature"]),
            max_attempts=max_attempts,
            batch_size=batch_size,
            verbose=verbose,
        )


def _build_artifacts(
    *,
    output_dir: Path,
    checkpoint: CollectionCheckpoint,
    config: dict,
    reference: dict,
    cells: list[ResponseCell],
    criteria: list[str],
    criterion_ids: list[str],
    templates: dict[str, PromptTemplate],
    responses_path: Path,
) -> list[dict]:
    cell_records = []
    stage_records = []
    raw_calls = []
    all_identities = _expected_identities(config["judges"], cells, criterion_ids)
    completed_calls, failure_records, unresolved = _call_outcomes(
        checkpoint, all_identities
    )
    if unresolved:
        raise RuntimeError(
            f"cannot build final artifacts with {len(unresolved)} unresolved calls"
        )

    for identity in all_identities:
        result = checkpoint.load_completed(identity)
        if result is None:
            continue
        stage_records.append(
            {
                **identity,
                "judge_name": identity["judge"],
                "evaluee_name": identity["evaluee"],
                "parsed": result["parsed"],
            }
        )
        raw_calls.append(
            {
                **identity,
                "judge_name": identity["judge"],
                "evaluee_name": identity["evaluee"],
                **result,
            }
        )

    for judge in config["judges"]:
        for cell in cells:
            for design in ("01", "03"):
                reflection_identity = _identity(
                    judge=judge,
                    cell=cell,
                    design=design,
                    stage="reflection",
                )
                reflection = checkpoint.load_completed(reflection_identity)
                if reflection is None:
                    continue

                if design == "01":
                    judgment_identity = _identity(
                        judge=judge,
                        cell=cell,
                        design=design,
                        stage="judgment",
                    )
                    judgment = checkpoint.load_completed(judgment_identity)
                    if judgment is None:
                        continue
                    ratings = judgment["parsed"]
                else:
                    ratings = {}
                    design_complete = True
                    for criterion_id in criterion_ids:
                        judgment_identity = _identity(
                            judge=judge,
                            cell=cell,
                            design=design,
                            stage="judgment",
                            criterion_id=criterion_id,
                        )
                        judgment = checkpoint.load_completed(judgment_identity)
                        if judgment is None:
                            design_complete = False
                            break
                        ratings[criterion_id] = judgment["parsed"]["rating"]
                    if not design_complete:
                        continue

                cell_records.append(
                    {
                        "record_key": (
                            f"s{cell.scenario_index}:{cell.evaluee_name}:"
                            f"{judge['name']}:d{design}"
                        ),
                        "scenario_index": cell.scenario_index,
                        "scenario": cell.scenario,
                        "evaluee_name": cell.evaluee_name,
                        "evaluee_model_id": cell.evaluee_model_id,
                        "response_sha256": cell.response_sha256,
                        "judge_name": judge["name"],
                        "judge_model": judge["model"]["repo_id"],
                        "judge_revision": judge["model"].get("revision"),
                        "design": design,
                        "ratings": ratings,
                    }
                )

    source_hash = _sha256_file(responses_path)
    manifest = {
        "runner_version": 2,
        "prompt_version": PROMPT_VERSION,
        "source_experiment": reference["source_experiment"],
        "sample_seed": reference["sample_seed"],
        "scenario_indices": reference["scenario_indices"],
        "criterion_ids": criterion_ids,
        "criteria": criteria,
        "constitution_sha256": reference["constitution_sha256"],
        "response_source": {
            "path": str(responses_path),
            "sha256": source_hash,
            "matches_reference_dataset": source_hash == reference["dataset_sha256"],
        },
        "evaluees": reference["evaluees"],
        "judges": config["judges"],
        "generation": config["generation"],
        "structured_outputs": "vLLM offline JSON schema",
        "prompts": {
            key: {
                "path": config["prompts"][key],
                "sha256": _sha256_file(_repo_path(config["prompts"][key])),
            }
            for key in templates
        },
        "inputs": [
            {
                "scenario_index": cell.scenario_index,
                "scenario_sha256": cell.scenario_sha256,
                "evaluee": cell.evaluee_name,
                "response_sha256": cell.response_sha256,
            }
            for cell in cells
        ],
        "planned_calls": len(all_identities),
    }
    manifest["run_fingerprint"] = _canonical_hash(manifest)[:16]
    actual_failures = [
        record for record in failure_records if record["status"] == "failed"
    ]
    dependency_skips = [
        record for record in failure_records if record["status"] == "skipped"
    ]
    manifest["call_outcomes"] = {
        "completed": completed_calls,
        "failed": len(actual_failures),
        "dependency_skipped": len(dependency_skips),
        "unresolved": 0,
    }

    save_records(output_dir / "stage_results.jsonl", stage_records)
    save_records(output_dir / "raw_calls.jsonl", raw_calls)
    save_records(output_dir / "failed_calls.jsonl", failure_records)
    failure_summary = {
        "actual_failure_count": len(actual_failures),
        "dependency_skip_count": len(dependency_skips),
        "actual_failures": actual_failures,
        "dependency_skips": dependency_skips,
    }
    (output_dir / "failure_summary.json").write_text(
        json.dumps(failure_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_records(
        output_dir / "inputs.jsonl",
        [
            {
                "scenario_index": cell.scenario_index,
                "scenario": cell.scenario,
                "evaluee_name": cell.evaluee_name,
                "evaluee_model_id": cell.evaluee_model_id,
                "response": cell.response,
                "scenario_sha256": cell.scenario_sha256,
                "response_sha256": cell.response_sha256,
            }
            for cell in cells
        ],
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    generate_reports(
        output_dir=output_dir,
        cell_records=cell_records,
        stage_records=stage_records,
        raw_calls=raw_calls,
        manifest=manifest,
    )
    return cell_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the matched D1/D3 experiment with four local 7B judges."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--responses",
        type=Path,
        help=(
            "Cached-response JSON containing the 10x5 reference cells. "
            "Defaults to the bundled path in config.json."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--judge",
        action="append",
        default=[],
        help="Run one configured judge by name; repeat for multiple judges.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--validate-input",
        action="store_true",
        help="Validate the fifty response hashes without downloading any model.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _print_plan(config: dict, reference: dict) -> None:
    scenarios = len(reference["scenario_indices"])
    evaluees = len(reference["evaluees"])
    judges = len(config["judges"])
    cells = scenarios * evaluees * judges
    d1_calls = cells * 2
    d3_calls = cells * 9
    print("D1/D3 local-judge plan")
    print(f"  scenarios: {scenarios}")
    print(f"  evaluees: {evaluees}")
    print(f"  judges: {judges}")
    print(f"  judge/evaluee/scenario cells per design: {cells}")
    print(f"  D1 calls: {d1_calls}")
    print(f"  D3 calls: {d3_calls}")
    print(f"  total calls: {d1_calls + d3_calls}")
    for judge in config["judges"]:
        model = judge["model"]
        print(
            f"  - {judge['name']}: {model['repo_id']}@{model.get('revision')} "
            f"({judge['prompt_format']})"
        )


def main() -> int:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    config_path = _repo_path(args.config)
    config = _load_config(config_path)
    reference_path = _repo_path(config["reference_manifest"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    _print_plan(config, reference)
    if args.plan:
        return 0
    configured_responses = config.get("responses")
    if args.responses is not None:
        responses_path = args.responses.expanduser().resolve()
    elif configured_responses:
        responses_path = _repo_path(str(configured_responses)).resolve()
    else:
        raise ValueError(
            "--responses is required when config.json has no bundled response path"
        )
    if not responses_path.exists():
        raise FileNotFoundError(
            f"cached response source does not exist: {responses_path}"
        )

    constitution_path = _repo_path(config["constitution"])
    if _sha256_file(constitution_path) != reference["constitution_sha256"]:
        raise ValueError(
            "kindness constitution differs from the cached-response reference"
        )
    criteria, criterion_ids = _load_criteria(constitution_path)
    cells = _load_response_cells(responses_path, reference)
    if args.validate_input:
        print(
            f"Validated {len(cells)} exact response cells from "
            f"{len(reference['scenario_indices'])} scenarios."
        )
        return 0
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    templates = {
        key: PromptTemplate.load(_repo_path(path))
        for key, path in config["prompts"].items()
    }
    schemas = _schemas(criterion_ids)
    assignments = [
        {
            "scenario_index": cell.scenario_index,
            "judge": judge["name"],
            "evaluee": cell.evaluee_name,
        }
        for judge in config["judges"]
        for cell in cells
    ]
    context = {
        "prompt_version": PROMPT_VERSION,
        "config": config,
        "reference_manifest": reference,
        "input_cells": [
            {
                "scenario_index": cell.scenario_index,
                "evaluee": cell.evaluee_name,
                "scenario_sha256": cell.scenario_sha256,
                "response_sha256": cell.response_sha256,
            }
            for cell in cells
        ],
        "prompt_hashes": {
            key: _sha256_file(_repo_path(config["prompts"][key])) for key in templates
        },
        "schemas": schemas,
    }
    checkpoint = CollectionCheckpoint(output_dir / "checkpoint")
    checkpoint.initialize_or_resume(context=context, assignments=assignments)

    selected = [
        judge
        for judge in config["judges"]
        if not args.judge or judge["name"] in args.judge
    ]
    if not selected:
        raise ValueError("no configured judges matched --judge")
    unknown = sorted(set(args.judge) - {judge["name"] for judge in selected})
    if unknown:
        raise ValueError(f"unknown judge names: {unknown}")
    max_attempts = (
        int(args.max_attempts)
        if args.max_attempts is not None
        else int(config["max_attempts"])
    )
    if max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")

    pending_judges = [
        judge
        for judge in selected
        if not _judge_complete(checkpoint, judge, cells, criterion_ids)
    ]
    if pending_judges:
        from pipeline.providers.vllm_local import group_models_for_vllm

        for judge in pending_judges:
            local_groups, local_tokenizers, remote = group_models_for_vllm(
                {judge["name"]: judge["model"]}
            )
            if remote:
                raise ValueError(f"this experiment accepts only local models: {remote}")
            match = [
                (base_key, info)
                for base_key, info in local_groups.items()
                if info.get("base_only") == judge["name"]
            ]
            if len(match) != 1:
                raise RuntimeError(
                    f"could not resolve one local base model for {judge['name']}"
                )
            base_key, base_info = match[0]
            _run_judge(
                judge=judge,
                base_info=base_info,
                tokenizer=local_tokenizers[base_key],
                cells=cells,
                criteria=criteria,
                criterion_ids=criterion_ids,
                templates=templates,
                schemas=schemas,
                generation=config["generation"],
                checkpoint=checkpoint,
                max_attempts=max_attempts,
                batch_size=args.batch_size,
                verbose=args.verbose,
            )
    else:
        print("Selected judges are already complete in the checkpoint.")

    all_identities = _expected_identities(config["judges"], cells, criterion_ids)
    completed, terminal, unresolved = _call_outcomes(checkpoint, all_identities)
    actual_failed = sum(record["status"] == "failed" for record in terminal)
    dependency_skipped = sum(record["status"] == "skipped" for record in terminal)
    print(
        "Checkpoint progress: "
        f"{completed} complete, {actual_failed} failed, "
        f"{dependency_skipped} dependency-skipped, "
        f"{len(unresolved)} unresolved ({len(all_identities)} planned)"
    )
    if unresolved:
        print(
            "The selected judge subset is complete. Run the remaining judges with the "
            "same responses and output directory to finalize the matrices."
        )
        return 0

    cell_results_path = output_dir / "cell_results.jsonl"
    if checkpoint.has_finalized_output():
        cell_records = checkpoint.load_finalized_output(cell_results_path)
    else:
        cell_records = _build_artifacts(
            output_dir=output_dir,
            checkpoint=checkpoint,
            config=config,
            reference=reference,
            cells=cells,
            criteria=criteria,
            criterion_ids=criterion_ids,
            templates=templates,
            responses_path=responses_path,
        )
        checkpoint.finalize(cell_results_path, cell_records)

    # Rebuild derived artifacts on resume so reporting is recoverable even if an
    # earlier process stopped immediately after finalizing cell_results.jsonl.
    _build_artifacts(
        output_dir=output_dir,
        checkpoint=checkpoint,
        config=config,
        reference=reference,
        cells=cells,
        criteria=criteria,
        criterion_ids=criterion_ids,
        templates=templates,
        responses_path=responses_path,
    )
    print(f"Experiment complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
