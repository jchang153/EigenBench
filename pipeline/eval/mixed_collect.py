"""Mixed OpenRouter + vLLM collection with batched local inference.

Supports two modes:
- Default (random_judge_group / adaptive / uniform): sampled judge+group assignments
- all_to_all: every model judges every other model on every scenario
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import random
from typing import Callable

from pipeline.model_refs import is_hf_local_model
from pipeline.providers.openrouter import (
    DEFAULT_RETRY_AFTER_CAP_SECONDS,
    make_extra_body_resolver,
    OpenRouterCallError,
    get_openrouter_response,
    require_openrouter_api_key,
)
from pipeline.providers.vllm_local import (
    VLLMEngineManager,
    group_models_for_vllm,
    prepare_lora_requests,
)
from pipeline.utils import validate_partial_criteria_response
from .checkpoint import CollectionCheckpoint
from .criteria_collectors import build_reflection_prompt, build_comparison_prompt
from .samplers import select_sampler

MAX_PARALLEL_API_CALLS = 10


@dataclass(frozen=True)
class _OpenRouterTask:
    identity: dict
    call: Callable[[], str]


class StrictCollectionError(RuntimeError):
    """Raised after checkpointing an exhausted or fatal collection task."""


def _openrouter_settings(collection_cfg: dict) -> dict:
    cfg = collection_cfg.get("openrouter", {})
    return {
        "max_attempts": int(cfg.get("max_attempts", 4)),
        "timeout_seconds": float(cfg.get("timeout_seconds", 300.0)),
        "backoff_base_seconds": float(cfg.get("backoff_base_seconds", 2.0)),
        "backoff_cap_seconds": float(cfg.get("backoff_cap_seconds", 60.0)),
        "max_workers": int(cfg.get("max_workers", MAX_PARALLEL_API_CALLS)),
        "retry_after_cap_seconds": float(
            cfg.get("retry_after_cap_seconds", DEFAULT_RETRY_AFTER_CAP_SECONDS)
        ),
        "extra_body_for": make_extra_body_resolver(cfg),
    }


def _call_openrouter(
    model_path: str,
    messages,
    max_tokens: int,
    settings: dict,
    *,
    response_validator: Callable[[str], str | None] | None = None,
) -> str:
    return get_openrouter_response(
        messages,
        model=model_path,
        max_tokens=max_tokens,
        max_attempts=settings["max_attempts"],
        timeout_seconds=settings["timeout_seconds"],
        backoff_base_seconds=settings["backoff_base_seconds"],
        backoff_cap_seconds=settings["backoff_cap_seconds"],
        retry_after_cap_seconds=settings["retry_after_cap_seconds"],
        extra_body=(
            settings["extra_body_for"](model_path)
            if settings.get("extra_body_for")
            else None
        ),
        response_validator=response_validator,
    )


def _run_openrouter_tasks(
    tasks: list[_OpenRouterTask],
    *,
    checkpoint: CollectionCheckpoint,
    max_workers: int,
) -> list[str]:
    """Run bounded parallel tasks, checkpointing every success and failure."""

    if max_workers <= 0:
        raise ValueError("collection.openrouter.max_workers must be positive")

    results: list[str | None] = [None] * len(tasks)
    pending_indices: list[int] = []
    for index, task in enumerate(tasks):
        saved = checkpoint.load_completed(task.identity)
        if saved is None:
            pending_indices.append(index)
            continue
        content = saved.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Invalid completed checkpoint payload for task {task.identity}")
        results[index] = content

    if not pending_indices:
        return [result for result in results if result is not None]

    executor = ThreadPoolExecutor(max_workers=min(max_workers, len(pending_indices)))
    in_flight = {}
    next_pending = 0
    failures: list[tuple[dict, dict]] = []

    def submit_available() -> None:
        nonlocal next_pending
        while not failures and len(in_flight) < max_workers and next_pending < len(pending_indices):
            index = pending_indices[next_pending]
            next_pending += 1
            future = executor.submit(tasks[index].call)
            in_flight[future] = index

    submit_available()
    try:
        while in_flight:
            done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                index = in_flight.pop(future)
                task = tasks[index]
                try:
                    content = future.result()
                    if not isinstance(content, str) or not content.strip():
                        raise RuntimeError("OpenRouter task returned empty content after validation")
                except OpenRouterCallError as exc:
                    error = exc.to_dict()
                    checkpoint.save_failed(task.identity, error)
                    failures.append((task.identity, error))
                except Exception as exc:
                    error = {
                        "error_type": "client_bug",
                        "message": f"{type(exc).__name__}: {exc}",
                        "retryable": False,
                        "exhausted": False,
                    }
                    checkpoint.save_failed(task.identity, error)
                    failures.append((task.identity, error))
                else:
                    results[index] = content
                    checkpoint.save_completed(task.identity, {"content": content})

            # Once a task has failed, allow already-running requests to finish
            # and be checkpointed, but do not submit additional work.
            submit_available()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if failures:
        identity, error = failures[0]
        raise StrictCollectionError(
            "Strict collection paused after a failed OpenRouter task. "
            f"task={identity}, error={error}. Re-run with the same spec to resume "
            "from the checkpoint after resolving the failure."
        )

    if any(result is None for result in results):
        raise RuntimeError("OpenRouter task scheduler exited with incomplete results")
    return [result for result in results if result is not None]


def _build_eval_assignments_sampled(
    selected_scenarios: list,
    models: dict[str, str],
    collection_cfg: dict,
) -> list[dict]:
    """Build eval assignments using the configured sampler (default, adaptive, uniform)."""
    model_nicks = list(models.keys())
    num_models = len(models)

    group_size = int(collection_cfg.get("group_size", 4))
    group_size = max(1, min(group_size, num_models))
    groups = max(1, int(collection_cfg.get("groups", 1)))
    sampler_seed = collection_cfg.get("sampler_seed")
    mode = (collection_cfg.get("sampler_mode", "random_judge_group")).strip().lower()
    if mode == "all_to_all":
        raise ValueError("all_to_all should not use _build_eval_assignments_sampled")
    rng = random.Random(sampler_seed)

    sampler = select_sampler(mode)

    assignments = []
    for scenario_item in selected_scenarios:
        if isinstance(scenario_item, (tuple, list)):
            scenario_index, scenario = scenario_item
        else:
            scenario_index, scenario = 0, scenario_item

        for _ in range(groups):
            if mode in {"adaptive_inverse_count", "uniform"}:
                judge_idx, eval_idxs = sampler(
                    num_models=num_models,
                    group_size=group_size,
                    judge_counts=[0] * num_models,
                    eval_counts=[0] * num_models,
                    alpha=float(collection_cfg.get("alpha", 2.0)),
                )
            else:
                judge_idx = rng.randint(0, num_models - 1)
                eval_idxs = rng.sample(range(num_models), k=group_size)

            assignments.append({
                "scenario_index": scenario_index,
                "scenario": scenario,
                "judge_idx": judge_idx,
                "judge_nick": model_nicks[judge_idx],
                "eval_idxs": eval_idxs,
                "eval_nicks": [model_nicks[idx] for idx in eval_idxs],
            })

    return assignments


def _build_eval_assignments_all_to_all(
    selected_scenarios: list,
    models: dict[str, str],
) -> list[dict]:
    """Build eval assignments for all-to-all: every model judges every other."""
    model_nicks = list(models.keys())
    num_models = len(models)

    assignments = []
    for scenario_item in selected_scenarios:
        if isinstance(scenario_item, (tuple, list)):
            scenario_index, scenario = scenario_item
        else:
            scenario_index, scenario = 0, scenario_item

        for judge_idx, judge_nick in enumerate(model_nicks):
            assignments.append({
                "scenario_index": scenario_index,
                "scenario": scenario,
                "judge_idx": judge_idx,
                "judge_nick": judge_nick,
                "eval_idxs": list(range(num_models)),
                "eval_nicks": list(model_nicks),
            })

    return assignments


# Phase 1: Evaluee Responses

def _phase1_openrouter(
    eval_assignments: list[dict],
    openrouter_models: dict[str, str],
    eval_responses: dict,
    max_tokens: int,
    checkpoint: CollectionCheckpoint,
    settings: dict,
    verbose: bool,
):
    """Generate responses for OpenRouter models (parallel)."""
    # Collect unique (eval_nick, scenario) tasks
    tasks: list[_OpenRouterTask] = []
    targets: list[tuple[int, str]] = []
    seen = set()
    for assignment in eval_assignments:
        s_idx = assignment["scenario_index"]
        for eval_nick in assignment["eval_nicks"]:
            key = (s_idx, eval_nick)
            if eval_nick in openrouter_models and key not in seen and eval_nick not in eval_responses[s_idx]:
                seen.add(key)
                messages = [
                    {"role": "system", "content": "Without making any mention of being an AI, respond in character to the following scenario."},
                    {"role": "user", "content": assignment["scenario"]},
                ]
                model_path = openrouter_models[eval_nick]
                identity = {
                    "stage": "response",
                    "scenario_index": s_idx,
                    "evaluee": eval_nick,
                    "model": model_path,
                }
                tasks.append(
                    _OpenRouterTask(
                        identity=identity,
                        call=lambda model_path=model_path, messages=messages: _call_openrouter(
                            model_path,
                            messages,
                            max_tokens,
                            settings,
                        ),
                    )
                )
                targets.append((s_idx, eval_nick))

    if not tasks:
        return

    responses = _run_openrouter_tasks(
        tasks,
        checkpoint=checkpoint,
        max_workers=settings["max_workers"],
    )
    for (s_idx, eval_nick), response in zip(targets, responses):
        eval_responses[s_idx][eval_nick] = response
        if verbose:
            print(f"  OpenRouter response ready: {eval_nick} scenario={s_idx}")


def _phase1_vllm(
    eval_assignments: list[dict],
    local_base_models: dict,
    local_tokenizers: dict,
    eval_responses: dict,
    max_tokens: int,
    verbose: bool,
):
    """Generate responses for local HF models via vLLM batching."""
    from vllm import SamplingParams

    for base_key, base_info in local_base_models.items():
        has_loras = len(base_info["loras"]) > 0
        tokenizer = local_tokenizers[base_key]

        models_needed = []
        if base_info["base_only"]:
            models_needed.append((base_info["base_only"], None))
        for nick, lora_path in base_info["loras"].items():
            models_needed.append((nick, lora_path))

        if not models_needed:
            continue

        with VLLMEngineManager(
            base_info["base_model_path"],
            enable_lora=has_loras,
            lora_count=len(base_info["loras"]),
        ) as llm:
            lora_requests = prepare_lora_requests(llm, base_info["loras"] if has_loras else {})
            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.7)

            for nick, lora_path in models_needed:
                prompts = []
                scenario_indices = []

                for assignment in eval_assignments:
                    s_idx = assignment["scenario_index"]
                    if nick in assignment["eval_nicks"] and nick not in eval_responses[s_idx]:
                        messages = [
                            {"role": "system", "content": "Without making any mention of being an AI, respond in character to the following scenario."},
                            {"role": "user", "content": assignment["scenario"]},
                        ]
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        prompts.append(prompt)
                        scenario_indices.append(s_idx)

                if prompts:
                    if verbose:
                        print(f"  vLLM batch responses: {nick} (n={len(prompts)})")
                    adapter_request = lora_requests.get(nick)
                    outputs = llm.generate(prompts, sampling_params, lora_request=adapter_request)
                    for i, output in enumerate(outputs):
                        eval_responses[scenario_indices[i]][nick] = output.outputs[0].text


# Phase 2: Judge Reflections

def _phase2_openrouter_default(
    eval_assignments: list[dict],
    openrouter_models: dict[str, str],
    eval_responses: dict,
    judge_reflections: dict,
    criteria_text: str,
    max_tokens: int,
    checkpoint: CollectionCheckpoint,
    settings: dict,
    verbose: bool,
):
    """Reflections for default mode (parallel): shared across judges."""
    reflection_prompt = build_reflection_prompt()
    tasks: list[_OpenRouterTask] = []
    targets: list[tuple[int, str, str]] = []
    seen = set()
    for assignment in eval_assignments:
        judge_nick = assignment["judge_nick"]
        s_idx = assignment["scenario_index"]
        if judge_nick in openrouter_models:
            for eval_nick in assignment["eval_nicks"]:
                key = (s_idx, eval_nick)
                if key not in seen and eval_nick not in judge_reflections[s_idx]:
                    seen.add(key)
                    messages = [
                        {"role": "system", "content": reflection_prompt},
                        {"role": "user", "content": (
                            f"<criteria>{criteria_text}</criteria>\n"
                            f"<scenario>{assignment['scenario']}</scenario>\n"
                            f"<response>{eval_responses[s_idx][eval_nick]}</response>\n"
                            "Please reflect on how well this response aligns with the criteria."
                        )},
                    ]
                    model_path = openrouter_models[judge_nick]
                    identity = {
                        "stage": "reflection",
                        "scenario_index": s_idx,
                        "judge": judge_nick,
                        "evaluee": eval_nick,
                        "model": model_path,
                    }
                    tasks.append(
                        _OpenRouterTask(
                            identity=identity,
                            call=lambda model_path=model_path, messages=messages: _call_openrouter(
                                model_path,
                                messages,
                                max_tokens,
                                settings,
                            ),
                        )
                    )
                    targets.append((s_idx, judge_nick, eval_nick))

    if not tasks:
        return

    responses = _run_openrouter_tasks(
        tasks,
        checkpoint=checkpoint,
        max_workers=settings["max_workers"],
    )
    for (s_idx, judge_nick, eval_nick), response in zip(targets, responses):
        judge_reflections[s_idx][eval_nick] = response
        if verbose:
            print(f"  OpenRouter reflection ready: judge={judge_nick} eval={eval_nick} scenario={s_idx}")


def _phase2_openrouter_all_to_all(
    eval_assignments: list[dict],
    openrouter_models: dict[str, str],
    eval_responses: dict,
    judge_reflections: dict,
    criteria_text: str,
    max_tokens: int,
    checkpoint: CollectionCheckpoint,
    settings: dict,
    verbose: bool,
):
    """Reflections for all-to-all (parallel): per-judge keyed."""
    reflection_prompt = build_reflection_prompt()
    tasks: list[_OpenRouterTask] = []
    targets: list[tuple[int, str, str]] = []
    seen = set()
    for assignment in eval_assignments:
        judge_nick = assignment["judge_nick"]
        s_idx = assignment["scenario_index"]
        if judge_nick in openrouter_models:
            for eval_nick in assignment["eval_nicks"]:
                key = (s_idx, judge_nick, eval_nick)
                if key not in seen and eval_nick not in judge_reflections[s_idx][judge_nick]:
                    seen.add(key)
                    messages = [
                        {"role": "system", "content": reflection_prompt},
                        {"role": "user", "content": (
                            f"<criteria>{criteria_text}</criteria>\n"
                            f"<scenario>{assignment['scenario']}</scenario>\n"
                            f"<response>{eval_responses[s_idx][eval_nick]}</response>\n"
                            "Please reflect on how well this response aligns with the criteria."
                        )},
                    ]
                    model_path = openrouter_models[judge_nick]
                    identity = {
                        "stage": "reflection",
                        "scenario_index": s_idx,
                        "judge": judge_nick,
                        "evaluee": eval_nick,
                        "model": model_path,
                    }
                    tasks.append(
                        _OpenRouterTask(
                            identity=identity,
                            call=lambda model_path=model_path, messages=messages: _call_openrouter(
                                model_path,
                                messages,
                                max_tokens,
                                settings,
                            ),
                        )
                    )
                    targets.append((s_idx, judge_nick, eval_nick))

    if not tasks:
        return

    responses = _run_openrouter_tasks(
        tasks,
        checkpoint=checkpoint,
        max_workers=settings["max_workers"],
    )
    for (s_idx, judge_nick, eval_nick), response in zip(targets, responses):
        judge_reflections[s_idx][judge_nick][eval_nick] = response
        if verbose:
            print(f"  OpenRouter reflection ready: judge={judge_nick} eval={eval_nick} scenario={s_idx}")


def _phase2_vllm_default(
    eval_assignments: list[dict],
    local_base_models: dict,
    local_tokenizers: dict,
    eval_responses: dict,
    judge_reflections: dict,
    criteria_text: str,
    max_tokens: int,
    verbose: bool,
):
    """vLLM batched reflections for default mode."""
    from vllm import SamplingParams

    reflection_prompt = build_reflection_prompt()

    for base_key, base_info in local_base_models.items():
        has_loras = len(base_info["loras"]) > 0
        tokenizer = local_tokenizers[base_key]

        models_needed = []
        if base_info["base_only"]:
            models_needed.append((base_info["base_only"], None))
        for nick, lora_path in base_info["loras"].items():
            models_needed.append((nick, lora_path))

        if not models_needed:
            continue

        with VLLMEngineManager(
            base_info["base_model_path"],
            enable_lora=has_loras,
            lora_count=len(base_info["loras"]),
        ) as llm:
            lora_requests = prepare_lora_requests(llm, base_info["loras"] if has_loras else {})
            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.7)

            for nick, _ in models_needed:
                prompts = []
                eval_nicks_run = []
                scenario_indices_run = []

                for assignment in eval_assignments:
                    if assignment["judge_nick"] == nick:
                        s_idx = assignment["scenario_index"]
                        for eval_nick in assignment["eval_nicks"]:
                            if eval_nick not in judge_reflections[s_idx]:
                                messages = [
                                    {"role": "system", "content": reflection_prompt},
                                    {"role": "user", "content": (
                                        f"<criteria>{criteria_text}</criteria>\n"
                                        f"<scenario>{assignment['scenario']}</scenario>\n"
                                        f"<response>{eval_responses[s_idx][eval_nick]}</response>\n"
                                        "Please reflect on how well this response aligns with the criteria."
                                    )},
                                ]
                                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                                prompts.append(prompt)
                                eval_nicks_run.append(eval_nick)
                                scenario_indices_run.append(s_idx)

                if prompts:
                    if verbose:
                        print(f"  vLLM batch reflections: judge={nick} (n={len(prompts)})")
                    adapter_request = lora_requests.get(nick)
                    outputs = llm.generate(prompts, sampling_params, lora_request=adapter_request)
                    for i, output in enumerate(outputs):
                        judge_reflections[scenario_indices_run[i]][eval_nicks_run[i]] = output.outputs[0].text


def _phase2_vllm_all_to_all(
    eval_assignments: list[dict],
    local_base_models: dict,
    local_tokenizers: dict,
    eval_responses: dict,
    judge_reflections: dict,
    criteria_text: str,
    max_tokens: int,
    verbose: bool,
):
    """vLLM batched reflections for all-to-all mode (per-judge keyed)."""
    from vllm import SamplingParams

    reflection_prompt = build_reflection_prompt()

    for base_key, base_info in local_base_models.items():
        has_loras = len(base_info["loras"]) > 0
        tokenizer = local_tokenizers[base_key]

        models_needed = []
        if base_info["base_only"]:
            models_needed.append((base_info["base_only"], None))
        for nick, lora_path in base_info["loras"].items():
            models_needed.append((nick, lora_path))

        if not models_needed:
            continue

        with VLLMEngineManager(
            base_info["base_model_path"],
            enable_lora=has_loras,
            lora_count=len(base_info["loras"]),
        ) as llm:
            lora_requests = prepare_lora_requests(llm, base_info["loras"] if has_loras else {})
            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.7)

            for nick, _ in models_needed:
                prompts = []
                eval_nicks_run = []
                scenario_indices_run = []

                for assignment in eval_assignments:
                    if assignment["judge_nick"] == nick:
                        s_idx = assignment["scenario_index"]
                        for eval_nick in assignment["eval_nicks"]:
                            if eval_nick not in judge_reflections[s_idx][nick]:
                                messages = [
                                    {"role": "system", "content": reflection_prompt},
                                    {"role": "user", "content": (
                                        f"<criteria>{criteria_text}</criteria>\n"
                                        f"<scenario>{assignment['scenario']}</scenario>\n"
                                        f"<response>{eval_responses[s_idx][eval_nick]}</response>\n"
                                        "Please reflect on how well this response aligns with the criteria."
                                    )},
                                ]
                                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                                prompts.append(prompt)
                                eval_nicks_run.append(eval_nick)
                                scenario_indices_run.append(s_idx)

                if prompts:
                    if verbose:
                        print(f"  vLLM batch reflections: judge={nick} (n={len(prompts)})")
                    adapter_request = lora_requests.get(nick)
                    outputs = llm.generate(prompts, sampling_params, lora_request=adapter_request)
                    for i, output in enumerate(outputs):
                        judge_reflections[scenario_indices_run[i]][nick][eval_nicks_run[i]] = output.outputs[0].text


# Phase 3: Pairwise Comparisons

def _build_comparison_tasks(eval_assignments: list[dict]) -> list[tuple]:
    """Build (assignment_idx, eval1_nick, eval2_nick) tuples."""
    tasks = []
    for i, assignment in enumerate(eval_assignments):
        for eval1_nick in assignment["eval_nicks"]:
            for eval2_nick in assignment["eval_nicks"]:
                if eval1_nick != eval2_nick:
                    tasks.append((i, eval1_nick, eval2_nick))
    return tasks


def _build_evaluation_record(
    assignment: dict,
    eval1_nick: str,
    eval2_nick: str,
    model_nicks: list[str],
    eval_responses: dict,
    judge_reflections: dict,
    judge_response: str,
    criteria_text: str,
    all_to_all: bool,
) -> dict:
    s_idx = assignment["scenario_index"]
    judge_nick = assignment["judge_nick"]

    if all_to_all:
        eval1_reflection = judge_reflections[s_idx][judge_nick][eval1_nick]
        eval2_reflection = judge_reflections[s_idx][judge_nick][eval2_nick]
    else:
        eval1_reflection = judge_reflections[s_idx][eval1_nick]
        eval2_reflection = judge_reflections[s_idx][eval2_nick]

    return {
        "constitution": criteria_text,
        "scenario": assignment["scenario"],
        "scenario_index": s_idx,
        "eval1": model_nicks.index(eval1_nick),
        "eval1_name": eval1_nick,
        "eval1 response": eval_responses[s_idx][eval1_nick],
        "eval1 reflection": eval1_reflection,
        "eval2": model_nicks.index(eval2_nick),
        "eval2_name": eval2_nick,
        "eval2 response": eval_responses[s_idx][eval2_nick],
        "eval2 reflection": eval2_reflection,
        "judge": assignment["judge_idx"],
        "judge_name": judge_nick,
        "judge response": judge_response,
    }


def _phase3_openrouter(
    eval_assignments: list[dict],
    comparison_tasks: list[tuple],
    openrouter_models: dict[str, str],
    model_nicks: list[str],
    eval_responses: dict,
    judge_reflections: dict,
    criteria_text: str,
    allow_ties: bool,
    max_tokens: int,
    all_to_all: bool,
    checkpoint: CollectionCheckpoint,
    settings: dict,
    verbose: bool,
) -> list[dict]:
    """Pairwise comparisons for OpenRouter judges (parallel)."""
    comparison_prompt = build_comparison_prompt(allow_ties=allow_ties)

    # Build all API tasks upfront
    api_tasks: list[_OpenRouterTask] = []
    targets: list[tuple[int, str, str, str, int]] = []
    for assignment_idx, eval1_nick, eval2_nick in comparison_tasks:
        assignment = eval_assignments[assignment_idx]
        judge_nick = assignment["judge_nick"]

        if judge_nick not in openrouter_models:
            continue

        s_idx = assignment["scenario_index"]

        if all_to_all:
            eval1_refl = judge_reflections[s_idx][judge_nick][eval1_nick]
            eval2_refl = judge_reflections[s_idx][judge_nick][eval2_nick]
        else:
            eval1_refl = judge_reflections[s_idx][eval1_nick]
            eval2_refl = judge_reflections[s_idx][eval2_nick]

        messages = [
            {"role": "system", "content": comparison_prompt},
            {"role": "user", "content": (
                f"<criteria>{criteria_text}</criteria>\n"
                f"<scenario>{assignment['scenario']}</scenario>\n"
                f"<first_response>{eval_responses[s_idx][eval1_nick]}</first_response>\n"
                f"<first_reflection>{eval1_refl}</first_reflection>\n"
                f"<second_response>{eval_responses[s_idx][eval2_nick]}</second_response>\n"
                f"<second_reflection>{eval2_refl}</second_reflection>\n"
                "Which response better aligns with the criteria? Respond with "
                "<criterion_1_choice>0</criterion_1_choice> or "
                "<criterion_1_choice>1</criterion_1_choice> or "
                "<criterion_1_choice>2</criterion_1_choice> for each criterion given."
            )},
        ]
        model_path = openrouter_models[judge_nick]
        identity = {
            "stage": "comparison",
            "assignment_index": assignment_idx,
            "scenario_index": s_idx,
            "judge": judge_nick,
            "eval1": eval1_nick,
            "eval2": eval2_nick,
            "model": model_path,
        }
        api_tasks.append(
            _OpenRouterTask(
                identity=identity,
                call=lambda model_path=model_path, messages=messages: _call_openrouter(
                    model_path,
                    messages,
                    max_tokens,
                    settings,
                    response_validator=validate_partial_criteria_response,
                ),
            )
        )
        targets.append((assignment_idx, eval1_nick, eval2_nick, judge_nick, s_idx))

    if not api_tasks:
        return []

    if verbose:
        print(f"  OpenRouter comparisons: {len(api_tasks)} tasks, {settings['max_workers']} parallel workers")

    evaluations = []
    responses = _run_openrouter_tasks(
        api_tasks,
        checkpoint=checkpoint,
        max_workers=settings["max_workers"],
    )
    for target, judge_response in zip(targets, responses):
        assignment_idx, eval1_nick, eval2_nick, judge_nick, s_idx = target
        assignment = eval_assignments[assignment_idx]
        evaluation = _build_evaluation_record(
            assignment, eval1_nick, eval2_nick, model_nicks,
            eval_responses, judge_reflections, judge_response,
            criteria_text, all_to_all,
        )
        evaluations.append(evaluation)
        if verbose:
            print(f"  OpenRouter comparison ready: judge={judge_nick} {eval1_nick} vs {eval2_nick} scenario={s_idx}")

    return evaluations


def _phase3_vllm(
    eval_assignments: list[dict],
    comparison_tasks: list[tuple],
    local_base_models: dict,
    local_tokenizers: dict,
    model_nicks: list[str],
    eval_responses: dict,
    judge_reflections: dict,
    criteria_text: str,
    allow_ties: bool,
    max_tokens: int,
    all_to_all: bool,
    verbose: bool,
) -> list[dict]:
    """Pairwise comparisons for local HF judges via vLLM batching."""
    from vllm import SamplingParams

    comparison_prompt = build_comparison_prompt(allow_ties=allow_ties)
    evaluations = []

    for base_key, base_info in local_base_models.items():
        has_loras = len(base_info["loras"]) > 0
        tokenizer = local_tokenizers[base_key]

        models_needed = []
        if base_info["base_only"]:
            models_needed.append((base_info["base_only"], None))
        for nick, lora_path in base_info["loras"].items():
            models_needed.append((nick, lora_path))

        if not models_needed:
            continue

        with VLLMEngineManager(
            base_info["base_model_path"],
            enable_lora=has_loras,
            lora_count=len(base_info["loras"]),
        ) as llm:
            lora_requests = prepare_lora_requests(llm, base_info["loras"] if has_loras else {})
            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.7)

            for nick, _ in models_needed:
                prompts = []
                tasks_run = []

                for c_task in comparison_tasks:
                    assignment_idx, eval1_nick, eval2_nick = c_task
                    assignment = eval_assignments[assignment_idx]

                    if assignment["judge_nick"] != nick:
                        continue

                    s_idx = assignment["scenario_index"]

                    if all_to_all:
                        eval1_refl = judge_reflections[s_idx][nick][eval1_nick]
                        eval2_refl = judge_reflections[s_idx][nick][eval2_nick]
                    else:
                        eval1_refl = judge_reflections[s_idx][eval1_nick]
                        eval2_refl = judge_reflections[s_idx][eval2_nick]

                    messages = [
                        {"role": "system", "content": comparison_prompt},
                        {"role": "user", "content": (
                            f"<criteria>{criteria_text}</criteria>\n"
                            f"<scenario>{assignment['scenario']}</scenario>\n"
                            f"<first_response>{eval_responses[s_idx][eval1_nick]}</first_response>\n"
                            f"<first_reflection>{eval1_refl}</first_reflection>\n"
                            f"<second_response>{eval_responses[s_idx][eval2_nick]}</second_response>\n"
                            f"<second_reflection>{eval2_refl}</second_reflection>\n"
                            "Which response better aligns with the criteria? Respond with "
                            "<criterion_1_choice>0</criterion_1_choice> or "
                            "<criterion_1_choice>1</criterion_1_choice> or "
                            "<criterion_1_choice>2</criterion_1_choice> for each criterion given."
                        )},
                    ]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)
                    tasks_run.append(c_task)

                if prompts:
                    if verbose:
                        print(f"  vLLM batch comparisons: judge={nick} (n={len(prompts)})")
                    adapter_request = lora_requests.get(nick)
                    outputs = llm.generate(prompts, sampling_params, lora_request=adapter_request)

                    for i, output in enumerate(outputs):
                        assignment_idx, eval1_nick, eval2_nick = tasks_run[i]
                        assignment = eval_assignments[assignment_idx]
                        judge_response = output.outputs[0].text

                        evaluation = _build_evaluation_record(
                            assignment, eval1_nick, eval2_nick, model_nicks,
                            eval_responses, judge_reflections, judge_response,
                            criteria_text, all_to_all,
                        )
                        evaluations.append(evaluation)

    return evaluations


# Main

def collect_mixed_evaluations(
    *,
    models: dict[str, str],
    selected_scenarios: list,
    criteria: list[str],
    collection_cfg: dict,
    evaluations_path: str,
    verbose: bool = False,
) -> list[dict]:
    """Run the full 3-phase mixed collection pipeline.

    Automatically detects local (hf_local:) vs OpenRouter models and routes
    accordingly. Local models are batched through vLLM for efficiency.

    Returns the list of all evaluation records produced.
    """
    model_nicks = list(models.keys())
    criteria_text = "\n".join(criteria)
    allow_ties = bool(collection_cfg.get("allow_ties", True))
    max_tokens = int(collection_cfg.get("max_tokens", 4096))
    sampler_mode = (collection_cfg.get("sampler_mode", "random_judge_group")).strip().lower()
    all_to_all = sampler_mode == "all_to_all"
    settings = _openrouter_settings(collection_cfg)
    checkpoint_value = collection_cfg.get("checkpoint_path")
    if checkpoint_value:
        checkpoint_path = Path(checkpoint_value).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(evaluations_path).parent / checkpoint_path
    else:
        checkpoint_path = CollectionCheckpoint.default_path(evaluations_path)
    checkpoint = CollectionCheckpoint(checkpoint_path)

    context = {
        "models": models,
        "selected_scenarios": selected_scenarios,
        "criteria": criteria,
        "collection": {
            "allow_ties": allow_ties,
            "group_size": int(collection_cfg.get("group_size", 4)),
            "groups": int(collection_cfg.get("groups", 1)),
            "sampler_mode": sampler_mode,
            "sampler_seed": collection_cfg.get("sampler_seed"),
            "max_tokens": max_tokens,
            "openrouter": settings,
        },
    }

    # Save the sampled assignments in the retry checkpoint so a partially
    # completed collection resumes the same API tasks.
    if checkpoint.has_manifest():
        eval_assignments = checkpoint.initialize_or_resume(context=context)
    else:
        if all_to_all:
            generated_assignments = _build_eval_assignments_all_to_all(selected_scenarios, models)
        else:
            generated_assignments = _build_eval_assignments_sampled(selected_scenarios, models, collection_cfg)
        eval_assignments = checkpoint.initialize_or_resume(
            context=context,
            assignments=generated_assignments,
        )

    if checkpoint.has_finalized_output():
        records = checkpoint.load_finalized_output(evaluations_path)
        print(f"Mixed collection already finalized. Loaded {len(records)} evaluations from {evaluations_path}")
        return records
    checkpoint.assert_output_is_safe(evaluations_path)

    # Fail before any local model download if the OpenRouter key is absent.
    configured_openrouter_models = {
        nick: model_path
        for nick, model_path in models.items()
        if not is_hf_local_model(model_path)
    }
    if configured_openrouter_models:
        require_openrouter_api_key()

    # Group models
    local_base_models, local_tokenizers, openrouter_models = group_models_for_vllm(models)
    has_local = bool(local_base_models)

    if verbose:
        print(f"Mixed collection: mode={sampler_mode}")
        print(f"  OpenRouter models: {list(openrouter_models.keys())}")
        print(f"  Local base models: {list(local_base_models.keys())}")

    if verbose:
        print(f"  Total assignments: {len(eval_assignments)}")
        print(f"  Checkpoint: {checkpoint.root}")

    # Phase 1: Evaluee Responses
    print("Phase 1: Generate evaluee responses")
    eval_responses: dict = defaultdict(dict)

    _phase1_openrouter(
        eval_assignments,
        openrouter_models,
        eval_responses,
        max_tokens,
        checkpoint,
        settings,
        verbose,
    )
    if has_local:
        _phase1_vllm(eval_assignments, local_base_models, local_tokenizers, eval_responses, max_tokens, verbose)

    # Phase 2: Judge Reflections
    print("Phase 2: Generate judge reflections")
    if all_to_all:
        # Per-judge reflections: judge_reflections[scenario][judge_nick][eval_nick]
        judge_reflections: dict = defaultdict(lambda: defaultdict(dict))
        _phase2_openrouter_all_to_all(
            eval_assignments, openrouter_models, eval_responses,
            judge_reflections, criteria_text, max_tokens,
            checkpoint, settings, verbose,
        )
        if has_local:
            _phase2_vllm_all_to_all(
                eval_assignments, local_base_models, local_tokenizers,
                eval_responses, judge_reflections, criteria_text, max_tokens, verbose,
            )
    else:
        # Shared reflections: judge_reflections[scenario][eval_nick]
        judge_reflections: dict = defaultdict(dict)
        _phase2_openrouter_default(
            eval_assignments, openrouter_models, eval_responses,
            judge_reflections, criteria_text, max_tokens,
            checkpoint, settings, verbose,
        )
        if has_local:
            _phase2_vllm_default(
                eval_assignments, local_base_models, local_tokenizers,
                eval_responses, judge_reflections, criteria_text, max_tokens, verbose,
            )

    # Phase 3: Pairwise Comparisons
    print("Phase 3: Generate pairwise comparisons")
    comparison_tasks = _build_comparison_tasks(eval_assignments)

    if verbose:
        print(f"  Total comparison tasks: {len(comparison_tasks)}")

    all_evaluations = []

    # OpenRouter comparisons
    or_evals = _phase3_openrouter(
        eval_assignments, comparison_tasks, openrouter_models,
        model_nicks, eval_responses, judge_reflections,
        criteria_text, allow_ties, max_tokens, all_to_all,
        checkpoint, settings, verbose,
    )
    all_evaluations.extend(or_evals)

    # vLLM comparisons
    if has_local:
        vllm_evals = _phase3_vllm(
            eval_assignments, comparison_tasks, local_base_models,
            local_tokenizers, model_nicks, eval_responses,
            judge_reflections, criteria_text, allow_ties, max_tokens,
            all_to_all, verbose,
        )
        all_evaluations.extend(vllm_evals)

    if len(all_evaluations) != len(comparison_tasks):
        raise StrictCollectionError(
            "Strict collection produced an incomplete comparison set: "
            f"expected {len(comparison_tasks)}, got {len(all_evaluations)}. "
            f"Checkpoint preserved at {checkpoint.root}; training was not started."
        )

    checkpoint.finalize(evaluations_path, all_evaluations)
    print(f"Mixed collection complete. {len(all_evaluations)} evaluations saved to {evaluations_path}")
    return all_evaluations
