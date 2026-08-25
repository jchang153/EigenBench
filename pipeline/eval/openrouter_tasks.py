"""Bounded, checkpoint-aware scheduling for OpenRouter collection tasks."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable

from pipeline.providers.openrouter import (
    OpenRouterCallError,
    get_openrouter_response,
)

from .checkpoint import CollectionCheckpoint
from .failures import FailureBudget


MAX_PARALLEL_API_CALLS = 36


@dataclass(frozen=True)
class OpenRouterTask:
    identity: dict
    call: Callable[[], str]


class StrictCollectionError(RuntimeError):
    """Raised after checkpointing an exhausted or fatal collection task."""


def openrouter_settings(collection_cfg: dict) -> dict:
    cfg = collection_cfg.get("openrouter", {}) or {}
    settings = {
        "max_attempts": int(cfg.get("max_attempts", 4)),
        "timeout_seconds": float(cfg.get("timeout_seconds", 300.0)),
        "backoff_base_seconds": float(cfg.get("backoff_base_seconds", 2.0)),
        "backoff_cap_seconds": float(cfg.get("backoff_cap_seconds", 60.0)),
        "max_workers": int(cfg.get("max_workers", MAX_PARALLEL_API_CALLS)),
    }
    if settings["max_attempts"] <= 0:
        raise ValueError("collection.openrouter.max_attempts must be positive")
    if settings["timeout_seconds"] <= 0:
        raise ValueError("collection.openrouter.timeout_seconds must be positive")
    if settings["backoff_base_seconds"] < 0 or settings["backoff_cap_seconds"] < 0:
        raise ValueError("collection.openrouter backoff values must be non-negative")
    if settings["max_workers"] <= 0:
        raise ValueError("collection.openrouter.max_workers must be positive")
    return settings


def call_openrouter(
    model_path: str,
    messages,
    max_tokens: int,
    settings: dict,
    *,
    temperature: float = 1.0,
    response_validator: Callable[[str], str | None] | None = None,
) -> str:
    return get_openrouter_response(
        messages,
        model=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        max_attempts=settings["max_attempts"],
        timeout_seconds=settings["timeout_seconds"],
        backoff_base_seconds=settings["backoff_base_seconds"],
        backoff_cap_seconds=settings["backoff_cap_seconds"],
        response_validator=response_validator,
    )


def run_openrouter_tasks(
    tasks: list[OpenRouterTask],
    *,
    checkpoint: CollectionCheckpoint,
    max_workers: int,
    failure_budget: FailureBudget | None = None,
) -> list[str | None]:
    """Run bounded parallel tasks, checkpointing every success and failure.

    The returned list is positionally aligned with ``tasks``. A slot is ``None``
    only where a task failed permanently and ``failure_budget`` allowed the run
    to continue, so callers that pass no budget still get an all-string list.
    """

    if max_workers <= 0:
        raise ValueError("collection.openrouter.max_workers must be positive")

    budget = failure_budget if failure_budget is not None else FailureBudget()

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
        return results

    executor = ThreadPoolExecutor(max_workers=min(max_workers, len(pending_indices)))
    in_flight = {}
    next_pending = 0
    tolerated: set[int] = set()
    fatal: list[tuple[dict, dict]] = []

    def submit_available() -> None:
        nonlocal next_pending
        while not fatal and len(in_flight) < max_workers and next_pending < len(pending_indices):
            index = pending_indices[next_pending]
            next_pending += 1
            future = executor.submit(tasks[index].call)
            in_flight[future] = index

    def note_failure(index: int, task: OpenRouterTask, error: dict) -> None:
        checkpoint.save_failed(task.identity, error)
        if budget.record(task.identity, error):
            tolerated.add(index)
        else:
            fatal.append((task.identity, error))

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
                    note_failure(index, task, exc.to_dict())
                except Exception as exc:
                    note_failure(index, task, {
                        "error_type": "client_bug",
                        "message": f"{type(exc).__name__}: {exc}",
                        "retryable": False,
                        "exhausted": False,
                    })
                else:
                    results[index] = content
                    checkpoint.save_completed(task.identity, {"content": content})

            # Once the budget is spent, let already-running calls finish and be
            # checkpointed, but stop submitting new calls.
            submit_available()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if fatal:
        identity, error = fatal[0]
        spent = "" if budget.limit == 0 else (
            f" The failure budget is spent ({budget.summary()}); raise "
            "collection.max_failed_tasks to tolerate more, or fix the cause."
        )
        raise StrictCollectionError(
            "Strict collection paused after a failed OpenRouter task. "
            f"task={identity}, error={error}.{spent} Re-run with the same spec to "
            "resume from the checkpoint after resolving the failure."
        )

    unaccounted = {index for index, result in enumerate(results) if result is None} - tolerated
    if unaccounted:
        raise RuntimeError("OpenRouter task scheduler exited with incomplete results")
    return results


# Backward-compatible private names for callers that used the old mixed module.
_OpenRouterTask = OpenRouterTask
_call_openrouter = call_openrouter
_openrouter_settings = openrouter_settings
_run_openrouter_tasks = run_openrouter_tasks


__all__ = [
    "MAX_PARALLEL_API_CALLS",
    "FailureBudget",
    "OpenRouterTask",
    "StrictCollectionError",
    "call_openrouter",
    "openrouter_settings",
    "run_openrouter_tasks",
]
