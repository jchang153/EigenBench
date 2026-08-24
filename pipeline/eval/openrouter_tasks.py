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


MAX_PARALLEL_API_CALLS = 10


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

            # Once a task has failed, let already-running calls finish and be
            # checkpointed, but stop submitting new calls.
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


# Backward-compatible private names for callers that used the old mixed module.
_OpenRouterTask = OpenRouterTask
_call_openrouter = call_openrouter
_openrouter_settings = openrouter_settings
_run_openrouter_tasks = run_openrouter_tasks


__all__ = [
    "MAX_PARALLEL_API_CALLS",
    "OpenRouterTask",
    "StrictCollectionError",
    "call_openrouter",
    "openrouter_settings",
    "run_openrouter_tasks",
]
