"""Model calls, output validation, and response reuse for Inspect tasks."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Sequence

from inspect_ai.model import (
    CachePolicy,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
)
from inspect_ai.solver import Generate, Solver, TaskState

from eigenbench.protocol import RESPONSE_SYSTEM_MESSAGE

STORE_RESPONSE = "eigenbench:response"


# Truncated and filtered completions remain retryable.
_RETRYABLE_STOP_REASONS = {"max_tokens", "model_length", "content_filter"}


# Supported collection.failure_policy values.
FAILURE_POLICIES = ("strict", "omit_invalid_judgments")


class InvalidJudgmentError(RuntimeError):
    """A judge's reflection or rating was still invalid after every attempt.

    collection.failure_policy="omit_invalid_judgments" omits these samples, as the native
    engine does. Any other sample error, such as a failed response, an authentication
    error or a bug, still stops the export.
    """


class ResponsePool:
    """Share one generation per protocol key across concurrent samples."""

    def __init__(self, seed: dict[tuple, str] | None = None) -> None:
        self._done: dict[tuple, str] = dict(seed or {})
        self._pending: dict[tuple, asyncio.Future[str]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def seeded(self) -> dict[tuple, str]:
        return dict(self._done)

    async def get(
        self, key: tuple, factory: Callable[[], Awaitable[str]]
    ) -> str:
        if key in self._done:
            return self._done[key]

        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            # Futures are loop-bound; a new loop means a new eval run.
            self._loop, self._pending = loop, {}

        pending = self._pending.get(key)
        if pending is not None:
            return await pending

        future: asyncio.Future[str] = loop.create_future()
        self._pending[key] = future
        try:
            value = await factory()
        except BaseException as exc:
            self._pending.pop(key, None)
            if not future.done():
                future.set_exception(exc)
            # Nobody may be awaiting it; avoid "never retrieved" warnings.
            future.exception()
            raise
        self._done[key] = value
        self._pending.pop(key, None)
        if not future.done():
            future.set_result(value)
        return value


def _cache_for_attempt(
    cache_enabled: bool, attempt: int, model_identity: str, cache_scope: str | None = None
) -> bool | CachePolicy:
    if not cache_enabled:
        return False
    # Never-expiring, attempt-scoped: reruns reuse valid outputs (checkpoint
    # semantics) while validation retries get a fresh key.
    scopes = {
        "attempt": str(attempt),
        "eigenbench_cache_version": "2",
        "model_identity": model_identity,
    }
    # Protocol units that must stay independent draws even when their prompts
    # are identical (e.g. one pairwise response per sampled group) add a scope.
    if cache_scope is not None:
        scopes["eigenbench_scope"] = str(cache_scope)
    return CachePolicy(expiry=None, scopes=scopes)


async def generate_validated(
    model: Model,
    messages: Sequence,
    *,
    config: GenerateConfig,
    max_attempts: int,
    cache_enabled: bool,
    validator: Callable[[str], str | None] | None,
    identity: str,
    exhausted: type[RuntimeError] = RuntimeError,
    cache_scope: str | None = None,
) -> ModelOutput:
    """Legacy retry contract: empty, truncated, filtered, or validator-rejected
    completions are retried up to ``max_attempts``, then ``exhausted`` is raised."""

    model_identity = getattr(model, "_eigenbench_cache_identity", None)
    if cache_enabled and model_identity is None:
        raise ValueError("Cached generation requires a model from the EigenBench resolver")
    last_error = "no attempts made"
    for attempt in range(1, max_attempts + 1):
        output = await model.generate(
            input=list(messages),
            config=config,
            cache=_cache_for_attempt(cache_enabled, attempt, model_identity or "", cache_scope),
        )
        content = output.completion
        if not isinstance(content, str) or not content.strip():
            last_error = "empty completion"
        elif output.stop_reason in _RETRYABLE_STOP_REASONS:
            last_error = f"completion stopped with stop_reason={output.stop_reason!r}"
        else:
            error = validator(content) if validator else None
            if error is None:
                return output
            last_error = error
    raise exhausted(
        f"generation failed validation after {max_attempts} attempts "
        f"({identity}): {last_error}"
    )


def phase_config(phase_cfg: dict, model: str | None = None) -> GenerateConfig:
    """Generation settings for a phase, with any per-model override applied.

    Context windows differ by an order of magnitude across a panel, so one
    budget either truncates the verbose models or overruns the small ones.
    """

    cfg = dict(phase_cfg)
    override = (phase_cfg.get("per_model") or {}).get(model or "")
    if override:
        cfg.update(override)
    return GenerateConfig(
        max_tokens=int(cfg["max_tokens"]),
        temperature=float(cfg["temperature"]),
    )


def response_only_solver(
    *,
    resolve_model: Callable[[str], Model],
    generation_cfg: dict,
    max_attempts: int,
    cache_enabled: bool,
    cache_scope: Callable[[dict], str] | None = None,
) -> Solver:
    """Generate one evaluee response per sample, and nothing else."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        md = state.metadata
        s_idx = int(md["scenario_index"])
        eval_nick = md["eval_nick"]
        config = phase_config(generation_cfg, eval_nick)
        messages = [
            ChatMessageSystem(content=RESPONSE_SYSTEM_MESSAGE),
            ChatMessageUser(content=md["scenario"]),
        ]
        output = await generate_validated(
            resolve_model(eval_nick),
            messages,
            config=config,
            max_attempts=max_attempts,
            cache_enabled=cache_enabled,
            validator=None,
            identity=f"response scenario_index={s_idx} evaluee={eval_nick}",
            cache_scope=cache_scope(md) if cache_scope else None,
        )
        state.messages = messages + [output.message]
        state.output = output
        state.store.set(STORE_RESPONSE, output.completion)
        return state

    return solve
