"""Run prepared Inspect tasks and manage single-model phases."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import anyio
from inspect_ai.model import Model


# Inspect serializes task arguments for native retry. Scope the prepared context
# to task construction so execution state never becomes part of those arguments.
_prepared_run: ContextVar[tuple[str, dict | None, dict] | None] = ContextVar(
    "eigenbench_prepared_run", default=None,
)


@contextmanager
def use_run_context(spec: str, models: dict | None, context: dict):
    token = _prepared_run.set((spec, models, context))
    try:
        yield
    finally:
        _prepared_run.reset(token)


def prepared_run_context(spec: str, models: dict | None) -> dict | None:
    prepared = _prepared_run.get()
    if prepared is not None and prepared[0] == spec and prepared[1] is models:
        return prepared[2]
    return None


def _uses_vllm(model) -> bool:
    return str(model).startswith("vllm/")


@contextmanager
def model_phase(context: dict, nick: str, *, reuse_resolver: bool = False):
    """Restrict resolution to one model and release its client at phase exit."""

    from ..context import _model_resolver

    value = context["models"][nick]
    resolve = (context["resolve_model"] if reuse_resolver
               else _model_resolver({nick: value}, memoize=False))
    loaded = None

    def resolve_only_this_model(name: str):
        nonlocal loaded
        if name != nick:
            raise RuntimeError(f"phased task for {nick!r} attempted to load {name!r}")
        model = resolve(name)
        # A shared client may be reused by its owner. A vLLM server must still
        # be released before another model can claim the GPU.
        if _uses_vllm(model) or (not reuse_resolver and not isinstance(value, Model)):
            loaded = model
        return model

    try:
        yield {**context, "resolve_model": resolve_only_this_model}
    finally:
        if loaded is not None:
            close = getattr(loaded.api, "aclose", None)
            if close is not None:
                anyio.run(close)


def eval_phase(task, eval_kwargs: dict, seen_logs: list, *, require_complete: bool = True):
    from inspect_ai import eval as inspect_eval

    logs = inspect_eval(tasks=task, model=None, max_tasks=1, **eval_kwargs)
    seen_logs.extend(logs)
    if len(logs) != 1 or (require_complete and logs[0].status != "success"):
        raise RuntimeError("phased task did not complete; nothing exported")
    log = logs[0]
    if require_complete and (
        len(log.samples or []) != len(task.dataset) or any(s.error for s in log.samples)
    ):
        raise RuntimeError("phased task has incomplete samples; nothing exported")
    return log
