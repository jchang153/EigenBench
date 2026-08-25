"""Strict, resumable-safe OpenRouter client helpers.

OpenRouter can report failures either as non-2xx HTTP responses or inside an
HTTP-200 Chat Completions response with ``finish_reason="error"``.  This
module normalizes both forms, retries only transient failures, and never
returns an error message as if it were model-generated text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import os
import random
import threading
import time
from typing import Any, Callable

import openai
from dotenv import load_dotenv

from pipeline.model_refs import parse_openrouter_model


DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_BACKOFF_BASE_SECONDS = 2.0
DEFAULT_BACKOFF_CAP_SECONDS = 60.0

# Canonical error_type values from OpenRouter's error documentation.
RETRYABLE_ERROR_TYPES = frozenset(
    {
        "rate_limit_exceeded",
        "provider_overloaded",
        "provider_unavailable",
        "server",
        "timeout",
        "unmapped",
        "connection_error",
        "empty_response",
        "invalid_response",
    }
)

FATAL_ERROR_TYPES = frozenset(
    {
        "authentication",
        "payment_required",
        "permission_denied",
        "invalid_request",
        "invalid_prompt",
        "not_found",
        "precondition_failed",
        "payload_too_large",
        "unprocessable",
        "context_length_exceeded",
        "max_tokens_exceeded",
        "token_limit_exceeded",
        "string_too_long",
        "content_policy_violation",
        "refusal",
    }
)


@dataclass(frozen=True)
class OpenRouterErrorDetails:
    """Serializable details for one failed OpenRouter attempt."""

    model: str
    attempt: int
    max_attempts: int
    error_type: str
    message: str
    retryable: bool
    status_code: int | None = None
    request_id: str | None = None
    retry_after_seconds: float | None = None
    # The rejected completion, for invalid_response only. A validation failure
    # is often valid data in an unexpected notation, and the error message alone
    # cannot distinguish that from a genuine refusal.
    content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenRouterCallError(RuntimeError):
    """Raised for fatal failures or after transient retries are exhausted."""

    def __init__(
        self,
        details: OpenRouterErrorDetails,
        *,
        exhausted: bool = False,
        attempts: list[OpenRouterErrorDetails] | None = None,
    ):
        self.details = details
        self.exhausted = exhausted
        self.attempts = list(attempts or [details])
        disposition = "retry attempts exhausted" if exhausted else "fatal failure"
        super().__init__(
            f"OpenRouter {disposition} for {details.model}: "
            f"{details.error_type} ({details.status_code or 'no status'}): "
            f"{details.message}"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.details.to_dict()
        payload["exhausted"] = self.exhausted
        payload["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        return payload


@dataclass(frozen=True)
class OpenRouterCompletion:
    """Validated successful completion metadata."""

    content: str
    model: str
    finish_reason: str | None
    request_id: str | None
    provider: str | None


class _AttemptFailure(Exception):
    """Internal exception used to route one failed attempt through retry logic."""

    def __init__(self, details: OpenRouterErrorDetails):
        self.details = details
        super().__init__(details.message)


class _SharedRetryCooldown:
    """Coordinate Retry-After pauses across parallel OpenRouter workers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._resume_at = 0.0

    def extend(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._lock:
            self._resume_at = max(self._resume_at, time.monotonic() + seconds)

    def wait(self) -> None:
        while True:
            with self._lock:
                remaining = self._resume_at - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(remaining)


_SHARED_RETRY_COOLDOWN = _SharedRetryCooldown()


def require_openrouter_api_key() -> str:
    """Load and validate the presence of ``OPENROUTER_API_KEY``."""

    load_dotenv()
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        details = OpenRouterErrorDetails(
            model="<preflight>",
            attempt=0,
            max_attempts=0,
            error_type="authentication",
            message="OPENROUTER_API_KEY is not set",
            retryable=False,
        )
        raise OpenRouterCallError(details)
    return api_key


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            payload = model_dump(mode="python")
        except TypeError:
            payload = model_dump()
        return payload if isinstance(payload, dict) else {}
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        payload = legacy_dict()
        return payload if isinstance(payload, dict) else {}
    return {}


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds)
    except (TypeError, ValueError):
        pass

    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _status_fallback_error_type(status_code: int | None) -> str:
    return {
        400: "invalid_request",
        401: "authentication",
        402: "payment_required",
        403: "permission_denied",
        404: "not_found",
        408: "timeout",
        409: "unmapped",
        412: "precondition_failed",
        413: "payload_too_large",
        422: "unprocessable",
        429: "rate_limit_exceeded",
        502: "provider_unavailable",
        503: "provider_overloaded",
        504: "timeout",
    }.get(status_code, "server" if status_code is not None and status_code >= 500 else "unmapped")


def _extract_error_payload(payload: Any) -> tuple[str | None, str | None, int | None]:
    """Return ``(error_type, message, code)`` from an OpenRouter error object."""

    data = _object_to_dict(payload)
    if not data:
        return None, None, None

    error = data.get("error", data)
    error_data = _object_to_dict(error)
    metadata = _object_to_dict(error_data.get("metadata"))

    error_type = (
        metadata.get("error_type")
        or error_data.get("error_type")
        or data.get("error_type")
    )
    message = error_data.get("message") or data.get("message")
    code = error_data.get("code") or data.get("code")
    try:
        status_code = int(code) if code is not None else None
    except (TypeError, ValueError):
        status_code = None
    return (
        str(error_type) if error_type else None,
        str(message) if message else None,
        status_code,
    )


def _is_retryable(error_type: str, status_code: int | None) -> bool:
    if error_type in FATAL_ERROR_TYPES:
        return False
    if error_type in RETRYABLE_ERROR_TYPES:
        return True
    return status_code in {408, 409, 429} or bool(status_code is not None and status_code >= 500)


def _details_from_sdk_exception(
    exc: Exception,
    *,
    model: str,
    attempt: int,
    max_attempts: int,
) -> OpenRouterErrorDetails:
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    response = getattr(exc, "response", None)
    retry_after = None
    payload: dict[str, Any] = {}

    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = _parse_retry_after(headers.get("Retry-After"))
        try:
            candidate = response.json()
            if isinstance(candidate, dict):
                payload = candidate
        except Exception:
            payload = {}

    error_type, payload_message, payload_status = _extract_error_payload(payload)
    if status_code is None:
        status_code = payload_status

    if isinstance(exc, getattr(openai, "APITimeoutError", ())):
        error_type = error_type or "timeout"
    elif isinstance(exc, getattr(openai, "APIConnectionError", ())):
        error_type = error_type or "connection_error"
    else:
        error_type = error_type or _status_fallback_error_type(status_code)

    message = payload_message or str(exc) or type(exc).__name__
    return OpenRouterErrorDetails(
        model=model,
        attempt=attempt,
        max_attempts=max_attempts,
        error_type=error_type,
        message=message[:2000],
        retryable=_is_retryable(error_type, status_code),
        status_code=status_code,
        request_id=request_id,
        retry_after_seconds=retry_after,
    )


def _details_from_embedded_error(
    error_payload: Any,
    *,
    model: str,
    attempt: int,
    max_attempts: int,
    request_id: str | None,
) -> OpenRouterErrorDetails:
    error_type, message, status_code = _extract_error_payload(error_payload)
    error_type = error_type or _status_fallback_error_type(status_code)
    return OpenRouterErrorDetails(
        model=model,
        attempt=attempt,
        max_attempts=max_attempts,
        error_type=error_type,
        message=(message or "OpenRouter returned finish_reason='error'")[:2000],
        retryable=_is_retryable(error_type, status_code),
        status_code=status_code,
        request_id=request_id,
    )


def _validate_completion(
    response: Any,
    *,
    model: str,
    attempt: int,
    max_attempts: int,
    response_validator: Callable[[str], str | None] | None,
) -> OpenRouterCompletion:
    response_data = _object_to_dict(response)
    request_id = getattr(response, "_request_id", None) or response_data.get("id")
    provider = getattr(response, "provider", None) or response_data.get("provider")
    choices = getattr(response, "choices", None) or response_data.get("choices") or []

    if not choices:
        details = OpenRouterErrorDetails(
            model=model,
            attempt=attempt,
            max_attempts=max_attempts,
            error_type="empty_response",
            message="OpenRouter response contained no choices",
            retryable=True,
            request_id=request_id,
        )
        raise _AttemptFailure(details)

    choice = choices[0]
    choice_data = _object_to_dict(choice)
    finish_reason = getattr(choice, "finish_reason", None) or choice_data.get("finish_reason")
    embedded_error = getattr(choice, "error", None) or choice_data.get("error") or response_data.get("error")
    if finish_reason == "error" or embedded_error:
        details = _details_from_embedded_error(
            embedded_error or {},
            model=model,
            attempt=attempt,
            max_attempts=max_attempts,
            request_id=request_id,
        )
        raise _AttemptFailure(details)

    message_obj = getattr(choice, "message", None) or choice_data.get("message")
    message_data = _object_to_dict(message_obj)
    content = getattr(message_obj, "content", None) if message_obj is not None else None
    if content is None:
        content = message_data.get("content")

    if not isinstance(content, str) or not content.strip():
        details = OpenRouterErrorDetails(
            model=model,
            attempt=attempt,
            max_attempts=max_attempts,
            error_type="empty_response",
            message="OpenRouter response contained no non-empty text content",
            retryable=True,
            request_id=request_id,
        )
        raise _AttemptFailure(details)

    if response_validator is not None:
        validation_error = response_validator(content)
        if validation_error:
            details = OpenRouterErrorDetails(
                model=model,
                attempt=attempt,
                max_attempts=max_attempts,
                error_type="invalid_response",
                message=str(validation_error)[:2000],
                retryable=True,
                request_id=request_id,
                content=content[:4000],
            )
            raise _AttemptFailure(details)

    return OpenRouterCompletion(
        content=content,
        model=model,
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        request_id=str(request_id) if request_id is not None else None,
        provider=str(provider) if provider is not None else None,
    )


def _retry_delay_seconds(
    details: OpenRouterErrorDetails,
    *,
    backoff_base_seconds: float,
    backoff_cap_seconds: float,
) -> float:
    if details.retry_after_seconds is not None:
        # Retry-After is an explicit server instruction, so do not apply the
        # local exponential-backoff cap to it.
        return max(0.0, details.retry_after_seconds)
    exponential_cap = min(
        backoff_cap_seconds,
        backoff_base_seconds * (2 ** max(0, details.attempt - 1)),
    )
    return random.uniform(0.0, max(0.0, exponential_cap))


def get_openrouter_response(
    messages,
    model: object,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    return_full_response: bool = False,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    backoff_cap_seconds: float = DEFAULT_BACKOFF_CAP_SECONDS,
    response_validator: Callable[[str], str | None] | None = None,
    client: Any | None = None,
):
    """Send one strict Chat Completions request through OpenRouter.

    The OpenAI SDK retry layer is disabled so every attempt is visible here and
    can be recorded by the collection checkpoint.  Fatal errors raise
    immediately; retryable errors raise only after ``max_attempts``.
    """

    model_ref = parse_openrouter_model(model)
    model_id = model_ref.model_id

    max_attempts = int(max_attempts)
    timeout_seconds = float(timeout_seconds)
    backoff_base_seconds = float(backoff_base_seconds)
    backoff_cap_seconds = float(backoff_cap_seconds)
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if backoff_base_seconds < 0 or backoff_cap_seconds < 0:
        raise ValueError("backoff settings must be non-negative")

    owns_client = client is None
    if client is None:
        api_key = require_openrouter_api_key()
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=0,
            timeout=timeout_seconds,
            default_headers={"X-OpenRouter-Metadata": "enabled"},
        )

    try:
        attempt_history: list[OpenRouterErrorDetails] = []
        for attempt in range(1, max_attempts + 1):
            _SHARED_RETRY_COOLDOWN.wait()
            try:
                request = {
                    "model": model_id,
                    "max_tokens": max_tokens,
                    "messages": messages,
                }
                if "temperature" not in model_ref.omit_parameters:
                    request["temperature"] = temperature
                if model_ref.extra_body is not None:
                    request["extra_body"] = model_ref.extra_body
                response = client.chat.completions.create(**request)
                completion = _validate_completion(
                    response,
                    model=model_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    response_validator=response_validator,
                )
                return response if return_full_response else completion.content
            except _AttemptFailure as exc:
                details = exc.details
            except Exception as exc:
                details = _details_from_sdk_exception(
                    exc,
                    model=model_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )

            attempt_history.append(details)

            if not details.retryable:
                raise OpenRouterCallError(
                    details,
                    exhausted=False,
                    attempts=attempt_history,
                )
            if attempt >= max_attempts:
                raise OpenRouterCallError(
                    details,
                    exhausted=True,
                    attempts=attempt_history,
                )

            delay = _retry_delay_seconds(
                details,
                backoff_base_seconds=backoff_base_seconds,
                backoff_cap_seconds=backoff_cap_seconds,
            )
            if details.error_type in {"rate_limit_exceeded", "provider_overloaded"}:
                _SHARED_RETRY_COOLDOWN.extend(delay)
            else:
                time.sleep(delay)
    finally:
        if owns_client:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    raise AssertionError("OpenRouter retry loop exited unexpectedly")


__all__ = [
    "DEFAULT_BACKOFF_BASE_SECONDS",
    "DEFAULT_BACKOFF_CAP_SECONDS",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "FATAL_ERROR_TYPES",
    "OpenRouterCallError",
    "OpenRouterCompletion",
    "OpenRouterErrorDetails",
    "RETRYABLE_ERROR_TYPES",
    "get_openrouter_response",
    "require_openrouter_api_key",
]
