"""Parsing helpers for model references in EigenBench run specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping


@dataclass(frozen=True)
class HFLocalModelRef:
    """Normalized description of a local Hugging Face model or LoRA."""

    repo_id: str
    kind: str | None = None
    subfolder: str | None = None
    revision: str | None = None
    base_model_id: str | None = None
    base_revision: str | None = None

    @property
    def expects_lora(self) -> bool:
        """Whether the spec explicitly describes an adapter."""

        if self.kind is not None:
            return self.kind == "lora"
        return self.subfolder is not None or self.base_model_id is not None


@dataclass(frozen=True)
class OpenRouterModelRef:
    """Normalized OpenRouter model plus instance-level reasoning behavior."""

    model_id: str
    reasoning: dict[str, Any] | None = None
    omit_parameters: frozenset[str] = frozenset()

    @property
    def extra_body(self) -> dict[str, Any] | None:
        """Return non-standard Chat Completions fields for OpenRouter.

        Reasoning-configured instances require an endpoint that honors every
        supplied parameter.  Otherwise OpenRouter may route to an endpoint
        that silently ignores an unsupported parameter, which would collapse
        an instant/reasoning experimental condition.
        """

        if self.reasoning is None:
            return None
        return {
            "reasoning": dict(self.reasoning),
            "provider": {"require_parameters": True},
        }


def is_hf_local_model(model_ref: object) -> bool:
    """Return whether a run-spec model reference selects local HF inference."""

    if isinstance(model_ref, str):
        return model_ref.startswith("hf_local:")
    return isinstance(model_ref, Mapping) and model_ref.get("provider") == "hf_local"


def _optional_nonempty_string(config: Mapping[str, Any], field: str) -> str | None:
    value = config.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Local model field {field!r} must be a non-empty string")
    return value.strip()


def _validate_repo_id(repo_id: str, field: str = "repo_id") -> str:
    if repo_id.count("/") != 1 or any(part in {"", ".", ".."} for part in repo_id.split("/")):
        raise ValueError(f"Local model field {field!r} must have Hugging Face form 'owner/repo'")
    return repo_id


def _validate_subfolder(subfolder: str | None) -> str | None:
    if subfolder is None:
        return None
    path = PurePosixPath(subfolder)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Local model subfolder must be a safe relative repository path")
    return str(path)


_REASONING_EFFORTS = {
    "max",
    "xhigh",
    "high",
    "medium",
    "low",
    "minimal",
    "none",
}


def _normalize_reasoning(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("OpenRouter reasoning must be a mapping")

    allowed_fields = {"effort", "max_tokens", "exclude", "enabled"}
    unknown_fields = sorted(set(value) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"Unknown OpenRouter reasoning fields: {unknown_fields}")

    reasoning = dict(value)
    effort = reasoning.get("effort")
    if effort is not None:
        if not isinstance(effort, str) or effort not in _REASONING_EFFORTS:
            raise ValueError(
                "OpenRouter reasoning.effort must be one of "
                f"{sorted(_REASONING_EFFORTS)}"
            )
    max_tokens = reasoning.get("max_tokens")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
    ):
        raise ValueError("OpenRouter reasoning.max_tokens must be a positive integer")
    if effort is not None and max_tokens is not None:
        raise ValueError(
            "OpenRouter reasoning cannot set effort and max_tokens together"
        )
    for field in ("exclude", "enabled"):
        if field in reasoning and not isinstance(reasoning[field], bool):
            raise ValueError(f"OpenRouter reasoning.{field} must be a boolean")
    if reasoning.get("enabled") is False and (
        effort is not None or max_tokens is not None
    ):
        raise ValueError(
            "Disabled OpenRouter reasoning cannot also set effort or max_tokens"
        )
    return reasoning


def parse_openrouter_model(model_ref: object) -> OpenRouterModelRef:
    """Normalize a legacy model-ID string or structured OpenRouter reference.

    Structured references bind request behavior to a population member, so the
    same reasoning mode follows that member when it acts as either evaluee or
    judge::

        {
            "provider": "openrouter",
            "model_id": "openai/gpt-5.6-sol",
            "reasoning": {"effort": "high", "exclude": True},
        }
    """

    if isinstance(model_ref, str):
        model_id = model_ref.strip()
        if not model_id or model_id.startswith("hf_local:"):
            raise ValueError("Model reference is not an OpenRouter model ID")
        return OpenRouterModelRef(model_id=model_id)

    if not isinstance(model_ref, Mapping) or model_ref.get("provider") != "openrouter":
        raise ValueError("OpenRouter model mapping must set provider='openrouter'")

    allowed_fields = {"provider", "model_id", "reasoning", "omit_parameters"}
    unknown_fields = sorted(set(model_ref) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"Unknown OpenRouter model fields: {unknown_fields}")

    model_id = model_ref.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("OpenRouter model mapping must set model_id")
    omit_parameters = model_ref.get("omit_parameters", [])
    if not isinstance(omit_parameters, (list, tuple, set, frozenset)) or any(
        value != "temperature" for value in omit_parameters
    ):
        raise ValueError(
            "OpenRouter omit_parameters currently supports only 'temperature'"
        )
    return OpenRouterModelRef(
        model_id=model_id.strip(),
        reasoning=_normalize_reasoning(model_ref.get("reasoning")),
        omit_parameters=frozenset(omit_parameters),
    )


def parse_hf_local_model(model_ref: object) -> HFLocalModelRef:
    """Normalize legacy string or revision-safe mapping syntax.

    Legacy syntax remains supported::

        hf_local:owner/repo[/adapter/subfolder]

    Revision-safe syntax is a mapping with ``provider='hf_local'`` and an
    explicit ``repo_id``. ``base_model_id`` overrides stale LoRA metadata.
    """

    if isinstance(model_ref, str):
        if not model_ref.startswith("hf_local:"):
            raise ValueError("Model reference is not an hf_local model")
        hf_path = model_ref.removeprefix("hf_local:").strip().strip("/")
        parts = hf_path.split("/")
        if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(
                "Legacy local model reference must have form "
                "'hf_local:owner/repo[/subfolder]'"
            )
        return HFLocalModelRef(
            repo_id=_validate_repo_id("/".join(parts[:2])),
            subfolder=_validate_subfolder("/".join(parts[2:]) or None),
        )

    if not isinstance(model_ref, Mapping) or model_ref.get("provider") != "hf_local":
        raise ValueError("Local model mapping must set provider='hf_local'")

    allowed_fields = {
        "provider",
        "kind",
        "repo_id",
        "subfolder",
        "revision",
        "base_model_id",
        "base_revision",
    }
    unknown_fields = sorted(set(model_ref) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"Unknown local model fields: {unknown_fields}")

    repo_id = _optional_nonempty_string(model_ref, "repo_id")
    if repo_id is None:
        raise ValueError("Local model mapping must set repo_id")
    repo_id = _validate_repo_id(repo_id)
    kind = _optional_nonempty_string(model_ref, "kind")
    if kind not in {None, "base", "lora"}:
        raise ValueError("Local model kind must be 'base' or 'lora'")
    subfolder = _validate_subfolder(_optional_nonempty_string(model_ref, "subfolder"))
    revision = _optional_nonempty_string(model_ref, "revision")
    base_model_id = _optional_nonempty_string(model_ref, "base_model_id")
    if base_model_id is not None:
        base_model_id = _validate_repo_id(base_model_id, "base_model_id")
    base_revision = _optional_nonempty_string(model_ref, "base_revision")
    if base_revision is not None and base_model_id is None:
        raise ValueError("Local model base_revision requires base_model_id")
    if kind == "base" and any(
        value is not None for value in (subfolder, base_model_id, base_revision)
    ):
        raise ValueError(
            "Local base model references cannot set subfolder, base_model_id, or base_revision"
        )

    return HFLocalModelRef(
        repo_id=repo_id,
        kind=kind,
        subfolder=subfolder,
        revision=revision,
        base_model_id=base_model_id,
        base_revision=base_revision,
    )


__all__ = [
    "HFLocalModelRef",
    "OpenRouterModelRef",
    "is_hf_local_model",
    "parse_hf_local_model",
    "parse_openrouter_model",
]
