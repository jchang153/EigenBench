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


__all__ = ["HFLocalModelRef", "is_hf_local_model", "parse_hf_local_model"]
