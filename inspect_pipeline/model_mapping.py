"""Map run-spec model references onto Inspect AI model names.

Legacy spec semantics are preserved:
- a bare string is an OpenRouter model id -> ``openrouter/<id>``
- ``hf_local:`` strings / ``provider='hf_local'`` mappings -> the ``vllm/``
  provider; LoRA adapters use Inspect's ``vllm/<base>:<adapter>[@revision]``
  syntax, with subfolder adapters resolved to a local snapshot path exactly
  like the legacy vLLM collector

New escape hatch: an ``inspect:`` prefix passes the rest through verbatim so
any Inspect provider can be used directly (e.g. ``inspect:anthropic/claude-…``,
``inspect:mockllm/model`` for tests).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Mapping

from pipeline.model_refs import HFLocalModelRef, is_hf_local_model, parse_hf_local_model

INSPECT_PREFIX = "inspect:"


@dataclass(frozen=True)
class InspectModelRef:
    """Inspect model name plus provider model_args for one spec entry."""

    name: str
    model_args: dict = field(default_factory=dict)
    is_local: bool = False


def _base_model_from_adapter_config(adapter_config_path: str) -> str:
    with open(adapter_config_path, "r", encoding="utf-8") as handle:
        adapter_cfg = json.load(handle)
    base_model_id = adapter_cfg.get("base_model_name_or_path")
    if not isinstance(base_model_id, str) or not base_model_id.strip():
        raise ValueError(
            f"adapter_config.json at {adapter_config_path} has no base_model_name_or_path"
        )
    return base_model_id.strip()


def _resolve_hf_local(ref: HFLocalModelRef) -> InspectModelRef:
    if not ref.expects_lora:
        model_args: dict = {}
        if ref.revision:
            model_args["revision"] = ref.revision
        return InspectModelRef(
            name=f"vllm/{ref.repo_id}", model_args=model_args, is_local=True
        )

    base_model_id = ref.base_model_id
    if ref.subfolder:
        # Subfolder adapters are not addressable as HF repos by the vllm
        # provider; snapshot the repo and point at the local path (legacy parity).
        from huggingface_hub import snapshot_download

        local_repo = snapshot_download(repo_id=ref.repo_id, revision=ref.revision)
        adapter_ref = os.path.join(local_repo, ref.subfolder)
        if base_model_id is None:
            base_model_id = _base_model_from_adapter_config(
                os.path.join(adapter_ref, "adapter_config.json")
            )
    else:
        if base_model_id is None:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=ref.repo_id,
                filename="adapter_config.json",
                revision=ref.revision,
            )
            base_model_id = _base_model_from_adapter_config(config_path)
        adapter_ref = ref.repo_id + (f"@{ref.revision}" if ref.revision else "")

    model_args = {}
    if ref.base_revision:
        model_args["revision"] = ref.base_revision
    return InspectModelRef(
        name=f"vllm/{base_model_id}:{adapter_ref}",
        model_args=model_args,
        is_local=True,
    )


def to_inspect_model(model_ref: object) -> InspectModelRef:
    if isinstance(model_ref, str) and model_ref.startswith(INSPECT_PREFIX):
        name = model_ref.removeprefix(INSPECT_PREFIX).strip()
        if not name or "/" not in name:
            raise ValueError(
                "inspect: model reference must have form 'inspect:provider/model'; "
                f"got {model_ref!r}"
            )
        return InspectModelRef(name=name)

    if is_hf_local_model(model_ref):
        return _resolve_hf_local(parse_hf_local_model(model_ref))

    if isinstance(model_ref, str) and model_ref.strip():
        return InspectModelRef(name=f"openrouter/{model_ref.strip()}")

    if isinstance(model_ref, Mapping):
        raise ValueError(
            f"Unsupported model mapping in spec (expected provider='hf_local'): {model_ref!r}"
        )
    raise ValueError(f"Unsupported model reference in spec: {model_ref!r}")


def resolve_spec_models(models: Mapping[str, object]) -> dict[str, InspectModelRef]:
    """Resolve the spec ``models`` dict, preserving insertion order (= model index)."""

    if not models:
        raise ValueError("spec models must not be empty")
    return {nick: to_inspect_model(value) for nick, value in models.items()}
