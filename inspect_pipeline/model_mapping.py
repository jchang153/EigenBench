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

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Mapping

from inspect_ai.model import Model, ModelName
from inspect_ai.model._providers.vllm import VLLMAPI

from pipeline.model_refs import HFLocalModelRef, is_hf_local_model, parse_hf_local_model
from pipeline.model_window import DEFAULT_MAX_MODEL_LEN, resolve_max_model_len

INSPECT_PREFIX = "inspect:"

# Preserve the local model context cap and limit KV-cache allocation.
VLLM_SERVER_DEFAULTS = {
    "max_model_len": DEFAULT_MAX_MODEL_LEN,
    "gpu_memory_utilization": 0.9,
    "enforce_eager": True,
}


@dataclass(frozen=True)
class InspectModelRef:
    """Inspect model name plus provider model_args for one spec entry."""

    name: str
    model_args: dict = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return model_execution(self) == "managed_vllm"


def _base_model_from_adapter_config(adapter_config_path: str) -> str:
    with open(adapter_config_path, "r", encoding="utf-8") as handle:
        adapter_cfg = json.load(handle)
    base_model_id = adapter_cfg.get("base_model_name_or_path")
    if not isinstance(base_model_id, str) or not base_model_id.strip():
        raise ValueError(
            f"adapter_config.json at {adapter_config_path} has no base_model_name_or_path"
        )
    return base_model_id.strip()


def vllm_server_args(model_id: str | None = None, revision: str | None = None) -> dict:
    """``vllm serve`` arguments for a local model.

    The native defaults, with max_model_len lowered to the model's own window when shorter,
    the native engine's ``EIGENBENCH_TENSOR_PARALLEL_SIZE``, then any overrides in
    ``EIGENBENCH_VLLM_SERVER_ARGS`` (a JSON object; ``null`` removes a default), e.g.
    ``{"max_model_len": 32768}``.
    """

    args = dict(VLLM_SERVER_DEFAULTS)
    tensor_parallel_size = int(os.environ.get("EIGENBENCH_TENSOR_PARALLEL_SIZE", "1"))
    if tensor_parallel_size < 1:
        raise ValueError("EIGENBENCH_TENSOR_PARALLEL_SIZE must be a positive integer")
    if tensor_parallel_size > 1:
        args["tensor_parallel_size"] = tensor_parallel_size
    overrides = os.environ.get("EIGENBENCH_VLLM_SERVER_ARGS")
    overrides = json.loads(overrides) if overrides else {}
    if not isinstance(overrides, dict):
        raise ValueError("EIGENBENCH_VLLM_SERVER_ARGS must be a JSON object")
    if model_id and "max_model_len" not in overrides:
        args["max_model_len"] = resolve_max_model_len(model_id, revision, args["max_model_len"])
    args.update(overrides)
    return {key: value for key, value in args.items() if value is not None}


@lru_cache(maxsize=None)
def _root_adapter_base(repo_id: str, revision: str | None) -> str | None:
    """The base model a root adapter_config.json names, for a reference of no declared kind.

    Only a confirmed answer is returned (and cached): the base model, or None when the Hub
    reports no such file. A lookup that cannot complete raises, so it is retried on the
    next call rather than remembered as a full model.
    """

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="adapter_config.json", revision=revision)
    except EntryNotFoundError as exc:
        if not isinstance(exc, LocalEntryNotFoundError):
            return None  # a full model
        error = exc
    except Exception as exc:
        error = exc
    else:
        return _base_model_from_adapter_config(config_path)
    raise RuntimeError(
        f"Could not check {repo_id} for a root adapter_config.json ({type(error).__name__}: "
        f"{error}). Retry, or declare kind='base' or kind='lora' for this model in the spec."
    ) from error


def _resolve_hf_local(ref: HFLocalModelRef) -> InspectModelRef:
    if ref.kind is None and not ref.expects_lora:
        # As the legacy collector did, a repo holding an adapter at its root is a LoRA.
        base_model_id = _root_adapter_base(ref.repo_id, ref.revision)
        if base_model_id is not None:
            ref = replace(ref, kind="lora", base_model_id=base_model_id)
    if not ref.expects_lora:
        model_args: dict = vllm_server_args(ref.repo_id, ref.revision)
        if ref.revision:
            model_args["revision"] = ref.revision
        return InspectModelRef(
            name=f"vllm/{ref.repo_id}", model_args=model_args
        )

    base_model_id = ref.base_model_id
    if ref.subfolder:
        # Subfolder adapters are not addressable as HF repos by the vllm
        # provider; snapshot the repo and point at the local path (legacy parity).
        from huggingface_hub import snapshot_download

        local_repo = snapshot_download(
            repo_id=ref.repo_id,
            revision=ref.revision,
            allow_patterns=[f"{ref.subfolder}/*"],
        )
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

    model_args = vllm_server_args(base_model_id, ref.base_revision)
    if ref.base_revision:
        model_args["revision"] = ref.base_revision
    return InspectModelRef(
        name=f"vllm/{base_model_id}:{adapter_ref}",
        model_args=model_args,
    )


def with_effective_vllm_revision(ref: InspectModelRef) -> InspectModelRef:
    """Pin the revision in the arguments Inspect uses to memoize model instances."""
    if not ref.name.startswith("vllm/") or "revision" in ref.model_args:
        return ref

    from inspect_ai._util.local_server import merge_env_server_args
    from inspect_ai.model._providers.vllm import VLLM_DEFAULT_SERVER_ARGS

    defaults = merge_env_server_args(VLLM_DEFAULT_SERVER_ARGS, {}, logging.getLogger(__name__))
    revision = defaults.get("revision")
    if revision is None:
        return ref
    return replace(ref, model_args={**ref.model_args, "revision": revision})


def _vllm_endpoint(model_args: dict, base_url: str | None = None) -> str | None:
    port = model_args.get("port")
    return base_url or model_args.get("base_url") or (
        f"http://localhost:{port}/v1" if port else os.environ.get("VLLM_BASE_URL")
    )


def check_vllm_endpoint(model: Model) -> None:
    """A live server shared by base model must match this run's requested endpoint."""
    if not isinstance(model.api, VLLMAPI):
        return
    server = model.api._server
    if server.base_url is None:
        return
    requested = _vllm_endpoint(model.model_args, model.explicit_base_url)
    active = None if server.process is not None else str(server.base_url)
    if (requested or "").rstrip("/") != (active or "").rstrip("/"):
        raise RuntimeError(
            f"vLLM already has a different server for {model.api.base_model}. "
            "Start a fresh Inspect process to change endpoints."
        )


def model_execution(value: object) -> str:
    """Identify managed vLLM, OpenRouter, or another provider without loading weights."""
    if isinstance(value, Model):
        provider = ModelName(value).api
        args, base_url = value.model_args, value.explicit_base_url
    elif is_hf_local_model(value):
        # A cost estimate needs only the provider, even for an adapter subfolder.
        provider, args, base_url = "vllm", vllm_server_args(), None
    else:
        ref = value if isinstance(value, InspectModelRef) else to_inspect_model(value)
        provider, args, base_url = ModelName(ref.name).api, ref.model_args, None
    if provider == "vllm" and not _vllm_endpoint(args, base_url):
        return "managed_vllm"
    return "openrouter" if provider == "openrouter" else "other"


def check_distinct_local_servers(models: Mapping[str, object]) -> None:
    """Inspect shares a vLLM server by base model, including across endpoints."""

    by_base: dict[str, dict] = {}
    for nick, value in models.items():
        if isinstance(value, Model):
            if not isinstance(value.api, VLLMAPI):
                continue
            base = value.api.base_model
            server_args = value.api.server_args
            endpoint = _vllm_endpoint(value.model_args, value.explicit_base_url)
        else:
            ref = with_effective_vllm_revision(to_inspect_model(value))
            if not ref.name.startswith("vllm/"):
                continue
            base = ref.name.removeprefix("vllm/").split(":", 1)[0]
            server_args = ref.model_args
            endpoint = _vllm_endpoint(server_args)
        identity = endpoint, server_args.get("revision")
        by_base.setdefault(base, {}).setdefault(identity, []).append(nick)
    clashes = []
    for base, revisions in by_base.items():
        if len(revisions) > 1:
            members = [f"{', '.join(nicks)} at {rev or 'the default revision'}"
                       for (_, rev), nicks in revisions.items()]
            clashes.append(f"{base} ({'; '.join(members)})")
    if clashes:
        raise ValueError(
            "These local models share a base model at different revisions or endpoints, which one vLLM "
            f"server cannot serve together: {'; '.join(clashes)}. "
            "Set collection.inspect.phased=true to serve them one at a time."
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


def model_identity(value: object) -> dict:
    """Requested identity, stable across provider startup and spec reloads."""

    endpoint = None
    if isinstance(value, Model):
        args = value.model_args
        if isinstance(value.api, VLLMAPI):
            endpoint = _vllm_endpoint(args, value.explicit_base_url)
            revision = value.api.server_args.get("revision")
            if revision is not None:
                args = {**args, "revision": revision}
        identity = {
            "name": str(value),
            # Runtime callbacks and output iterators are not stable provider settings.
            "args": json.loads(json.dumps(args, default=lambda _: None)),
            "base_url": value.explicit_base_url,
            "config": value.config.model_dump(exclude_none=True),
        }
    else:
        ref = value if isinstance(value, InspectModelRef) else to_inspect_model(value)
        ref = with_effective_vllm_revision(ref)
        if ref.name.startswith("vllm/"):
            endpoint = _vllm_endpoint(ref.model_args)
        identity = {"name": ref.name, "args": ref.model_args}
    if endpoint:
        # An endpoint can contain credentials; retain only its fingerprint in run metadata.
        identity.pop("base_url", None)
        identity["args"] = {k: v for k, v in identity["args"].items() if k != "base_url"}
        identity["vllm_endpoint_sha256"] = hashlib.sha256(endpoint.rstrip("/").encode()).hexdigest()
    return identity


def resolve_spec_models(models: Mapping[str, object]) -> dict[str, InspectModelRef]:
    """Resolve the spec ``models`` dict, preserving insertion order (= model index)."""

    if not models:
        raise ValueError("spec models must not be empty")
    return {nick: to_inspect_model(value) for nick, value in models.items()}
