"""Model provider helpers."""

from typing import TYPE_CHECKING

from .openrouter import get_openrouter_response

if TYPE_CHECKING:  
    from .vllm_local import (
        VLLMEngineManager,
        group_models_for_vllm,
        prepare_lora_requests,
    )

_VLLM_EXPORTS = frozenset(
    {
        "VLLMEngineManager",
        "group_models_for_vllm",
        "prepare_lora_requests",
    }
)

__all__ = [
    "get_openrouter_response",
    "group_models_for_vllm",
    "prepare_lora_requests",
    "VLLMEngineManager",
]


def __getattr__(name: str):
    if name in _VLLM_EXPORTS:
        from . import vllm_local

        return getattr(vllm_local, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
