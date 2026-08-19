"""Model provider helpers with lazy optional-dependency imports."""

__all__ = [
    "get_openrouter_response",
    "group_models_for_vllm",
    "prepare_lora_requests",
    "VLLMEngineManager",
]


def __getattr__(name):
    if name == "get_openrouter_response":
        from .openrouter import get_openrouter_response

        return get_openrouter_response
    if name in {"group_models_for_vllm", "prepare_lora_requests", "VLLMEngineManager"}:
        from .vllm_local import VLLMEngineManager, group_models_for_vllm, prepare_lora_requests

        return {
            "group_models_for_vllm": group_models_for_vllm,
            "prepare_lora_requests": prepare_lora_requests,
            "VLLMEngineManager": VLLMEngineManager,
        }[name]
    raise AttributeError(name)
