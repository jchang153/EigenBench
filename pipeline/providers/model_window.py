"""The context window a local model was trained for, shared by the native and Inspect engines."""

from __future__ import annotations

from functools import lru_cache

# Limit KV-cache allocation for models with large context windows.
DEFAULT_MAX_MODEL_LEN = 8192


@lru_cache(maxsize=None)
def resolve_max_model_len(
    model_id: str, revision: str | None = None, requested: int = DEFAULT_MAX_MODEL_LEN
) -> int:
    """Cap the requested context at the model's window, including multimodal text configs.

    Keep ``requested`` if the config is unavailable or has no usable limit.
    """

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id, revision=revision)
    except Exception as exc:
        print(f"  Could not read the config of {model_id} ({exc}); using max_model_len={requested}")
        return requested

    configs = [config]
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            text_config = get_text_config()
        except Exception:
            text_config = None
        if text_config is not None and text_config is not config:
            configs.append(text_config)

    limits = []
    for candidate in configs:
        for attr in (
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "seq_length",
            "model_max_length",
            "max_target_positions",
            "max_sequence_length",
            "max_seq_length",
            "seq_len",
        ):
            value = getattr(candidate, attr, None)
            # OLMo-2 stores max_position_embeddings as a float.
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                limits.append(int(value))
    window = min(limits, default=requested)
    if window >= requested:
        return requested
    print(f"  Capping max_model_len {requested} -> {window} (the window of {model_id})")
    return window
