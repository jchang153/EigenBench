"""Compatibility imports for direct rating and ValueArena's generation settings."""

from eigenbench.protocol.direct_rating import (
    DIRECT_PROMPT_VERSION as DIRECT_PROMPT_VERSION,
    DIRECT_SAMPLER_ALIASES as DIRECT_SAMPLER_ALIASES,
    DIRECT_SAMPLER_ALL_TO_ALL as DIRECT_SAMPLER_ALL_TO_ALL,
    DIRECT_SAMPLER_BALANCED as DIRECT_SAMPLER_BALANCED,
    DIRECT_SAMPLER_PARTITIONED as DIRECT_SAMPLER_PARTITIONED,
    build_direct_assignments as build_direct_assignments,
    build_direct_rating_prompt as build_direct_rating_prompt,
    build_direct_rating_user_prompt as build_direct_rating_user_prompt,
    build_direct_reflection_prompt as build_direct_reflection_prompt,
    build_direct_reflection_user_prompt as build_direct_reflection_user_prompt,
    direct_rating_validator as direct_rating_validator,
    estimate_direct_calls as estimate_direct_calls,
    normalize_direct_sampling_settings as normalize_direct_sampling_settings,
    parse_direct_ratings as parse_direct_ratings,
    resolve_direct_sampling_settings as resolve_direct_sampling_settings,
    validate_direct_sampling_panel as validate_direct_sampling_panel,
)


def resolve_direct_generation_settings(
    collection_cfg: dict | None,
    *,
    generation_key: str = "generation",
    defaults: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Resolve phase settings, with an explicit key and defaults for pairwise callers."""
    # ValueArena extracts this function from the source without importing the module.
    import math

    collection_cfg = collection_cfg or {}
    allow_token_strings = defaults is None
    if defaults is None:
        defaults = {
            "response": {"max_tokens": collection_cfg.get("max_tokens", 4096), "temperature": 0.7},
            "reflection": {"max_tokens": collection_cfg.get("max_tokens", 2048), "temperature": 0.2},
            "direct_rating": {"max_tokens": collection_cfg.get("max_tokens", 512), "temperature": 0.0},
        }

    def mapping(value, path: str) -> dict:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping")
        return value

    def checked(values: dict, path: str) -> dict:
        raw_tokens = values["max_tokens"]
        try:
            tokens = int(raw_tokens)
            integer = (allow_token_strings and isinstance(raw_tokens, str)) or tokens == raw_tokens
            if isinstance(raw_tokens, bool) or not integer or tokens <= 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"invalid generation {path}.max_tokens: expected a positive integer, got {raw_tokens!r}"
            ) from None

        raw_temperature = values["temperature"]
        try:
            temperature = float(raw_temperature)
            if not math.isfinite(temperature) or temperature < 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"invalid generation {path}.temperature: expected a finite non-negative number, "
                f"got {raw_temperature!r}"
            ) from None
        return {"max_tokens": tokens, "temperature": temperature}

    path = f"collection.{generation_key}"
    configured = mapping(collection_cfg.get(generation_key), path)
    unknown = configured.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"{path} has unknown phases {sorted(unknown, key=str)}; expected {list(defaults)}")

    resolved = {}
    for phase, phase_defaults in defaults.items():
        phase_path = f"{path}.{phase}"
        phase_config = mapping(configured.get(phase), phase_path)
        values = checked({**phase_defaults, **phase_config}, phase_path)
        overrides_path = f"{phase_path}.per_model"
        overrides = mapping(phase_config.get("per_model"), overrides_path)
        values["per_model"] = {}
        for nick, override in overrides.items():
            override_path = f"{overrides_path}.{nick}"
            if not isinstance(override, dict):
                raise ValueError(f"{override_path} must be a mapping")
            values["per_model"][nick] = checked({**values, **override}, override_path)
        resolved[phase] = values
    return resolved
