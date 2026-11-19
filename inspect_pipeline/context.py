"""Load the run spec, select data, and prepare an Inspect panel."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from inspect_ai.model import Model, ModelName, get_model

from eigenbench.config import (
    get_criteria_from_spec,
    load_dataset_scenarios_from_spec,
    load_run_spec,
    select_scenarios,
)
from eigenbench.config.airisk import DATASET_ID, DATASET_REVISION
from eigenbench.protocol.direct_rating import (
    build_direct_assignments, resolve_direct_generation_settings, resolve_direct_sampling_settings,
)
from eigenbench.protocol.response_cache import cached_seed

from .generation import FAILURE_POLICIES
from .models import (
    check_vllm_endpoint,
    model_execution,
    model_identity,
    to_inspect_model,
    with_effective_vllm_revision,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_SPEC = "runs/default/spec.py"


def resolve_spec_ref(spec: str) -> str:
    """Resolve a relative spec path against the repo root.

    Inspect runs a task file with its own directory as the working directory,
    so `-T spec=runs/...` would otherwise resolve under inspect_pipeline/.
    Dotted module refs are passed through untouched.
    """

    looks_like_path = spec.endswith(".py") or "/" in spec or "\\" in spec
    if not looks_like_path:
        return spec
    path = Path(spec).expanduser()
    if path.is_absolute() or path.exists():
        return spec
    candidate = _REPO_ROOT / path
    return str(candidate) if candidate.exists() else spec


def load_selection(spec: dict, run_dir: Path):
    """Scenario + criteria selection, shared by every Inspect execution layout."""

    ds = spec["dataset"]
    scenarios = load_dataset_scenarios_from_spec(ds, run_dir=run_dir)
    count = ds.get("count")
    shuffle_seed = ds.get("shuffle_seed")
    selected = select_scenarios(
        scenarios,
        start=int(ds.get("start", 0)),
        count=None if count is None else int(count),
        shuffle=bool(ds.get("shuffle", False)),
        shuffle_seed=None if shuffle_seed is None else int(shuffle_seed),
    )

    constitution = spec["constitution"]
    if "num_criteria" not in constitution:
        raise ValueError(
            "Set constitution.num_criteria in your run spec. "
            "This controls criterion truncation during collection."
        )
    requested = int(constitution["num_criteria"])
    if requested <= 0:
        raise ValueError("constitution.num_criteria must be a positive integer.")
    criteria = get_criteria_from_spec(constitution, run_dir=run_dir)
    if requested < len(criteria):
        criteria = criteria[:requested]
    elif requested > len(criteria):
        raise ValueError(
            f"constitution.num_criteria={requested} exceeds criteria found in "
            f"constitution file ({len(criteria)})."
        )
    return selected, criteria


def _model_resolver(models: dict[str, object], *, memoize: bool = True):
    """Resolve nick -> Model lazily so tasks build without provider API keys."""
    from inspect_ai.model._providers.vllm import VLLMAPI

    # Validate the mapping eagerly; only client creation is deferred.
    refs = {
        nick: (
            value if isinstance(value, Model)
            else with_effective_vllm_revision(to_inspect_model(value))
        )
        for nick, value in models.items()
    }
    cache: dict[str, Model] = {}

    def resolve(nick: str) -> Model:
        model = cache.get(nick)
        if model is None:
            ref = refs[nick]
            if isinstance(ref, Model):
                model = ref
            else:
                model = get_model(ref.name, memoize=memoize, **ref.model_args)
                # Inspect can reuse an unpinned model memoized under earlier env defaults.
                if isinstance(model, Model) and isinstance(model.api, VLLMAPI):
                    revision = ref.model_args.get("revision")
                    if model.api.server_args.get("revision") != revision:
                        raise RuntimeError(
                            f"{ref.name} was resolved with a different vLLM revision than "
                            "requested. Start a fresh Inspect process after changing the defaults."
                        )
            # Provider initialization can replace the adapter name with its base.
            # Capture the requested identity before any generation takes place.
            identity = model_identity(ref)
            if isinstance(ref, Model):
                identity["nick"] = nick
            model._eigenbench_cache_identity = hashlib.sha256(
                json.dumps(identity, sort_keys=True, default=str).encode()
            ).hexdigest()
            cache[nick] = model
        if isinstance(model, Model):
            check_vllm_endpoint(model)
        return model

    return resolve


def _run_context(spec: str, models: dict[str, object] | None, cache: bool | None = None,
                 *, build_assignments: bool = True, resolved_sampler_seed: int | None = None) -> dict:
    run_spec, run_dir = load_run_spec(resolve_spec_ref(spec))
    return _run_context_from_spec(
        run_spec, run_dir, models, cache, build_assignments=build_assignments,
        resolved_sampler_seed=resolved_sampler_seed,
    )


def _run_context_from_spec(
    run_spec: dict,
    run_dir: Path,
    models: dict[str, object] | None,
    cache: bool | None = None,
    *,
    build_assignments: bool = True,
    resolved_sampler_seed: int | None = None,
) -> dict:
    """Everything the task builders need from a run spec."""

    is_direct = run_spec.get("evaluation", {}).get("mode") == "direct_rating"
    spec_models = models if models is not None else run_spec["models"]
    if not spec_models:
        raise ValueError("spec models must not be empty")

    evaluation_cfg = run_spec.get("evaluation", {})
    collection_cfg = run_spec.get("collection", {})
    direct_cfg = evaluation_cfg.get("direct_rating", {}) or {}
    include_self = bool(direct_cfg.get("include_self", True))
    if not include_self and len(spec_models) < 2:
        raise ValueError("include_self=False requires at least two models")
    scale_min = int(direct_cfg.get("scale_min", 1))
    scale_max = int(direct_cfg.get("scale_max", 10))
    if (scale_min, scale_max) != (1, 10):
        raise ValueError("direct rating collection currently uses the fixed 1-10 scale")

    selected, criteria = load_selection(run_spec, run_dir)
    if is_direct:
        generation = resolve_direct_generation_settings(collection_cfg)
    else:
        from eigenbench.protocol.pairwise_plan import resolve_pairwise_generation_settings
        generation = resolve_pairwise_generation_settings(collection_cfg)
    assignments = []
    if is_direct:
        sampling = resolve_direct_sampling_settings(
            collection_cfg, num_models=len(spec_models), include_self=include_self
        )
        if build_assignments:
            plan_sampling = sampling if resolved_sampler_seed is None else {
                **sampling, "sampler_seed": resolved_sampler_seed,
            }
            assignments = build_direct_assignments(
                selected, spec_models, include_self=include_self, **plan_sampling
            )
    else:
        sampling = {"sampler_mode": collection_cfg.get("sampler_mode")}

    inspect_cfg = collection_cfg.get("inspect", {}) or {}
    cache_enabled = bool(inspect_cfg.get("cache", True)) if cache is None else bool(cache)
    max_attempts = int(
        (collection_cfg.get("openrouter", {}) or {}).get(
            "max_attempts", DEFAULT_MAX_ATTEMPTS
        )
    )

    failure_policy = collection_cfg.get("failure_policy", "strict")
    if failure_policy not in FAILURE_POLICIES:
        raise ValueError(
            f"Unknown collection.failure_policy {failure_policy!r}; "
            f"expected one of {', '.join(FAILURE_POLICIES)}"
        )

    seed = cached_seed(collection_cfg.get("cached_responses_path"), selected, spec_models,
                       merge_rows=is_direct)

    return {
        "is_direct": is_direct,
        "run_spec": run_spec,
        "models": spec_models,
        "selected": selected,
        "criteria": criteria,
        "assignments": assignments,
        "generation": generation,
        "sampling": sampling,
        "include_self": include_self,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "cache_enabled": cache_enabled,
        "max_attempts": max_attempts,
        "collection_cfg": collection_cfg,
        "resolve_model": _model_resolver(spec_models),
        "seed": seed,
        "meta": {
            "run_name": run_spec["name"],
            "criteria": criteria,
            "model_order": list(spec_models),
            "include_self": include_self,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "sampler_mode": sampling["sampler_mode"],
            "evaluations_path": collection_cfg.get("evaluations_path"),
            "cached_responses_path": collection_cfg.get("cached_responses_path"),
            "num_scenarios": len(selected),
            "failure_policy": failure_policy,
            "evaluation": evaluation_cfg,
            "panel": [{"nick": nick, **model_identity(value)}
                      for nick, value in spec_models.items()],
            "dataset": {
                **{key: value for key, value in run_spec["dataset"].items() if key != "path"},
                "selected_sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
                **({"source": DATASET_ID, "revision": DATASET_REVISION}
                   if not run_spec["dataset"].get("path")
                   and str(run_spec["dataset"].get("id", "")).strip().lower() == "airisk" else {}),
            },
            "criteria_sha256": hashlib.sha256(json.dumps(criteria).encode()).hexdigest(),
            "sampling": sampling,
            "generation": generation,
            "max_attempts": max_attempts,
            "cache_enabled": cache_enabled,
        },
    }


def has_local_models(models: dict[str, object]) -> bool:
    """Whether any model needs a vLLM server, and so GPU memory."""

    return any(model_execution(value) == "managed_vllm" for value in models.values())


def sanitize(nick: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", nick).strip("_") or "model"


def _eval_model() -> Model | None:
    """Read Inspect's active model, excluding its ``none/none`` placeholder."""

    from inspect_ai.model._model import active_model

    model = active_model()
    if model is None or str(ModelName(model)) == "none/none":
        return None
    return model


def with_eval_model(spec: str, models: dict[str, object] | None) -> dict[str, object] | None:
    """Join the CLI model to the panel, retaining matching nicknames and settings."""

    model = _eval_model()
    if model is None:
        return models
    panel = dict(models if models is not None else load_run_spec(resolve_spec_ref(spec))[0]["models"])
    name = str(ModelName(model))
    for nick, value in panel.items():
        if value is model:
            return panel
        if not isinstance(value, Model):
            ref = to_inspect_model(value)
            if ref.name == name:
                if ref.model_args and ref.model_args != model.model_args:
                    raise ValueError(
                        f"--model {name} conflicts with the settings of panel member {nick!r}. "
                        "Configure this model in the spec and omit --model, or use matching model arguments."
                    )
                panel[nick] = model
                return panel
    nick = name if name not in panel else f"{name} (--model)"
    suffix = 2
    while nick in panel:
        nick = f"{name} (--model {suffix})"
        suffix += 1
    return {**panel, nick: model}
