"""Run-spec helpers for folder-based defaults."""

from __future__ import annotations

import copy
import importlib
import importlib.util
from pathlib import Path
import types

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path_for_run(path_value: str | None, run_dir: Path, default_name: str) -> str:
    if not path_value:
        return str((run_dir / default_name).resolve())
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return str(p)
    return str((run_dir / p).resolve())


def _resolve_optional_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return str(p)
    return str((_REPO_ROOT / p).resolve())


def infer_run_name_and_dir(spec_ref: str, module_file: str, spec: dict) -> tuple[str, Path]:
    module_path = Path(module_file).resolve()

    # Package layout (recommended): runs/<name>/__init__.py or runs/<name>/spec.py
    if module_path.name in {"__init__.py", "config.py", "spec.py"}:
        fallback_name = module_path.parent.name
        run_dir = module_path.parent
    else:
        fallback_name = module_path.stem
        # Legacy single-file layout: runs/<name>.py -> place outputs in runs/<name>/
        if module_path.parent.name == "runs":
            run_dir = module_path.parent / fallback_name
        else:
            run_dir = module_path.parent

    run_name = str(spec.get("name") or fallback_name)
    if module_path.name not in {"__init__.py", "config.py", "spec.py"} and module_path.parent.name == "runs":
        run_dir = module_path.parent / run_name

    return run_name, run_dir


def apply_run_defaults(spec_ref: str, module_file: str, spec: dict) -> tuple[dict, Path]:
    """Return a normalized spec with run-folder defaults applied."""

    normalized = copy.deepcopy(spec)
    run_name, run_dir = infer_run_name_and_dir(spec_ref, module_file, normalized)

    normalized["name"] = run_name
    normalized["verbose"] = bool(normalized.get("verbose", False))

    evaluation = normalized.setdefault("evaluation", {})
    mode = str(evaluation.get("mode", "pairwise_btd")).strip().lower()
    aliases = {
        "pairwise": "pairwise_btd",
        "btd": "pairwise_btd",
        "direct": "direct_rating",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"pairwise_btd", "direct_rating"}:
        raise ValueError(
            "evaluation.mode must be 'pairwise_btd' or 'direct_rating'; "
            f"got {evaluation.get('mode')!r}"
        )
    evaluation["mode"] = mode

    if mode == "direct_rating":
        direct = evaluation.setdefault("direct_rating", {})
        direct["include_self"] = bool(direct.get("include_self", True))
        direct["scale_min"] = int(direct.get("scale_min", 1))
        direct["scale_max"] = int(direct.get("scale_max", 10))
        direct["criterion_aggregation"] = str(
            direct.get("criterion_aggregation", "mean")
        ).strip().lower()
        direct["scenario_aggregation"] = str(
            direct.get("scenario_aggregation", "mean")
        ).strip().lower()
        direct["normalization"] = str(
            direct.get("normalization", "zscore_softmax")
        ).strip().lower()
        supported_normalizations = {
            "zscore_softmax",
            "rank_softmax",
            "raw_l1",
            "minmax_l1",
            "positive_centered_l1",
        }
        if direct["normalization"] not in supported_normalizations:
            raise ValueError(
                "direct_rating.normalization must be one of "
                f"{sorted(supported_normalizations)}"
            )
        direct["softmax_temperature"] = float(
            direct.get("softmax_temperature", 1.0)
        )
        if direct["scale_min"] >= direct["scale_max"]:
            raise ValueError("direct_rating.scale_min must be less than scale_max")
        if (direct["scale_min"], direct["scale_max"]) != (1, 10):
            raise ValueError("direct_rating currently uses the fixed 1-10 rating scale")
        if direct["criterion_aggregation"] != "mean":
            raise ValueError("direct_rating.criterion_aggregation currently supports only 'mean'")
        if direct["scenario_aggregation"] != "mean":
            raise ValueError("direct_rating.scenario_aggregation currently supports only 'mean'")
        if direct["softmax_temperature"] <= 0:
            raise ValueError("direct_rating.softmax_temperature must be positive")
        direct["eigentrust_alpha"] = float(direct.get("eigentrust_alpha", 0.0))
        if not 0.0 <= direct["eigentrust_alpha"] <= 1.0:
            raise ValueError("direct_rating.eigentrust_alpha must be between 0 and 1")

    collection = normalized.setdefault("collection", {})
    collection["evaluations_path"] = _resolve_path_for_run(
        collection.get("evaluations_path"),
        run_dir,
        "evaluations.jsonl",
    )
    cache_path = _resolve_optional_path(collection.get("cached_responses_path"))
    if cache_path is not None:
        collection["cached_responses_path"] = cache_path
    elif "cached_responses_path" in collection:
        # Keep explicit null if user sets it intentionally.
        collection["cached_responses_path"] = None

    if mode == "direct_rating":
        sampler_aliases = {
            "exhaustive": "all_to_all",
            "all_to_all": "all_to_all",
            "partitioned": "partitioned_random_judge",
            "random_partition": "partitioned_random_judge",
            "partitioned_random_judge": "partitioned_random_judge",
        }
        raw_sampler = str(collection.get("sampler_mode", "all_to_all")).strip().lower()
        sampler_mode = sampler_aliases.get(raw_sampler)
        if sampler_mode is None:
            raise ValueError(
                "direct collection.sampler_mode must be 'all_to_all' or "
                "'partitioned_random_judge'"
            )
        collection["sampler_mode"] = sampler_mode
        collection["group_size"] = int(collection.get("group_size", 4))
        collection["response_redundancy"] = int(
            collection.get("response_redundancy", 1)
        )
        if collection["group_size"] <= 0:
            raise ValueError("direct collection.group_size must be positive")
        if collection["response_redundancy"] <= 0:
            raise ValueError("direct collection.response_redundancy must be positive")
        raw_seed = collection.get("sampler_seed", 42)
        collection["sampler_seed"] = None if raw_seed is None else int(raw_seed)
        num_models = len(normalized.get("models", {}))
        if num_models:
            max_redundancy = num_models if direct["include_self"] else num_models - 1
            if collection["response_redundancy"] > max_redundancy:
                raise ValueError(
                    "direct collection.response_redundancy exceeds the number of "
                    "distinct eligible judges"
                )
            if (
                sampler_mode == "partitioned_random_judge"
                and not direct["include_self"]
                and collection["group_size"] >= num_models
            ):
                raise ValueError(
                    "partitioned direct sampling with include_self=False requires "
                    "collection.group_size < number of models"
                )

    training = normalized.setdefault("training", {})
    if training.get("output_dir"):
        training["output_dir"] = _resolve_path_for_run(
            training.get("output_dir"),
            run_dir,
            ".",
        )
    else:
        # Default: write btd_d* folders directly under runs/<run_name>/.
        training["output_dir"] = str(run_dir.resolve())

    return normalized, run_dir


def _is_probable_path(spec_ref: str) -> bool:
    return spec_ref.endswith(".py") or "/" in spec_ref or "\\" in spec_ref


def _load_module_from_path(path_ref: str) -> tuple[types.ModuleType, str]:
    path = Path(path_ref).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Run spec path does not exist: {path}")

    module_name = f"_eigenbench_run_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load run spec from path: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, str(path)


def load_run_spec(spec_ref: str) -> tuple[dict, Path]:
    """Load RUN_SPEC from either dotted module path or python file path.

    Examples:
    - runs.example.spec
    - runs/my_run/spec.py
    """

    if _is_probable_path(spec_ref):
        module, ref = _load_module_from_path(spec_ref)
    else:
        try:
            module = importlib.import_module(spec_ref)
            ref = spec_ref
        except ModuleNotFoundError:
            # Convenience: allow "runs.my_run" to resolve to "runs.my_run.spec".
            module = importlib.import_module(f"{spec_ref}.spec")
            ref = f"{spec_ref}.spec"

    if not hasattr(module, "RUN_SPEC"):
        # Convenience: if caller passed "runs.my_run", resolve via "runs.my_run.spec".
        if not _is_probable_path(spec_ref) and not spec_ref.endswith(".spec"):
            module = importlib.import_module(f"{spec_ref}.spec")
            ref = f"{spec_ref}.spec"
        if not hasattr(module, "RUN_SPEC"):
            raise AttributeError(f"RUN_SPEC not found in run spec: {spec_ref}")

    return apply_run_defaults(ref, module.__file__, module.RUN_SPEC)
