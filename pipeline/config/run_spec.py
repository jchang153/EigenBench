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
