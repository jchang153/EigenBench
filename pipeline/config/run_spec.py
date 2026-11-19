"""Compatibility imports for the run-spec loader.

Allow consumers to load this file directly from a checkout without installing the package.
"""

if __package__:
    from eigenbench.config import run_spec as _run_spec
else:
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "eigenbench" / "config" / "run_spec.py"
    spec = spec_from_file_location("_eigenbench_run_spec", source)
    _run_spec = module_from_spec(spec)
    spec.loader.exec_module(_run_spec)

DIRECT_SAMPLER_ALIASES = _run_spec.DIRECT_SAMPLER_ALIASES
DIRECT_SAMPLER_ALL_TO_ALL = _run_spec.DIRECT_SAMPLER_ALL_TO_ALL
DIRECT_SAMPLER_BALANCED = _run_spec.DIRECT_SAMPLER_BALANCED
DIRECT_SAMPLER_PARTITIONED = _run_spec.DIRECT_SAMPLER_PARTITIONED
normalize_direct_sampling_settings = _run_spec.normalize_direct_sampling_settings
validate_direct_sampling_panel = _run_spec.validate_direct_sampling_panel
_resolve_path_for_run = _run_spec._resolve_path_for_run
infer_run_name_and_dir = _run_spec.infer_run_name_and_dir
apply_run_defaults = _run_spec.apply_run_defaults
load_run_spec = _run_spec.load_run_spec
