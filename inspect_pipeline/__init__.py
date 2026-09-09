"""Inspect AI collection engine for EigenBench direct-rating runs.

The Inspect task lives in ``inspect_pipeline.eigenbench``:

    inspect eval inspect_pipeline/eigenbench.py -T spec=runs/my_run/spec.py

The protocol itself — sampling plans, prompts, rating validation — is imported
unchanged from ``pipeline.eval.direct_rating``, and the exported output is the
same ``evaluations.jsonl`` consumed by ``pipeline.train.direct_analysis``.
"""

from __future__ import annotations

__all__ = ["eigenbench", "collect_direct_ratings_inspect", "export_log"]


def __getattr__(name):
    if name == "eigenbench":
        from .eigenbench import eigenbench

        return eigenbench
    if name == "collect_direct_ratings_inspect":
        from .collect import collect_direct_ratings_inspect

        return collect_direct_ratings_inspect
    if name == "export_log":
        from .export import export_log

        return export_log
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
