"""Compatibility entry points for the native Inspect tasks.

The Inspect task lives in ``inspect_pipeline.eigenbench``:

    inspect eval inspect_pipeline/eigenbench.py -T spec=runs/my_run/spec.py

The implementation lives in ``eigenbench.inspect``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# A session started in this directory can import the compatibility package
# before loading any task file. Prefer the package over the local eigenbench.py.
_TASK_DIR = Path(__file__).resolve().parent
if sys.path[:1] != [str(_TASK_DIR.parent)] and Path.cwd() == _TASK_DIR:
    sys.path.insert(0, str(_TASK_DIR.parent))

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
