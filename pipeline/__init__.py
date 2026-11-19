"""Compatibility imports for the former EigenBench pipeline package."""

import sys
from pathlib import Path

# Existing Python callers may import this API from the native task directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[:1] != [str(_REPO_ROOT)] and Path.cwd() == _REPO_ROOT / "inspect_pipeline":
    sys.path.insert(0, str(_REPO_ROOT))
