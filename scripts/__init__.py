"""Command entry points and legacy runner imports."""

import sys
from pathlib import Path

# Preserve Python entry points imported from the native task directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[:1] != [str(_REPO_ROOT)] and Path.cwd() == _REPO_ROOT / "inspect_pipeline":
    sys.path.insert(0, str(_REPO_ROOT))
