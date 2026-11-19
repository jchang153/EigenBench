"""Run collection, analysis, and publication from a Python run spec."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eigenbench import run


if __name__ == "__main__":
    run.cli()
else:
    sys.modules[__name__] = run
