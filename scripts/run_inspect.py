"""Alias for the EigenBench runner; all collection uses Inspect.

    python scripts/run_inspect.py runs/my_run/spec.py [--estimate-calls]
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run import cli, main


if __name__ == "__main__":
    cli()
