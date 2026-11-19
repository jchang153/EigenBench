"""Upload EigenBench run results to a Hugging Face dataset."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eigenbench import publish


if __name__ == "__main__":
    publish.main()
else:
    sys.modules[__name__] = publish
