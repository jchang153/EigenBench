"""Keep the saved-run analysis entry point used by ValueArena."""

import sys

if __name__ == "__main__":
    raise SystemExit(
        "run_train.py is an internal stage. "
        "Use: python scripts/run.py <spec_module_or_path>"
    )

from eigenbench import score

sys.modules[__name__] = score
