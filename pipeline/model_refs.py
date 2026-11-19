"""Backward-compatible module path for eigenbench.inspect.model_refs."""

import sys

from eigenbench.inspect import model_refs

sys.modules[__name__] = model_refs
