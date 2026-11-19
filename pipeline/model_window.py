"""Backward-compatible module path for eigenbench.inspect.model_window."""

import sys

from eigenbench.inspect import model_window

sys.modules[__name__] = model_window
