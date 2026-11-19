"""Backward-compatible module path for eigenbench.inspect.export."""

import sys

from eigenbench.inspect import export

sys.modules[__name__] = export
