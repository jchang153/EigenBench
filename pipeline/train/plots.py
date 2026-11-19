"""Compatibility import for `eigenbench.analysis.plots`."""

import sys

from eigenbench.analysis import plots

sys.modules[__name__] = plots
