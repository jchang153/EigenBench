"""Compatibility import for `eigenbench.analysis.fit`."""

import sys

from eigenbench.analysis import fit

sys.modules[__name__] = fit
