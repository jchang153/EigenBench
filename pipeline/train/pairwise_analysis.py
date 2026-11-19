"""Compatibility import for `eigenbench.analysis.pairwise`."""

import sys

from eigenbench.analysis import pairwise

sys.modules[__name__] = pairwise
