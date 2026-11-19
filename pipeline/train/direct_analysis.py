"""Compatibility import for `eigenbench.analysis.direct_rating`."""

import sys

from eigenbench.analysis import direct_rating

# ValueArena patches this module's bootstrap plotter before running the analysis.
sys.modules[__name__] = direct_rating
