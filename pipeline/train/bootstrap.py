"""Compatibility import for `eigenbench.analysis.pairwise_bootstrap`."""

import sys

from eigenbench.analysis import pairwise_bootstrap

sys.modules[__name__] = pairwise_bootstrap
