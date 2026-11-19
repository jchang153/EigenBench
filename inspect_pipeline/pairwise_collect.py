"""Compatibility module for eigenbench.inspect.collection.pairwise."""

import sys

from eigenbench.inspect.collection import pairwise

sys.modules[__name__] = pairwise
