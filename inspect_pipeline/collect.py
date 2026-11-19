"""Compatibility module for eigenbench.inspect.collection.direct_rating."""

import sys

from eigenbench.inspect.collection import direct_rating

sys.modules[__name__] = direct_rating
