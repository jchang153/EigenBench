"""The collection module patched by the ValueArena worker."""

import sys

from .collection import direct_rating

sys.modules[__name__] = direct_rating
