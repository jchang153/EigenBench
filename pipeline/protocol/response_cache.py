"""Keep the historical response-cache import bound to the active reader."""

import sys

from eigenbench.protocol import response_cache

sys.modules[__name__] = response_cache
