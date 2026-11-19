"""Keep the historical airisk import bound to the active loader."""

import sys

from eigenbench.config import airisk

sys.modules[__name__] = airisk
