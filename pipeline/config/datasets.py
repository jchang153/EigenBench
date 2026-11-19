"""Keep the historical datasets import bound to the active loader."""

import sys

from eigenbench.config import datasets

sys.modules[__name__] = datasets
