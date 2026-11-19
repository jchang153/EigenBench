"""Compatibility import for `eigenbench.analysis.bt_models`."""

import sys

from eigenbench.analysis import bt_models

sys.modules[__name__] = bt_models
