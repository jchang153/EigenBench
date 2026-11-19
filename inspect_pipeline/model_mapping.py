"""Backward-compatible module path for eigenbench.inspect.models."""

import sys

from eigenbench.inspect import models

sys.modules[__name__] = models
