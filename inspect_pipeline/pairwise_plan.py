"""Compatibility module for eigenbench.protocol.pairwise_plan."""

import sys

from eigenbench.protocol import pairwise_plan

sys.modules[__name__] = pairwise_plan
