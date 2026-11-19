"""Resolve run-spec generation settings.

ValueArena reads the resolver's definition from the legacy file. Keep using
that definition until the worker can follow its new location.
"""

from pipeline.protocol.direct_rating import resolve_direct_generation_settings as resolve_direct_generation_settings
