"""Native Inspect tasks, collection, and log export for EigenBench."""

__all__ = ["eigenbench", "collect_direct_ratings_inspect", "export_log"]


def __getattr__(name):
    if name == "eigenbench":
        from .eigenbench import eigenbench

        return eigenbench
    if name == "collect_direct_ratings_inspect":
        from .collection.direct_rating import collect_direct_ratings_inspect

        return collect_direct_ratings_inspect
    if name == "export_log":
        from .export import export_log

        return export_log
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
